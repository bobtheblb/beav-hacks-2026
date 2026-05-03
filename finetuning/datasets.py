"""Self-contained MMFR dataset loader, shaped like nemo_automodel's make_cord_v2_dataset.

The function name `make_cord_v2_dataset` is preserved per the project's YAML config;
it actually loads the MMFR fake/real subset.

Pipeline:
    1. Download shards from AnnaGao/MMFR-Dataset on demand via huggingface_hub
       (cached under HF_HOME). Always download dataset.part1.tar first because it
       contains forgery_reasoning_cot.json (the per-image CoT index).
    2. For each ordered shard (FAKE_SHARDS / REAL_SHARDS), list the images present
       and pull bytes for entries the CoT JSON has reasoning for. Stop once we
       have enough rows.
    3. Format each row into a {"conversation": [user, assistant]} sample.

Output format the assistant is trained to produce:

    <think>
    {per-image reasoning, pulled from MMFR's forgery_reasoning_cot.json}
    </think>

    ```json
    {"status": "ai_generated"|"real", "confidence": 1.0, "reason": "..."}
    ```
"""

import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor

# Per-batch debug prints fire on the first N batches when MMFR_DEBUG=1.
# Off by default so training logs stay clean.
MMFR_DEBUG = os.environ.get("MMFR_DEBUG", "").lower() in ("1", "true", "yes")

# When MMFR_SKIP_OPTIM_LOAD=1, monkey-patch Checkpointer.load_optimizer to a no-op.
# Use this to recover from a corrupt optimizer state in a checkpoint: model weights
# load fine, but the AdamW state has missing per-parameter keys (typically from a
# kill mid-save). The optimizer restarts fresh; AdamW momentum rebuilds in ~hundreds
# of steps. Run once with the env var, then unset for normal subsequent runs.
if os.environ.get("MMFR_SKIP_OPTIM_LOAD", "").lower() in ("1", "true", "yes"):
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

    def _skip_load_optimizer(self, *args, **kwargs):
        print("[shim] MMFR_SKIP_OPTIM_LOAD: skipping optimizer state load (model weights still loaded)", flush=True)

    Checkpointer.load_optimizer = _skip_load_optimizer


def load_processor_with_pad(pretrained_model_name_or_path, **kwargs):
    """AutoProcessor.from_pretrained, then set pad_token = eos_token if missing.

    Some VLM tokenizers (incl. Nemotron-Nano-VL) ship without a pad_token,
    which breaks the padding step of the default VLM collate fn.
    """
    proc = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
    tok = getattr(proc, "tokenizer", proc)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    return proc


def load_nemotron_vl_model(pretrained_model_name_or_path, **kwargs):
    """NeMoAutoModelForCausalLM.from_pretrained, then re-fix RADIO's `summary_idxs`.

    The C-RADIOv2-H vision tower computes `summary_idxs = [i for i,t in enumerate(args.teachers)
    if t.get('use_summary', True)]` in __init__, then later does
    `bb_summary = all_summary[:, self.summary_idxs]`. transformers' loader logs
    `summary_idxs` as MISSING and may overwrite the constructor's tensor with
    random ints that index out of bounds — CUDA assertion in the vision tower.
    We rebuild the correct buffer post-load.
    """
    import torch
    from nemo_automodel import NeMoAutoModelForCausalLM

    model = NeMoAutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)

    radio = getattr(getattr(model, "vision_model", None), "radio_model", None)
    if radio is not None:
        teachers = getattr(getattr(radio, "config", None), "args", {}).get("teachers")
        if teachers is None:
            # Fall back to the model's vision_config args
            vc = getattr(model.config, "vision_config", None)
            if vc is not None:
                teachers = getattr(vc, "args", {}).get("teachers", [])
        idxs = [i for i, t in enumerate(teachers or []) if t.get("use_summary", True)]
        if idxs:
            target = next(radio.parameters(), None)
            device = target.device if target is not None else torch.device("cpu")
            new_buf = torch.tensor(idxs, dtype=torch.int64, device=device)
            existing = getattr(radio, "summary_idxs", None)
            if existing is None or not torch.equal(existing.cpu(), new_buf.cpu()):
                # Re-register the buffer so tensor indexing in radio_model.forward works.
                if hasattr(radio, "summary_idxs") and "summary_idxs" in radio._buffers:
                    del radio._buffers["summary_idxs"]
                radio.register_buffer("summary_idxs", new_buf, persistent=False)
                print(f"[load_nemotron_vl_model] reset summary_idxs to {idxs}")
    return model


_DEBUG_BATCH_LOGS = 3 if MMFR_DEBUG else 0
_debug_batches_logged = 0


def nemotron_vl_collate_fn(examples, processor, **kwargs):
    """Wraps the default VLM collate fn and adds Nemotron-Nano-VL's `image_flags`.

    The model's forward() does `image_flags.squeeze(-1)` and uses it to mask
    `vit_embeds[image_flags == 1]`, so it must have shape (num_tiles, 1) and
    match `pixel_values.shape[0]`. Every sample in our dataset has a real image,
    so all flags are 1.
    """
    global _debug_batches_logged
    import torch
    from nemo_automodel.components.datasets.vlm import collate_fns as _cfns

    batch = _cfns.default_collate_fn(examples, processor, **kwargs)

    # Nemotron-Nano-VL's tokenizer doesn't register <|im_start|>/<|im_end|> as
    # single special tokens (they BPE-split), so neither the im_start marker scan
    # nor the legacy text-pattern-match build_labels path can locate assistant
    # turns reliably. We rebuild labels via a prefix-diff: re-render each
    # conversation up to the assistant turn (add_generation_prompt=True), use
    # the resulting length to mark where assistant content begins, and unmask
    # only positions past that index.
    conversations = [ex["conversation"] for ex in examples]
    new_labels = torch.full_like(batch["labels"], -100)
    seq_len = batch["input_ids"].shape[1]

    for i, conv in enumerate(conversations):
        prefix_conv = [m for m in conv if m.get("role") != "assistant"]
        try:
            prefix_batch = processor.apply_chat_template(
                [prefix_conv],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=False,
                return_dict=True,
            )
            prefix_len = prefix_batch["input_ids"].shape[1]
        except Exception as e:
            print(f"[label-debug] prefix render failed (sample {i}): {e}")
            continue

        # batch is shifted: input_ids[t] = full[t]; labels[t] should be full[t+1].
        # We want labels[t] != -100 only when t+1 >= prefix_len (assistant region).
        unmask_start = max(0, prefix_len - 1)
        unmask_end = seq_len - 1  # last position has no next-token in input_ids
        if unmask_start < unmask_end:
            new_labels[i, unmask_start:unmask_end] = batch["input_ids"][i, unmask_start + 1:unmask_end + 1]
            # Don't supervise on padding.
            attn = batch["attention_mask"][i]
            pad_mask = attn[unmask_start + 1:unmask_end + 1] == 0
            new_labels[i, unmask_start:unmask_end][pad_mask] = -100

    batch["labels"] = new_labels
    if _debug_batches_logged < _DEBUG_BATCH_LOGS:
        print(f"[label-debug] num_label_tokens (post-rebuild): {(new_labels != -100).sum().item()}")

    if "pixel_values" in batch:
        n = batch["pixel_values"].shape[0]
        batch["image_flags"] = torch.ones(n, 1, dtype=torch.long, device=batch["pixel_values"].device)

    if _debug_batches_logged < _DEBUG_BATCH_LOGS:
        shapes = {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__ for k, v in batch.items()}
        # Find the image context token id. The Nemotron-Nano-VL processor exposes it as
        # `image_token` (the literal token string) on the processor itself.
        tok = getattr(processor, "tokenizer", processor)
        cand_strs = []
        for attr in ("image_token", "img_context_token"):
            v = getattr(processor, attr, None)
            if isinstance(v, str):
                cand_strs.append(v)
        cand_strs += ["<image>", "<|image|>", "<img>"]
        token_counts = {}
        for s in set(cand_strs):
            try:
                tid = tok.convert_tokens_to_ids(s)
            except Exception:
                continue
            if tid is None or tid == getattr(tok, "unk_token_id", None):
                continue
            token_counts[s] = (tid, (batch["input_ids"] == tid).sum().item())
        # vit_embeds will have shape (5_tiles, num_patches_post_shuffle, C); the
        # model writes one row per (tile, patch) into inputs_embeds[selected].
        expected_per_tile = "?"
        cfg = getattr(processor, "image_processor", None)
        if cfg is not None:
            expected_per_tile = getattr(cfg, "num_image_token", "?")
        n_tiles = batch["pixel_values"].shape[0] if "pixel_values" in batch else 0
        print(
            f"[collate-debug] shapes={shapes}\n"
            f"[collate-debug] candidate img-token counts in input_ids: {token_counts}\n"
            f"[collate-debug] tiles={n_tiles}  num_image_token_per_tile={expected_per_tile}  "
            f"expected total img tokens = {n_tiles * (expected_per_tile if isinstance(expected_per_tile, int) else 0)}"
        )
        _debug_batches_logged += 1

    return batch

REPO_ID = "AnnaGao/MMFR-Dataset"
COT_JSON_TARPATH = "dataset/forgery_reasoning_cot.json"

# Shard ranges discovered empirically:
#   parts 1-69  -> diffusiondb (fake)         — part1 also carries the CoT JSON
#   parts 70-79 -> evaluation_sets (skip)
#   parts 80-86 -> laion (real)
FAKE_SHARDS = [f"dataset.part{i}.tar" for i in range(1, 70)]
REAL_SHARDS = [f"dataset.part{i}.tar" for i in range(80, 87)]

FAKE_JSON_PREFIX = "diffusiondb/"
REAL_JSON_PREFIX = "laion/"

USER_PROMPT = (
    "Analyze the provided image and determine whether it is AI-generated.\n"
    "Think through the visual cues, then respond with a JSON object with these fields:\n"
    "  - status: either \"ai_generated\" or \"real\"\n"
    "  - confidence: a number between 0 and 1\n"
    "  - reason: a brief explanation of your decision"
)

_REASONING_RE = re.compile(r"<REASONING>(.*?)</REASONING>", re.DOTALL)

# Module-level cache so the train and validation calls share the gather work.
# Keyed on the total rows-per-class needed; values are (fake_rows, real_rows).
_GATHER_CACHE: dict[int, tuple[list[dict], list[dict]]] = {}


def _extract_member(tar_path: str, member: str) -> bytes | None:
    """Stream a single tar member's bytes via system tar (tolerant of MMFR's tar prefix)."""
    proc = subprocess.run(
        ["tar", "-xOf", tar_path, member],
        capture_output=True, check=False,
    )
    return proc.stdout or None


def _short_reason(reasoning: str, max_chars: int = 240) -> str:
    text = reasoning.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(". ")
    return (cut[: last_period + 1] if last_period > 50 else cut).strip()


def _gather_across_shards(cot_filtered, shard_list, json_prefix, total_needed, label):
    """Walk shards in order, bulk-extracting matching images per shard, until total_needed is hit.

    For each shard, runs ONE `tar -x --wildcards 'dataset/<prefix>*'` invocation
    into a temp directory, then reads the matching files. This is ~100x faster
    than the previous one-subprocess-per-image approach.
    """
    cot_by_image = {e["image"]: e for e in cot_filtered}
    rows = []
    for shard in shard_list:
        if len(rows) >= total_needed:
            break
        try:
            tar_path = hf_hub_download(repo_id=REPO_ID, filename=shard, repo_type="dataset")
        except Exception as e:
            print(f"[warn] couldn't download {shard}: {e}")
            continue

        added = 0
        with tempfile.TemporaryDirectory(prefix="mmfr_") as tmp:
            subprocess.run(
                [
                    "tar", "-xf", tar_path, "-C", tmp,
                    "--wildcards", "--no-anchored",
                    f"dataset/{json_prefix}*",
                ],
                check=False, capture_output=True,
            )
            base = Path(tmp) / "dataset"
            if not base.exists():
                print(f"[shard {shard}] no matching files extracted")
                continue
            # Walk the extracted tree, decode files for which we have CoT entries.
            for path in base.rglob("*"):
                if len(rows) >= total_needed:
                    break
                if not path.is_file():
                    continue
                if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                img_rel = path.relative_to(base).as_posix()
                entry = cot_by_image.get(img_rel)
                if entry is None:
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    continue
                gpt_resp = entry["conversations"][1]["value"]
                m = _REASONING_RE.search(gpt_resp)
                reasoning = m.group(1).strip() if m else gpt_resp.strip()
                rows.append({"image": img, "label": label, "reasoning": reasoning})
                added += 1
        print(f"[shard {shard}] +{added} {('fake' if label == 1 else 'real')} rows  (total {len(rows)})")
    return rows


def make_cord_v2_dataset(
    path_or_dataset=REPO_ID,
    split="train",
    n_train_per_class=100,
    n_val_per_class=20,
    **kwargs,
):
    """Load and preprocess the MMFR fake/real subset for image+text VLM fine-tuning.

    Train and validation slices are disjoint: validation takes the first
    ``n_val_per_class`` rows of each class; training takes the next
    ``n_train_per_class``. Shards are downloaded on demand until enough rows
    are collected (cap depends on dataset size — currently up to ~50k per class).

    If the requested split's count is 0 we short-circuit and return ``[]``
    without touching the network — useful when you only want one split.
    """
    if split == "train" and n_train_per_class == 0:
        return []
    if split in ("validation", "val", "test") and n_val_per_class == 0:
        return []

    total = n_train_per_class + n_val_per_class

    if total in _GATHER_CACHE:
        fake_rows, real_rows = _GATHER_CACHE[total]
        print(f"[cache] reusing gathered rows for total={total} per class")
    else:
        # part1.tar always needed for the CoT JSON
        part1 = hf_hub_download(repo_id=path_or_dataset, filename="dataset.part1.tar", repo_type="dataset")
        cot_blob = _extract_member(part1, COT_JSON_TARPATH)
        if not cot_blob:
            raise RuntimeError(f"Could not extract {COT_JSON_TARPATH} from {part1}")
        cot = json.loads(cot_blob)

        cot_fake = [e for e in cot if e["image"].startswith(FAKE_JSON_PREFIX)]
        cot_real = [e for e in cot if e["image"].startswith(REAL_JSON_PREFIX)]
        print(f"[cot] {len(cot_fake)} fake / {len(cot_real)} real candidate entries")

        fake_rows = _gather_across_shards(cot_fake, FAKE_SHARDS, FAKE_JSON_PREFIX, total, label=1)
        real_rows = _gather_across_shards(cot_real, REAL_SHARDS, REAL_JSON_PREFIX, total, label=0)
        _GATHER_CACHE[total] = (fake_rows, real_rows)

    if split == "train":
        dataset = fake_rows[n_val_per_class:] + real_rows[n_val_per_class:]
    elif split in ("validation", "val", "test"):
        dataset = fake_rows[:n_val_per_class] + real_rows[:n_val_per_class]
    else:
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'validation'.")

    def format(example):
        obj = {
            "status": "ai_generated" if example["label"] == 1 else "real",
            "confidence": 1.0,
            "reason": _short_reason(example["reasoning"]),
        }
        assistant_text = (
            f"<think>\n{example['reasoning'].strip()}\n</think>\n\n"
            f"```json\n{json.dumps(obj, indent=2)}\n```"
        )
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
            ],
        }

    return [format(example) for example in dataset]
