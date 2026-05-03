"""Standalone repro for the Nemotron-Nano-12B-VL forward-pass CUDA assertion.

Loads the model + processor manually, builds one MMFR sample, and walks the
model's forward step-by-step with prints so we can pinpoint the failing op.

Run:
    CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=. \
        conda run -n beav python finetuning/debug_nemotron.py
"""

import os
import sys
import traceback

import torch
from transformers import AutoModelForCausalLM

from finetuning.datasets import (
    load_processor_with_pad,
    make_cord_v2_dataset,
    nemotron_vl_collate_fn,
)

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"


def header(label):
    print(f"\n{'=' * 8} {label} {'=' * 8}")


def desc(name, t):
    if isinstance(t, torch.Tensor):
        extra = f" min={t.min().item() if t.is_floating_point() or t.dtype == torch.long else 'n/a'}"
        if t.dtype == torch.long:
            extra = f" min={t.min().item()} max={t.max().item()}"
        print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}{extra}")
    else:
        print(f"  {name}: {type(t).__name__}")


def main():
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    header("Load processor")
    processor = load_processor_with_pad(MODEL_ID, trust_remote_code=True)
    tok = processor.tokenizer
    print(f"  vocab_size: {tok.vocab_size}  pad_token={tok.pad_token!r} pad_id={tok.pad_token_id}")
    img_id = tok.convert_tokens_to_ids(getattr(processor, "image_token", "<image>"))
    print(f"  image_token id: {img_id}")

    header("Load model (bf16) — patching flash_attn_2 support flag")
    # Force-enable flash_attn_2 support on the dynamically-loaded model class.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module(
        "modeling.NemotronH_Nano_Omni_Reasoning_V3" if False else "modeling.NemotronH_Nano_VL_V2",
        MODEL_ID,
        trust_remote_code=True,
    )
    cls._supports_flash_attn = True
    cls._supports_flash_attn_2 = True
    cls._supports_sdpa = True
    print(f"  patched class: {cls.__name__}  _supports_flash_attn={cls._supports_flash_attn}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.eval()
    emb = model.language_model.get_input_embeddings()
    print(f"  embedding.weight: shape={tuple(emb.weight.shape)} dtype={emb.weight.dtype}")
    print(f"  img_context_token_id (from model): {getattr(model, 'img_context_token_id', '?')}")
    print(f"  vision_model: {type(model.vision_model).__name__}")

    header("Build one sample")
    convs = make_cord_v2_dataset(n_train_per_class=1, n_val_per_class=0, split="train")
    print(f"  conversations: {len(convs)}")
    batch = nemotron_vl_collate_fn(convs[:1], processor)
    for k, v in batch.items():
        desc(k, v)

    # Move to GPU
    batch = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    header("Step 1: embedding lookup")
    try:
        with torch.no_grad():
            inputs_embeds = emb(batch["input_ids"])
        desc("inputs_embeds", inputs_embeds)
        print("  ✓ embedding lookup OK")
    except Exception as e:
        print(f"  ✗ embedding failed: {e}")
        traceback.print_exc()
        return

    header("Step 2: vision_model(pixel_values).features")
    try:
        with torch.no_grad():
            vit_out = model.vision_model(batch["pixel_values"])
        feats = getattr(vit_out, "features", vit_out)
        desc("vit_out.features", feats)
        torch.cuda.synchronize()
        print("  ✓ vision_model OK")
    except Exception as e:
        print(f"  ✗ vision_model failed: {e}")
        traceback.print_exc()
        return

    header("Step 3: extract_feature(pixel_values)")
    try:
        with torch.no_grad():
            vit_embeds = model.extract_feature(batch["pixel_values"])
        desc("vit_embeds (post extract_feature)", vit_embeds)
        print("  ✓ extract_feature OK")
    except Exception as e:
        print(f"  ✗ extract_feature failed: {e}")
        traceback.print_exc()
        return

    header("Step 4: full model(**batch) forward")
    try:
        with torch.no_grad():
            out = model(**batch)
        print(f"  ✓ forward OK; loss={getattr(out, 'loss', None)}")
    except Exception as e:
        print(f"  ✗ forward failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
