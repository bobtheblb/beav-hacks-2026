"""Merge a LoRA adapter into the base model at the state-dict level.

Skips model instantiation entirely (avoids meta-tensor issues with
custom-code multimodal models like Nemotron-Nano-VL). Reads base
safetensors from the HF cache, applies the adapter delta to matching
weights, and writes a complete merged checkpoint.

Usage:
    python inference/merge_lora.py <adapter_dir> <output_dir>

<adapter_dir> contains adapter_model.safetensors + adapter_config.json
(e.g. /nfs/hpc/share/$USER/checkpoints/mmfr_lora/LATEST/model).
"""

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", f"/nfs/hpc/share/{os.environ['USER']}/hf_cache")

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

BASE_MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"


def find_base_snapshot() -> Path:
    """Return path to the local base-model snapshot dir, downloading if needed."""
    return Path(snapshot_download(BASE_MODEL, allow_patterns=None))


def strip_peft_prefix(name: str) -> str:
    return name.removeprefix("base_model.model.")


def load_adapter(adapter_dir: Path):
    """Returns (lora_A_dict, lora_B_dict, alpha, rank, target_modules)."""
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank = config["r"]
    alpha = config["lora_alpha"]
    target_modules = config.get("target_modules", [])

    lora_A, lora_B = {}, {}
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            stripped = strip_peft_prefix(key)
            # PEFT layout: <module>.lora_A.weight, <module>.lora_B.weight
            if stripped.endswith(".lora_A.weight"):
                base_key = stripped[: -len(".lora_A.weight")] + ".weight"
                lora_A[base_key] = tensor
            elif stripped.endswith(".lora_B.weight"):
                base_key = stripped[: -len(".lora_B.weight")] + ".weight"
                lora_B[base_key] = tensor
            else:
                # Other adapter keys (e.g. modules_to_save). Print and skip.
                print(f"  [adapter] unhandled key: {key}")
    return lora_A, lora_B, alpha, rank, target_modules


def merge_one_shard(shard_path: Path, lora_A, lora_B, scaling: float):
    """Read a base safetensors shard, return merged state_dict."""
    out = {}
    merged_count = 0
    with safe_open(shard_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key in lora_A and key in lora_B:
                a, b = lora_A[key], lora_B[key]
                # tensor shape: (out, in); a: (rank, in); b: (out, rank)
                delta = (b.to(torch.float32) @ a.to(torch.float32)) * scaling
                tensor = (tensor.to(torch.float32) + delta).to(tensor.dtype)
                merged_count += 1
            out[key] = tensor
    return out, merged_count


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    adapter_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Locating base model snapshot: {BASE_MODEL}")
    base_dir = find_base_snapshot()
    print(f"  -> {base_dir}")

    print(f"Loading adapter: {adapter_dir}")
    lora_A, lora_B, alpha, rank, target_modules = load_adapter(adapter_dir)
    scaling = alpha / rank
    print(f"  rank={rank} alpha={alpha} scaling={scaling}")
    print(f"  {len(lora_A)} LoRA tensor pairs")

    # Find all base safetensors shards
    index_path = base_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        # Single-file model
        shard_files = ["model.safetensors"]
    print(f"Found {len(shard_files)} base shard(s)")

    total_merged = 0
    for shard in shard_files:
        src = base_dir / shard
        dst = out_dir / shard
        print(f"Merging {shard}...")
        merged_state, n = merge_one_shard(src, lora_A, lora_B, scaling)
        save_file(merged_state, str(dst), metadata={"format": "pt"})
        total_merged += n
        print(f"  merged {n} tensors, wrote {dst}")
    print(f"Total tensors with LoRA delta applied: {total_merged}")
    if total_merged != len(lora_A):
        print(
            f"WARNING: applied {total_merged} but adapter had {len(lora_A)} pairs."
            " Some adapter targets did not match any base tensor."
        )

    # Copy config + tokenizer + custom code from base + adapter
    print("Copying config / tokenizer / custom code")
    for fname in os.listdir(base_dir):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        src = base_dir / fname
        dst = out_dir / fname
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    if index_path.exists():
        shutil.copy2(index_path, out_dir / "model.safetensors.index.json")

    # Adapter dir may have updated tokenizer / custom code — overlay it
    for fname in os.listdir(adapter_dir):
        if fname.startswith("adapter") or fname.startswith("automodel"):
            continue
        src = adapter_dir / fname
        dst = out_dir / fname
        if src.is_file():
            shutil.copy2(src, dst)

    print(f"\nDone. Merged checkpoint: {out_dir}")


if __name__ == "__main__":
    main()
