#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/nfs/hpc/share/$USER/hf_cache
export XDG_CACHE_HOME=/nfs/hpc/share/$USER/.cache
export FLASHINFER_WORKSPACE_BASE=/nfs/hpc/share/$USER

exec vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 \
    --kv-cache-dtype fp8 \
    --served-model-name nemotron \
    --host 0.0.0.0 \
    --port 5000 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --trust-remote-code \
    --video-pruning-rate 0.5 \
    --media-io-kwargs '{"video": {"num_frames": 512, "fps": 1}}' \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
