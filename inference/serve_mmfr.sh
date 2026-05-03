#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/nfs/hpc/share/$USER/hf_cache
export XDG_CACHE_HOME=/nfs/hpc/share/$USER/.cache
export FLASHINFER_WORKSPACE_BASE=/nfs/hpc/share/$USER
export VLLM_USE_DEEP_GEMM=0

# Override at launch:
#   MODEL_PATH=/path/to/merged bash inference/serve_mmfr.sh
# Default: serve a merged checkpoint produced by merge_lora.py.
MODEL_PATH="${MODEL_PATH:-/nfs/hpc/share/$USER/checkpoints/mmfr_lora/latest_merged}"

exec vllm serve "$MODEL_PATH" \
    --dtype auto \
    --trust-remote-code \
    --served-model-name nemotron-vl \
    --host 0.0.0.0 \
    --port 5000 \
    --max-model-len 8192 \
    --enforce-eager \
    --gpu-memory-utilization 0.85
