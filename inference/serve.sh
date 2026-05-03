#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/nfs/hpc/share/$USER/hf_cache

exec python -m vllm.entrypoints.openai.api_server \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --dtype auto \
    --trust-remote-code \
    --served-model-name nemotron \
    --host 0.0.0.0 \
    --port 5000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --kv-cache-dtype fp8 \
    --enforce-eager
