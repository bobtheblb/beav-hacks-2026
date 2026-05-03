#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/nfs/hpc/share/$USER/hf_cache
export FLASHINFER_WORKSPACE_BASE=/nfs/hpc/share/$USER

# Run from repo root so `finetuning.mmfr_dataset` resolves as a Python module.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CONFIG="finetuning/automodel_lora.yaml"

exec automodel "$CONFIG" "$@"
