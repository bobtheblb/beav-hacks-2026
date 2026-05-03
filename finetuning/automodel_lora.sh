#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/nfs/hpc/share/$USER/hf_cache

CONFIG="$(dirname "$0")/automodel_lora.yaml"

exec automodel "$CONFIG" "$@"
