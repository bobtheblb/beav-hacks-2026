# beav-hacks-2026

> **Note:** This project was built for BeavHacks 2026, a 24-hour hackathon hosted at Oregon State University.

LoRA fine-tune of [NVIDIA-Nemotron-Nano-12B-v2-VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16) on the [MMFR](https://huggingface.co/datasets/AnnaGao/MMFR-Dataset) fake-vs-real image dataset, plus a vLLM-served inference path.

## Team (alphabetical by last name)

- Kevin Chang
- Digesh Chitrakar
- Josh Negreanu
- Jose Sanchez-Gonzalez

The fine-tuned model takes a single image and emits a chain-of-thought verdict:

````
<think>
{visual analysis of the image}
</think>

```json
{"status": "ai_generated" | "real", "confidence": 0.0-1.0, "reason": "..."}
```
````

## Layout

- [finetuning/](finetuning/) — training entry point, dataset loaders, model-loading shims
  - [automodel_lora.yaml](finetuning/automodel_lora.yaml) — NeMo-AutoModel recipe (LoRA r=8, α=16, FSDP2, AdamW lr=1e-4)
  - [automodel_lora.sh](finetuning/automodel_lora.sh) — launcher that sets `HF_HOME`, `PYTHONPATH`, and execs `automodel`
  - [datasets.py](finetuning/datasets.py) — streams MMFR shards, builds the `<think>…</think> + JSON` conversation format, custom collate that rebuilds labels via prefix-diff (Nemotron's tokenizer BPE-splits the chat markers)
  - [patch_radio.py](finetuning/patch_radio.py) — patches the cached `radio_model.py` and Nemotron VL `modeling.py` so they survive FSDP2 meta-device init and the Mamba/SSM language head's missing KV cache
- [inference/](inference/) — merge + serve + clients
  - [merge_lora.py](inference/merge_lora.py) — state-dict-level LoRA merge that skips model instantiation (avoids meta-tensor issues with custom-code multimodal models)
  - [serve_mmfr.sh](inference/serve_mmfr.sh) — `vllm serve` wrapper for the merged checkpoint (port 5000, served as `nemotron-vl`)
  - [chat_image.py](inference/chat_image.py) — send an image (or a directory of them) to the running server and print the verdict
- [Automodel/](Automodel/) — git submodule fork of NeMo-AutoModel

## Setup

```bash
git clone --recurse-submodules <repo>
cd beav-hacks-2026

# point caches at NFS share so the 12B base + dataset shards don't fill $HOME
export HF_HOME=/nfs/hpc/share/$USER/hf_cache

# install NeMo-AutoModel from the submodule
pip install -e Automodel
pip install vllm openai huggingface_hub safetensors

# patch the cached custom code (safe to re-run)
python finetuning/patch_radio.py
```

`patch_radio.py` only finds files after the model has been downloaded once — run a quick `from huggingface_hub import snapshot_download; snapshot_download("nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")` first if needed, then re-run the patch.

## Train

```bash
bash finetuning/automodel_lora.sh
```

Edit [finetuning/automodel_lora.yaml](finetuning/automodel_lora.yaml) to change `checkpoint_dir`, batch size, validation cadence, or `wandb` settings. Checkpoints land at `/nfs/hpc/share/$USER/checkpoints/mmfr_lora/`; staging goes to `…/ckpt_staging/` (NFS, not tmpfs — a 12B-LoRA save briefly stages gigabytes).

Useful env vars:

- `MMFR_DEBUG=1` — print collate/label diagnostics for the first few batches
- `MMFR_SKIP_OPTIM_LOAD=1` — one-shot recovery flag if a kill-mid-save left the AdamW state corrupt; loads model weights, skips optimizer state, lets momentum rebuild

## Merge LoRA → standalone checkpoint

```bash
python inference/merge_lora.py \
  /nfs/hpc/share/$USER/checkpoints/mmfr_lora/<step>/model \
  /nfs/hpc/share/$USER/checkpoints/mmfr_lora/latest_merged
```

This works at the safetensors level — it never instantiates the model, so it sidesteps meta-tensor problems with the custom modeling code.

## Serve & query

```bash
# default points at .../latest_merged
bash inference/serve_mmfr.sh

# or override
MODEL_PATH=/path/to/merged bash inference/serve_mmfr.sh
```

Then hit it:

```bash
python inference/chat_image.py data/mmfr_samples/real          # iterate a directory
python inference/chat_image.py path/to/image.jpg               # single image
```

The client prefills `<think>\n` and uses `continue_final_message` so vLLM emits the reasoning block deterministically.

## Desktop App

The [desktop-app/](desktop-app/) directory contains an Electron app (built with electron-vite + React) that captures screenshots and runs AI detection analysis against the inference server.

### Prerequisites

- Node.js 18+
- The vLLM inference server running (see [Serve & query](#serve--query) above)

### Install dependencies

```bash
cd desktop-app
npm install
```

### Run in development mode

```bash
npm run dev
```

### Build a distributable

```bash
npm run package
```

The packaged app lands in `desktop-app/out/`.

---

## Notes / gotchas

- The Nemotron-Nano-VL tokenizer ships without a `pad_token` and BPE-splits `<|im_start|>`/`<|im_end|>`; the loader sets `pad_token = eos_token` and the collate rebuilds labels from a re-rendered prefix instead of relying on marker scans.
- The model's forward pass requires `image_flags` of shape `(num_tiles, 1)` matching `pixel_values.shape[0]`; the collate adds it.
- C-RADIOv2-H's `summary_idxs` buffer can be silently filled with uninitialized GPU memory under FSDP2 meta-device init — `patch_radio.py` rebuilds it from a saved Python list at the top of `forward()`.
- Nemotron-H is a Mamba/SSM hybrid; its `CausalLMOutput` has no `past_key_values`. The wrapper's output assembly is patched to use `getattr(..., None)`.

---

**Disclosure:** Agentic AI coding tools were used to assist with troubleshooting the fine-tuning pipeline and for creating the desktop application GUI.
