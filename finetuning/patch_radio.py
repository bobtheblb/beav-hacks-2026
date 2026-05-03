"""Patch the cached Nemotron-Nano-VL custom code so it can train under
NeMo-AutoModel + transformers + FSDP2.

Two files get edited (idempotently):

1. ``C-RADIOv2-H/.../radio_model.py`` — make `summary_idxs` survive
   FSDP2's meta-device init by storing the int list as a plain Python attribute
   and rebuilding the buffer at the top of ``forward()``. Without this, the
   buffer materializes to uninitialized GPU memory and indexing trips
   CUDA's `IndexKernel` "index out of bounds" assertion.

2. ``NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/.../modeling.py`` — replace
   ``outputs.past_key_values`` (etc.) with ``getattr(outputs, ..., None)``
   in the wrapper's output assembly. The Nemotron-H language model is a
   Mamba/SSM hybrid whose output object doesn't carry KV cache.

Run after clearing the HF cache or on a fresh checkout:

    HF_HOME=/path/to/cache python3 finetuning/patch_radio.py
"""

import os
import sys
from pathlib import Path

# Edit 1 — save indices as a Python list alongside the buffer.
INIT_OLD = """        if summary_idxs is not None:
            self.register_buffer('summary_idxs', summary_idxs)
        else:
            self.summary_idxs = None"""

INIT_NEW = """        if summary_idxs is not None:
            # FSDP2/meta-device init can materialize this buffer to fresh GPU memory
            # without copying values, leaving uninitialized garbage that breaks
            # `all_summary[:, self.summary_idxs]`. Save the indices as a Python list
            # and rebuild the buffer in forward() each call.
            self._summary_idxs_list = [int(i) for i in summary_idxs.tolist()]
            self.register_buffer('summary_idxs', summary_idxs, persistent=False)
        else:
            self._summary_idxs_list = None
            self.summary_idxs = None"""

# Edit 2 — rebuild summary_idxs at the top of forward() from the saved list.
FORWARD_OLD = """        '''
        Forward process for model.
        Args:
            x: Input tensor. Unless `make_preprocessor_external` has been called, then the dynamic range of `x` is expected to be `[0, 1]`,
                             otherwise `x` is expected to be mean centered with unit standard deviation.
            feature_format: ['NLC', 'NCHW'] - The output format for the features.
        '''
        res_step = self.min_resolution_step"""

FORWARD_NEW = """        '''
        Forward process for model.
        Args:
            x: Input tensor. Unless `make_preprocessor_external` has been called, then the dynamic range of `x` is expected to be `[0, 1]`,
                             otherwise `x` is expected to be mean centered with unit standard deviation.
            feature_format: ['NLC', 'NCHW'] - The output format for the features.
        '''
        # Rebuild summary_idxs from the saved Python list each call — the buffer
        # can be silently replaced with uninitialized memory under FSDP2 meta-device init.
        if getattr(self, '_summary_idxs_list', None):
            self.summary_idxs = torch.tensor(self._summary_idxs_list, dtype=torch.int64, device=x.device)

        res_step = self.min_resolution_step"""


# Edit 3 — Nemotron-H language model output doesn't have past_key_values.
NEMOTRON_OLD = """        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )"""

NEMOTRON_NEW = """        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            # Nemotron-H is a Mamba/SSM hybrid whose CausalLMOutput doesn't carry
            # past_key_values (or attentions). Use getattr so this wrapper works
            # regardless of the underlying language-model output schema.
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )"""


def hf_modules_root() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "modules" / "transformers_modules"


def radio_files() -> list[Path]:
    return list(hf_modules_root().glob("**/C_hyphen_RADIOv2_hyphen_H/**/radio_model.py"))


def nemotron_files() -> list[Path]:
    return list(hf_modules_root().glob("**/NVIDIA_hyphen_Nemotron_hyphen_Nano*VL*/**/modeling.py"))


def apply(text: str, old: str, new: str) -> tuple[str, str]:
    if new in text:
        return text, "skip"
    if old in text:
        return text.replace(old, new), "patched"
    return text, "no-match"


def patch_file(path: Path, edits: list[tuple[str, str]]) -> str:
    text = path.read_text()
    statuses = []
    for old, new in edits:
        text, s = apply(text, old, new)
        statuses.append(s)
    if "patched" in statuses:
        path.write_text(text)
    if "no-match" in statuses:
        return "warn (" + ",".join(statuses) + ")"
    if all(s == "skip" for s in statuses):
        return "skip"
    return "patched"


def main() -> int:
    targets = [
        ("radio_model.py", radio_files(), [(INIT_OLD, INIT_NEW), (FORWARD_OLD, FORWARD_NEW)]),
        ("Nemotron VL modeling.py", nemotron_files(), [(NEMOTRON_OLD, NEMOTRON_NEW)]),
    ]
    found_any = False
    rc = 0
    for label, paths, edits in targets:
        if not paths:
            print(f"[skip] no cached {label} found")
            continue
        found_any = True
        for p in paths:
            status = patch_file(p, edits)
            marker = "[done]" if status == "patched" else ("[skip]" if status == "skip" else "[warn]")
            print(f"{marker} {status:>8}  {label:>28}  {p}")
            if status.startswith("warn"):
                rc = 2
    if not found_any:
        print("\nNo cached files found under "
              f"{os.environ.get('HF_HOME', '~/.cache/huggingface')}.")
        print("Run training once first to populate the cache, then re-run this script.")
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
