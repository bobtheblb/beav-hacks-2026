"""Dataset loader for the MMFR fake/real subset built by prepare_mmfr.py.

Mirrors the shape of nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset:
same (path_or_dataset, split, **kwargs) signature, inner format() function,
returning a list of {"conversation": [...]} samples.

Output the assistant is trained to produce per sample:

    <think>
    {per-image reasoning, pulled from MMFR's forgery_reasoning_cot.json}
    </think>

    ```json
    {"status": "ai_generated"|"real", "confidence": 1.0, "reason": "..."}
    ```
"""

import json
from pathlib import Path

from datasets import load_from_disk

USER_PROMPT = (
    "Analyze the provided image and determine whether it is AI-generated.\n"
    "Think through the visual cues, then respond with a JSON object with these fields:\n"
    "  - status: either \"ai_generated\" or \"real\"\n"
    "  - confidence: a number between 0 and 1\n"
    "  - reason: a brief explanation of your decision"
)

_DEFAULT_PATH = str(Path(__file__).resolve().parent.parent / "data" / "mmfr_subset")


def _short_reason(reasoning: str, max_chars: int = 240) -> str:
    text = reasoning.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(". ")
    return (cut[: last_period + 1] if last_period > 50 else cut).strip()


def make_mmfr_dataset(
    path_or_dataset=_DEFAULT_PATH,
    split="train",
    **kwargs,
):
    """Load and preprocess the MMFR fake/real subset for image+text VLM fine-tuning."""
    dataset = load_from_disk(path_or_dataset)

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
