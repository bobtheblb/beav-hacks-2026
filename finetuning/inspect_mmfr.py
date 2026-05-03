"""Sanity-check the MMFR dataset produced by make_cord_v2_dataset().

Calls the dataset function directly (no on-disk Arrow dump needed), then dumps
all images to data/mmfr_samples/<class>/<idx>.{png|jpg} and writes a browsable
markdown index at data/mmfr_samples/INDEX.md.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from finetuning.datasets import USER_PROMPT, make_cord_v2_dataset

REPO = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO / "data" / "mmfr_samples"
INDEX_MD = SAMPLES_DIR / "INDEX.md"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _parse_assistant(text):
    think = (m.group(1).strip() if (m := _THINK_RE.search(text)) else "")
    payload = (m.group(1) if (m := _JSON_RE.search(text)) else "{}")
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        obj = {}
    return think, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation"], default="train")
    ap.add_argument("--n-train-per-class", type=int, default=100)
    ap.add_argument("--n-val-per-class", type=int, default=20)
    args = ap.parse_args()

    print(
        f"=== make_cord_v2_dataset(split={args.split!r}, "
        f"n_train_per_class={args.n_train_per_class}, n_val_per_class={args.n_val_per_class}) ==="
    )
    convs = make_cord_v2_dataset(
        split=args.split,
        n_train_per_class=args.n_train_per_class,
        n_val_per_class=args.n_val_per_class,
    )
    print(f"  rows: {len(convs)}")

    by_class: dict[str, list[int]] = {"ai_generated": [], "real": []}
    rlens, sizes = [], Counter()
    for i, sample in enumerate(convs):
        msg = sample["conversation"]
        img = msg[0]["content"][0]["image"]
        think, obj = _parse_assistant(msg[1]["content"][0]["text"])
        status = obj.get("status", "unknown")
        by_class.setdefault(status, []).append(i)
        rlens.append(len(think))
        sizes[img.size] += 1

    print(f"  class balance: {{'ai_generated': {len(by_class['ai_generated'])}, 'real': {len(by_class['real'])}}}")
    if rlens:
        rlens.sort()
        print(f"  reasoning chars: min={rlens[0]}  median={rlens[len(rlens)//2]}  max={rlens[-1]}")
    print(f"  top image sizes: {', '.join(f'{s}×{c}' for s, c in sizes.most_common(5))}")

    # dump images + write markdown index
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLES_DIR / "ai_generated").mkdir(exist_ok=True)
    (SAMPLES_DIR / "real").mkdir(exist_ok=True)

    md = ["# MMFR subset — browse all samples\n"]
    md.append(
        f"Class balance: **{len(by_class['ai_generated'])} ai_generated**, "
        f"**{len(by_class['real'])} real**.\n"
    )
    md.append(f"User prompt:\n\n```\n{USER_PROMPT}\n```\n")

    for cls, idxs in by_class.items():
        if not idxs:
            continue
        md.append(f"\n---\n\n## {cls.upper()} ({len(idxs)})\n")
        for n, i in enumerate(idxs):
            img = convs[i]["conversation"][0]["content"][0]["image"]
            assistant_text = convs[i]["conversation"][1]["content"][0]["text"]
            ext = "png" if cls == "ai_generated" else "jpg"
            rel = f"{cls}/{n:03d}.{ext}"
            img.save(SAMPLES_DIR / rel)
            md.append(f"\n### {cls} #{n}  (row {i}, {img.size[0]}×{img.size[1]})\n")
            md.append(f"![{rel}]({rel})\n")
            md.append(f"\n**Assistant target:**\n\n````\n{assistant_text}\n````\n")

    INDEX_MD.write_text("\n".join(md))
    total = sum(len(v) for v in by_class.values())
    print(f"\n  wrote {total} images under {SAMPLES_DIR.relative_to(REPO)}/{{ai_generated,real}}/")
    print(f"  wrote browsable index: {INDEX_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
