"""Sanity-check the MMFR subset built by prepare_mmfr.py.

Prints schema + class balance, dumps all images to data/mmfr_samples/<label>/<idx>.{png|jpg},
and writes a browsable markdown index at data/mmfr_samples/INDEX.md.
"""

from collections import Counter
from pathlib import Path

from datasets import load_from_disk

from finetuning.mmfr_dataset import USER_PROMPT, make_mmfr_dataset

REPO = Path(__file__).resolve().parent.parent
DS_PATH = REPO / "data" / "mmfr_subset"
SAMPLES_DIR = REPO / "data" / "mmfr_samples"
INDEX_MD = SAMPLES_DIR / "INDEX.md"


def main():
    ds = load_from_disk(str(DS_PATH))
    print(f"=== Dataset @ {DS_PATH.relative_to(REPO)} ===")
    print(f"  rows:     {len(ds)}")
    print(f"  columns:  {ds.column_names}")
    print(f"  features: {ds.features}")

    # class balance + reasoning length
    counts = Counter(ds["label_text"])
    print(f"\n  class balance: {dict(counts)}")
    rlens = [len(r) for r in ds["reasoning"]]
    print(f"  reasoning chars: min={min(rlens)}  median={sorted(rlens)[len(rlens)//2]}  max={max(rlens)}")
    img_sizes = Counter(ex["image"].size for ex in ds.select(range(len(ds))))
    top_sizes = ", ".join(f"{s}×{c}" for s, c in img_sizes.most_common(5))
    print(f"  top image sizes: {top_sizes}")

    # dump every image + write markdown index
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLES_DIR / "fake").mkdir(exist_ok=True)
    (SAMPLES_DIR / "real").mkdir(exist_ok=True)
    for old in SAMPLES_DIR.glob("*.png"):
        old.unlink()

    md = ["# MMFR subset — browse all 200 samples\n"]
    md.append(f"Class balance: **{counts['fake']} fake**, **{counts['real']} real**.\n")
    md.append(f"User prompt:\n\n```\n{USER_PROMPT}\n```\n")

    convs = make_mmfr_dataset()
    assistant_targets = [c["conversation"][1]["content"][0]["text"] for c in convs]

    by_class: dict[str, list[int]] = {"fake": [], "real": []}
    for i in range(len(ds)):
        by_class[ds[i]["label_text"]].append(i)

    for cls in ("fake", "real"):
        md.append(f"\n---\n\n## {cls.upper()} ({len(by_class[cls])})\n")
        for n, i in enumerate(by_class[cls]):
            ex = ds[i]
            ext = "png" if cls == "fake" else "jpg"
            rel = f"{cls}/{n:03d}.{ext}"
            ex["image"].save(SAMPLES_DIR / rel)

            md.append(f"\n### {cls} #{n}  (row {i}, {ex['image'].size[0]}×{ex['image'].size[1]})\n")
            md.append(f"![{rel}]({rel})\n")
            md.append(f"\n**Assistant target:**\n\n````\n{assistant_targets[i]}\n````\n")

    INDEX_MD.write_text("\n".join(md))
    print(f"\n  wrote {len(ds)} images under {SAMPLES_DIR.relative_to(REPO)}/{{fake,real}}/")
    print(f"  wrote browsable index: {INDEX_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
