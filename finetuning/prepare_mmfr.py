"""Build a small MMFR train subset (N fake + N real) as a local HF Dataset.

Source layout (AnnaGao/MMFR-Dataset):
- dataset.part1.tar contains:
    dataset/forgery_reasoning_cot.json   <- per-image CoT reasoning (real + fake)
    dataset/diffusiondb/part-000001/<uuid>.png   <- fake images
- dataset.part80.tar contains:
    dataset/laion/00000/<id>.jpg                 <- real images

We pick the first N JSON entries that map into shards we have on disk, pull
the image bytes out of the matching tar, and save an Arrow dataset with:
    image:      PIL.Image
    label:      int  (0 = real, 1 = fake)
    label_text: str
    reasoning:  str  (extracted <REASONING> block from the GPT CoT response)
"""

import argparse
import io
import json
import os
import re
import subprocess
from pathlib import Path

import requests
from datasets import Dataset, Features
from datasets import Image as HFImage
from datasets import Value
from PIL import Image

REPO = "AnnaGao/MMFR-Dataset"
FAKE_SHARD = "dataset.part1.tar"   # diffusiondb part-000001
REAL_SHARD = "dataset.part80.tar"  # laion 00000
COT_JSON_TARPATH = "dataset/forgery_reasoning_cot.json"

# JSON-image-prefix -> (shard filename, tar prefix to prepend for member lookup)
SHARD_MAP = {
    "diffusiondb/part-000001/": (FAKE_SHARD, "dataset/"),
    "laion/00000/":              (REAL_SHARD, "dataset/"),
}

_REASONING_RE = re.compile(r"<REASONING>(.*?)</REASONING>", re.DOTALL)


def shard_url(name: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/main/{name}"


def download(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url} -> {dst}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"[done] {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def extract_member(tar_path: Path, member: str) -> bytes | None:
    """Extract a single tar member's bytes via system tar (tolerant of MMFR's tar prefix)."""
    proc = subprocess.run(
        ["tar", "-xOf", str(tar_path), member],
        capture_output=True, check=False,
    )
    return proc.stdout or None


def load_cot_index(fake_tar: Path, cache_path: Path) -> list[dict]:
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    print(f"[get ] extracting {COT_JSON_TARPATH} from {fake_tar.name}")
    blob = extract_member(fake_tar, COT_JSON_TARPATH)
    if not blob:
        raise RuntimeError(f"Could not extract {COT_JSON_TARPATH} from {fake_tar}")
    cache_path.write_bytes(blob)
    print(f"[save] {cache_path} ({len(blob) / 1e6:.1f} MB)")
    return json.loads(blob)


def extract_reasoning(gpt_response: str) -> str:
    m = _REASONING_RE.search(gpt_response)
    return m.group(1).strip() if m else gpt_response.strip()


def gather(cot: list[dict], json_prefix: str, tar_path: Path, n: int, label: int, label_text: str):
    rows = []
    for entry in cot:
        if len(rows) >= n:
            break
        img_rel = entry["image"]
        if not img_rel.startswith(json_prefix):
            continue
        member = "dataset/" + img_rel
        data = extract_member(tar_path, member)
        if not data:
            continue
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            print(f"[warn] decode fail {member}: {e}")
            continue
        gpt_resp = entry["conversations"][1]["value"]
        rows.append({
            "image": img,
            "label": label,
            "label_text": label_text,
            "reasoning": extract_reasoning(gpt_resp),
        })
    print(f"[take] {len(rows)} {label_text} images from {tar_path.name}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--cache-dir", default=os.environ.get("MMFR_CACHE", f"/nfs/hpc/share/{os.environ['USER']}/mmfr_raw"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "mmfr_subset"))
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    fake_tar = download(shard_url(FAKE_SHARD), cache / FAKE_SHARD)
    real_tar = download(shard_url(REAL_SHARD), cache / REAL_SHARD)

    cot = load_cot_index(fake_tar, cache / "forgery_reasoning_cot.json")
    print(f"[load] CoT entries: {len(cot)}")

    fake_rows = gather(cot, "diffusiondb/part-000001/", fake_tar, args.n_per_class, label=1, label_text="fake")
    real_rows = gather(cot, "laion/00000/",              real_tar, args.n_per_class, label=0, label_text="real")

    rows = fake_rows + real_rows
    if not rows:
        raise RuntimeError("No images extracted — check shard contents.")

    features = Features({
        "image": HFImage(),
        "label": Value("int64"),
        "label_text": Value("string"),
        "reasoning": Value("string"),
    })
    ds = Dataset.from_list(rows, features=features).shuffle(seed=42)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"[save] {len(ds)} rows -> {out}")


if __name__ == "__main__":
    main()
