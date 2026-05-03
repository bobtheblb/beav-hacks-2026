"""Send an image to the running vLLM server and print the verdict.

Usage:
    python inference/chat_image.py                      # prompt for path
    python inference/chat_image.py path/to/img.jpg      # explicit image
    python inference/chat_image.py data/mmfr_samples/real    # iterate a dir
"""

import base64
import mimetypes
import re
import sys
from pathlib import Path

from openai import OpenAI

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
THINK_END_RE = re.compile(r"^(.*?)</think>", re.DOTALL)
THINK_PREFILL = "<think>\n"

client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="null")
MODEL = "nemotron-vl"


def encode_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def ask(path: Path) -> None:
    ground_truth = path.parent.name  # "real" or "ai_generated"
    print(f"\n=== {path}  (ground truth: {ground_truth}) ===")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You must always begin your response with a <think>...</think> "
                    "block describing your visual analysis, then output ONLY a JSON "
                    "object after </think>. Responses without <think> are invalid."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_image(path)}},
                    {
                        "type": "text",
                        "text": (
                            "Analyze the provided image and determine whether it is AI-generated.\n"
                            "First reason inside <think>...</think> tags about the visual cues you observe.\n"
                            "After </think>, output ONLY a JSON object with these fields:\n"
                            "  - status: either \"ai_generated\" or \"real\"\n"
                            "  - confidence: a calibrated number between 0 and 1.\n"
                            "      * Reserve 1.0 ONLY for unmistakable evidence (e.g. obvious GAN artifacts, impossible anatomy).\n"
                            "      * Use 0.85-0.95 for clear cases.\n"
                            "      * Use 0.6-0.8 when there are some ambiguous details.\n"
                            "      * Use 0.5-0.6 when the image is genuinely hard to classify.\n"
                            "      * Never use exactly 1.0 unless you would bet money on it.\n"
                            "  - reason: a brief explanation of your decision"
                        ),
                    },
                ],
            },
            {"role": "assistant", "content": THINK_PREFILL},
        ],
        temperature=0.2,
        max_tokens=1024,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "continue_final_message": True,
            "add_generation_prompt": False,
        },
    )

    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    content = msg.content or ""
    if not reasoning:
        # Case 1: full <think>...</think> block in content
        match = THINK_RE.search(content)
        if match:
            reasoning = match.group(1).strip()
            content = THINK_RE.sub("", content).strip()
        else:
            # Case 2: prefill stripped <think>; only </think> remains
            match = THINK_END_RE.search(content)
            if match:
                reasoning = match.group(1).strip()
                content = content[match.end():].strip()
    if reasoning:
        print(f"--- thinking ---\n{reasoning}")
    print(f"--- answer ---\n{content}")


def main():
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path(input("Image path or directory: ").strip())

    if target.is_dir():
        images = sorted(p for p in target.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not images:
            print(f"No images found under {target}")
            sys.exit(1)
        for p in images:
            ask(p)
    else:
        ask(target)


if __name__ == "__main__":
    main()
