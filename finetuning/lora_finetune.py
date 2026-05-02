"""
Nemotron-3-Nano-Omni-30B-A3B LoRA Fine-tuning with Unsloth
===========================================================
Hardware requirement: ~60GB VRAM for 16-bit LoRA (A100 80GB recommended)
                      ~35GB VRAM for QLoRA 4-bit (2x A100 40GB or 1x A100 80GB)

Install:
    pip install unsloth
    pip install --upgrade trl transformers accelerate peft datasets bitsandbytes
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import json

# ── 2. Config ─────────────────────────────────────────────────────────────────
MODEL_NAME      = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
MAX_SEQ_LENGTH  = 4096      # Increase up to 262144 if VRAM allows (256K context)
LOAD_IN_4BIT    = True      # True = QLoRA (~35GB VRAM), False = LoRA (~60GB VRAM)
LOAD_IN_16BIT   = False     # Set True and LOAD_IN_4BIT=False for full 16-bit LoRA

# LoRA hyperparameters
LORA_R          = 16        # Rank. 8-16 for fast fine-tunes, up to 64 for complex tasks
LORA_ALPHA      = 32        # Alpha = 2x rank is a safe default
LORA_DROPOUT    = 0.05      # Small dropout helps prevent overfitting

# Training hyperparameters
BATCH_SIZE      = 2         # Per-device batch size (lower if OOM)
GRAD_ACCUM      = 8         # Effective batch = BATCH_SIZE * GRAD_ACCUM = 16
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 3         # 1-3 epochs is typically enough for SFT
MAX_STEPS       = -1        # Set to a positive int to override epochs (useful for testing)
WARMUP_RATIO    = 0.05
OUTPUT_DIR      = "./nemotron_lora_output"

# ── 3. Load Model & Tokenizer ─────────────────────────────────────────────────
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = LOAD_IN_4BIT,
    load_in_16bit   = LOAD_IN_16BIT,
    # Nemotron is a gated model — set your HF token here or via HF_TOKEN env var
    # token           = "hf_your_token_here",
    trust_remote_code = True,
)

# ── 4. Attach LoRA Adapters ───────────────────────────────────────────────────
# NOTE: For MoE models like Nemotron, Unsloth disables the router layer by default
# (fine-tuning the router is generally not recommended and can destabilize training).
# We target all major linear layers in attention and MLP blocks.
model = FastLanguageModel.get_peft_model(
    model,
    r               = LORA_R,
    lora_alpha      = LORA_ALPHA,
    lora_dropout    = LORA_DROPOUT,
    target_modules  = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention layers
        "gate_proj", "up_proj", "down_proj",        # MLP layers
    ],
    bias            = "none",
    use_gradient_checkpointing = "unsloth",  # Unsloth's memory-efficient implementation
    random_state    = 42,
)

# Print trainable parameter summary
model.print_trainable_parameters()

# ── 5. Dataset ────────────────────────────────────────────────────────────────
# Nemotron uses a specific chat template with optional <think> reasoning traces.
# The format below uses the reasoning mode (think=True).
# To disable reasoning traces, set think=False in apply_chat_template.

def format_example(example):
    """
    Format a dataset example into Nemotron's chat template.

    Your dataset should have these fields (adjust as needed):
      - "image_description": text description of the image content
      - "reasoning":         step-by-step analysis (the <think> block)
      - "label":             "REAL" or "FAKE"
      - "explanation":       final explanation of the verdict

    For a vision deepfake task, you would preprocess your images to
    text descriptions or base64 strings before this step, or use
    the full multimodal pipeline (see notes at bottom of file).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert deepfake detection system. "
                "When given an image or image description, you must carefully "
                "analyze visual artifacts, inconsistencies, and manipulation "
                "evidence. Reason step by step before giving your verdict."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analyze this image for signs of manipulation:\n\n"
                f"{example['image_description']}"
            ),
        },
        {
            "role": "assistant",
            # Nemotron reasoning format: wrap reasoning in <think> tags
            "content": (
                f"<think>\n{example['reasoning']}\n</think>\n\n"
                f"**Verdict: {example['label']}**\n\n{example['explanation']}"
            ),
        },
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize          = False,
            add_generation_prompt = False,
        )
    }


# ── Option A: Load your own dataset from a local JSONL file ──────────────────
# Each line in your JSONL should be a JSON object with the fields above.
# Uncomment and point to your file:
#
# raw_dataset = load_dataset("json", data_files={"train": "your_dataset.jsonl"}, split="train")
# dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

# ── Option B: Load from HuggingFace Hub ──────────────────────────────────────
# raw_dataset = load_dataset("your_org/your_deepfake_dataset", split="train")
# dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

# ── Option C: Minimal synthetic dataset for testing ──────────────────────────
# Replace this with your real data before actual training.
synthetic_data = [
    {
        "image_description": "A portrait photo of a woman with smooth skin and natural lighting.",
        "reasoning": (
            "Examining facial features: skin texture appears uniform with no visible pores. "
            "Hairline shows clean separation with no blending artifacts. "
            "Lighting is consistent across the face. Eyes show natural catchlights. "
            "No GAN checkerboard artifacts visible. Background has natural depth-of-field blur."
        ),
        "label": "REAL",
        "explanation": "No signs of manipulation detected. The image shows consistent lighting, natural skin texture, and no visible artifacts.",
    },
    {
        "image_description": "A man's face with slightly blurred edges around the jaw and unnatural eye reflections.",
        "reasoning": (
            "Examining facial boundaries: jaw edges show soft, blended transition inconsistent "
            "with the sharp background. Eye reflections are asymmetric — left eye shows a window "
            "reflection absent in the right eye. Skin texture is overly smooth and lacks natural pores. "
            "Hairline shows slight color fringing. These are characteristic of face-swap deepfakes."
        ),
        "label": "FAKE",
        "explanation": "Multiple manipulation artifacts detected: asymmetric eye reflections, unnatural jaw blending, and overly uniform skin texture consistent with a face-swap operation.",
    },
]

dataset = Dataset.from_list(synthetic_data).map(
    format_example,
    remove_columns=["image_description", "reasoning", "label", "explanation"],
)
print(f"\nDataset size: {len(dataset)} examples")
print(f"\nSample formatted text:\n{dataset[0]['text'][:500]}...\n")

# ── 6. Trainer ────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    args            = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LENGTH,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        num_train_epochs            = NUM_EPOCHS,
        max_steps                   = MAX_STEPS,
        warmup_ratio                = WARMUP_RATIO,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = "cosine",
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        save_steps                  = 100,
        save_total_limit            = 3,
        output_dir                  = OUTPUT_DIR,
        report_to                   = "none",  # Change to "wandb" if you want tracking
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        seed                        = 42,
        dataset_num_proc            = 4,
        packing                     = False,   # Set True to pack short sequences for efficiency
    ),
)

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("Starting training...")
trainer_stats = trainer.train()
print(f"\nTraining complete!")
print(f"  Runtime:        {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"  Samples/second: {trainer_stats.metrics['train_samples_per_second']:.2f}")

# ── 8. Save & Export ──────────────────────────────────────────────────────────
# Save as a small LoRA adapter (~100MB) — use this to load on top of the base model
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
print(f"\nLoRA adapter saved to: {OUTPUT_DIR}/lora_adapter")

# Export to GGUF for local inference with llama.cpp / Ollama
# Uncomment to export — choose quantization level based on your deployment target:
#   q4_k_m   → best balance of size and accuracy (~20GB)
#   q8_0     → higher accuracy, larger file (~35GB)
#   f16      → full precision, largest (~60GB)
#
# model.save_pretrained_gguf(
#     f"{OUTPUT_DIR}/gguf",
#     tokenizer,
#     quantization_method = "q4_k_m",
# )
# print(f"GGUF saved to: {OUTPUT_DIR}/gguf")

# Push adapter to HuggingFace Hub (optional)
# model.push_to_hub("your_username/nemotron-deepfake-lora", token="hf_...")
# tokenizer.push_to_hub("your_username/nemotron-deepfake-lora", token="hf_...")

# ── 9. Inference Test ─────────────────────────────────────────────────────────
print("\nRunning inference test...")
FastLanguageModel.for_inference(model)  # Enable faster inference mode

test_messages = [
    {
        "role": "system",
        "content": (
            "You are an expert deepfake detection system. "
            "Analyze visual artifacts, inconsistencies, and manipulation evidence. "
            "Reason step by step before giving your verdict."
        ),
    },
    {
        "role": "user",
        "content": "Analyze this image for signs of manipulation:\n\nA photo of a celebrity with unusually smooth skin and slight color banding around the hair.",
    },
]

inputs = tokenizer.apply_chat_template(
    test_messages,
    tokenize              = True,
    add_generation_prompt = True,
    return_tensors        = "pt",
).to("cuda")

outputs = model.generate(
    input_ids        = inputs,
    max_new_tokens   = 512,
    temperature      = 1.0,   # Nemotron recommends temp=1.0 for reasoning mode
    top_p            = 1.0,
    do_sample        = True,
)

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=False)
print(f"\nModel response:\n{response}")

# ── Notes on Multimodal (Vision) Fine-tuning ──────────────────────────────────
# The Omni variant includes vision (CRADIO v4-H) and audio (Parakeet) encoders.
# Fine-tuning those multimodal components requires:
#   1. A dataset with actual image tensors, not text descriptions
#   2. Using the model's processor (not just tokenizer) to encode images
#   3. Selecting which layers to fine-tune via Unsloth's vision fine-tune flags:
#      model = FastLanguageModel.get_peft_model(model, ..., finetune_vision_layers=True)
# The Omni model's vision fine-tuning support via Unsloth is still maturing —
# check https://unsloth.ai/docs/models/nemotron-3-nano-omni for the latest guides.