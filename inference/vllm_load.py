import os

os.environ["HF_HOME"] = f"/nfs/hpc/share/{os.environ['USER']}/hf_cache"

from vllm import LLM, SamplingParams

llm = LLM(
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    trust_remote_code=True,
    dtype="auto",
)

print("Model ready")

params = SamplingParams(temperature=0.6, max_tokens=200)

single = llm.generate(["Give me 3 bullet points about vLLM."], sampling_params=params)
print(single[0].outputs[0].text)

# prompts = [
#     "Hello, my name is",
#     "The capital of France is",
#     "Explain quantum computing in simple terms:",
# ]
# outputs = llm.generate(prompts, sampling_params=params)
# for i, out in enumerate(outputs):
#     print(f"\nPrompt {i+1}: {out.prompt!r}")
#     print(out.outputs[0].text)
