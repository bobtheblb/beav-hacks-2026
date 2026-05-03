import os

os.environ["HF_HOME"] = f"/nfs/hpc/share/{os.environ['USER']}/hf_cache"
os.environ["XDG_CACHE_HOME"] = f"/nfs/hpc/share/{os.environ['USER']}/.cache"
os.environ["FLASHINFER_WORKSPACE_BASE"] = f"/nfs/hpc/share/{os.environ['USER']}"

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        trust_remote_code=True,
        dtype="auto",
        max_model_len=131072,
        kv_cache_dtype="fp8",
    )

    print("Model ready")

    params = SamplingParams(temperature=0.6, max_tokens=200)

    single = llm.generate(
        ["Give me 3 bullet points about vLLM."], sampling_params=params
    )
    print(single[0].outputs[0].text)


if __name__ == "__main__":
    main()
