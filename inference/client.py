from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="null")

resp = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about GPUs."},
    ],
    temperature=1,
    max_tokens=256,
)
print("Reasoning:", getattr(resp.choices[0].message, "reasoning_content", None))
print("Content:", resp.choices[0].message.content)

resp2 = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me 3 interesting facts about vLLM."},
    ],
    temperature=0,
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print("\n--- thinking disabled ---")
print(resp2.choices[0].message.content)

print("\n--- streaming ---")
stream = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the first 5 prime numbers?"},
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True,
)
print("Reasoning:", getattr(resp.choices[0].message, "reasoning_content", None))
print("Content:", resp.choices[0].message.content)

resp2 = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me 3 interesting facts about vLLM."},
    ],
    temperature=0,
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print("\n--- thinking disabled ---")
print(resp2.choices[0].message.content)

print("\n--- streaming ---")
stream = cli
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
