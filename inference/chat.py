from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="null")

prompt = input("You: ")

resp = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.7,
    max_tokens=1024,
)

msg = resp.choices[0].message
reasoning = getattr(msg, "reasoning_content", None)
if reasoning:
    print("\n--- thinking ---")
    print(reasoning)
print("\n--- answer ---")
print(msg.content)
