from openai import OpenAI

client = OpenAI(
    api_key="你的API_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

stream = client.chat.completions.create(
    model="qwen3-vl-235b-a22b-thinking",
    messages=[{"role": "user", "content": "夏天适合吃什么水果？"}],
    stream=True,
    extra_body={
        "enable_thinking": False,
        "thinking_budget": 8192,
    },
)

for chunk in stream:
    if chunk.choices:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

print()