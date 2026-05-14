import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_message = "You are a helpful assistant."

def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    # stream=True returns an iterator of chunks instead of waiting for the full response
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        if content:
            # flush=True forces each token to print immediately (typing effect)
            print(content, end="", flush=True)


if __name__ == "__main__":
    stream_gpt("Tell me a joke about penguins")
