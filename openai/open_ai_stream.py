# Run with: python openai/open_ai_stream.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY from .env and create the SDK client.
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# The system message is reused for each streamed request.
system_message = "You are a helpful assistant."

def stream_gpt(prompt):
    """Stream one assistant response token-by-token in the terminal."""
    # Build a fresh message list for this one prompt.
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
        # Some chunks only carry metadata, so delta.content can be None.
        # `or ""` means: use the streamed text if it exists; otherwise use an empty string.
        # This does not return a boolean — it keeps `content` safe to print as text.
        content = chunk.choices[0].delta.content or ""
        if content:
            # flush=True forces each token to print immediately (typing effect)
            print(content, end="", flush=True)


if __name__ == "__main__":
    # Swap the prompt below to experiment with different streamed outputs.
    # stream_gpt("Tell me a joke about penguins")
    stream_gpt("Write a 500 word story about a penguin learning Python.")
