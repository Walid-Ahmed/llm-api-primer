import os
from dotenv import load_dotenv
import anthropic

load_dotenv(override=True)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env")

client = anthropic.Anthropic(api_key=anthropic_api_key)

system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

# messages.stream() returns a context manager; the stream stays open until the `with` block exits
result = client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[{"role": "user", "content": user_prompt}],
)

# text_stream yields individual text deltas as they arrive from the API
with result as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
