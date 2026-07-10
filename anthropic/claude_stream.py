# Run with: python anthropic/claude_stream.py
import os
from dotenv import load_dotenv
import anthropic

load_dotenv(override=True)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env")

client = anthropic.Anthropic(api_key=anthropic_api_key)

system_message = "You are an assistant that is great at telling jokes"
# A longer prompt makes the streaming effect easier to see in the terminal.
# Keep it bounded so the demo streams for a few seconds without wasting tokens.
user_prompt = (
    "Write a playful 180-word mini story about a data scientist teaching "
    "a confused penguin how to use Python. Make it fun and easy to follow."
)

# messages.stream() returns a context manager; the stream stays open until the `with` block exits
result = client.messages.stream(
    # Haiku is fast and lower-cost, which makes it good for streaming demos.
    model="claude-haiku-4-5-20251001",
    # max_tokens limits how many output tokens Claude can generate.
    # It does not count the input prompt tokens.
    # Higher values allow longer streamed responses; lower values cut replies shorter.
    max_tokens=350,
    # temperature controls randomness/creativity.
    # 0.0 is more predictable and focused; higher values are more varied.
    # 0.7 is a good middle ground for a creative joke/story demo.
    temperature=0.7,
    system=system_message,
    messages=[{"role": "user", "content": user_prompt}],
)

# text_stream yields individual text deltas as they arrive from the API
with result as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
