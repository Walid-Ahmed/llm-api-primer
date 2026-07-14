# Run with: python Anthropic/anthropic_thinking.py
import os

import anthropic
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY from .env and create the SDK client.
load_dotenv(override=True)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env")

client = anthropic.Anthropic(api_key=anthropic_api_key)

PROBLEM = (
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
    "than the ball. How much does the ball cost?"
)

response = client.messages.create(
    # Haiku 4.5 is Anthropic's lowest-cost current model with extended thinking.
    model="claude-haiku-4-5-20251001",
    # max_tokens includes both thinking tokens and final-answer tokens.
    max_tokens=2048,
    # budget_tokens must be lower than max_tokens.
    thinking={"type": "enabled", "budget_tokens": 1024},
    messages=[{"role": "user", "content": PROBLEM}],
)

# Extended-thinking responses contain thinking blocks followed by text blocks.
# The visible thinking may be summarized; it is not guaranteed to be the exact
# raw internal reasoning that was billed.
for block in response.content:
    if block.type == "thinking":
        print("=== Visible thinking ===")
        print(block.thinking)
    elif block.type == "text":
        print("\n=== Final answer ===")
        print(block.text)

# output_tokens is the total billed output, including thinking and answer tokens.
# thinking_tokens reports how many of those tokens were used for thinking.
details = response.usage.output_tokens_details
print("\n=== Token usage ===")
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Thinking tokens: {details['thinking_tokens']}")
print(f"Total output tokens: {response.usage.output_tokens}")
