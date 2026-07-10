# Run with: python anthropic/anthropic_claude.py
import os
from dotenv import load_dotenv
import anthropic

# Load API keys from .env before creating the provider client.
# override=True lets values in .env replace matching shell variables.
load_dotenv(override=True)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Fail early with a clear message if the key is missing.
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env")

# The client object is reused for every Anthropic request in this script.
client = anthropic.Anthropic(api_key=anthropic_api_key)

# Keep the system instruction and user prompt separate to make the request shape clear.
system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

# Anthropic keeps the system instruction separate from the user messages.
response = client.messages.create(
    # Haiku is fast and lower-cost, which makes it good for small tutorial demos.
    model="claude-haiku-4-5-20251001",
    # max_tokens sets the maximum number of tokens Claude can generate
    # in the response. It does not count the input prompt tokens.
    # Lower values make shorter/cheaper replies; higher values allow longer replies.
    max_tokens=200,
    system=system_message,
    messages=[{"role": "user", "content": user_prompt}]
)

# response.content is a list of content blocks; [0].text extracts the first text block.
print("Claude response:\n", response.content[0].text)
