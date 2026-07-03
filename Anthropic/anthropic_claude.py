import os
from dotenv import load_dotenv
import anthropic

# override=True forces dotenv to overwrite any vars already set in the shell
load_dotenv(override=True)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env")

client = anthropic.Anthropic(api_key=anthropic_api_key)

system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

# Anthropic keeps the system instruction separate from the user messages.
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    # max_tokens sets the maximum number of tokens Claude can generate
    # in the response. It does not count the input prompt tokens.
    # Lower values make shorter/cheaper replies; higher values allow longer replies.
    max_tokens=200,
    system=system_message,
    messages=[{"role": "user", "content": user_prompt}]
)

# response.content is a list of content blocks; [0].text extracts the text
print("Claude response:\n", response.content[0].text)
