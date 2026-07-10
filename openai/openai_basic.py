# Run with: python openai/openai_basic.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY from .env so the key does not live in source code.
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# The SDK reads the key from this client for every request below.
client = OpenAI(api_key=api_key)

# This is the smallest useful chat-completion request: model + messages.
response = client.chat.completions.create(
    model="gpt-4o-mini",  # lightweight, fast, cheap — good default for testing
    # A system message sets behavior; a user message asks for the work.
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke"}
    ]
)

# choices[0] is the first (and usually only) completion candidate
print(response.choices[0].message.content)
