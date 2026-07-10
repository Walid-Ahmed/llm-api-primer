# Run with: python openai/openai_responses.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY from .env and create the SDK client.
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# The Responses API is OpenAI's newer text-generation API.
# Compared with Chat Completions:
#   - `instructions` is where you put the system/developer prompt.
#   - `input` is the user's message or task.
#   - `response.output_text` gives you the final assistant text directly.

response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a helpful assistant.",
    input="Tell me a joke",
)

# output_text is the easiest way to read the final text from a Responses API call.
print(response.output_text)
