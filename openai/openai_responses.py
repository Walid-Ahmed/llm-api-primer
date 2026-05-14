import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# responses.create() is the newer OpenAI API (2025+), replacing chat.completions.create()
# Key differences:
#   - system prompt is a top-level param, not a message role
#   - output text is at response.output_text (not response.choices[0].message.content)
#   - built-in tools (web_search, file_search) are natively supported

response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a helpful assistant.",
    input="Tell me a joke",
)

print(response.output_text)
