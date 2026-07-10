# Run with: python openai/gpt.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env before creating the provider client.
# override=True lets values in .env replace any matching shell variables.
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Fail early with a clear message if the key is missing.
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

# The client object is reused for every OpenAI request in this script.
client = OpenAI(api_key=openai_api_key)

# Keep the assistant behavior and user task separate for readability.
system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

# Chat Completions expects a list of role-tagged messages.
prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=prompts
)

# choices[0] is the first response candidate returned by the chat API.
print("GPT response:\n", completion.choices[0].message.content)
