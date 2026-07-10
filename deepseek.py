# Run with: python deepseek.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# DeepSeek uses its own key, but the request format is OpenAI-compatible.
load_dotenv(override=True)
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# Fail early with a clear message if the key is missing.
if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY not found in .env")

# Separate the assistant behavior from the user's actual task.
system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

# Chat models expect a list of role-tagged messages.
prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

# DeepSeek exposes an OpenAI-compatible API, so we reuse the OpenAI client
# by pointing base_url at DeepSeek's endpoint instead
client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=prompts,
)

# choices[0] is the first response candidate returned by the chat API.
print(response.choices[0].message.content)
