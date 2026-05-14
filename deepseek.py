import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY not found in .env")

system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

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

print(response.choices[0].message.content)
