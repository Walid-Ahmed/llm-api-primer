import os
from dotenv import load_dotenv
from openai import OpenAI

# override=True lets values in .env replace any matching shell variables.
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=openai_api_key)

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

print("GPT response:\n", completion.choices[0].message.content)
