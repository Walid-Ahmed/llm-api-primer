import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Keep reusable prompt pieces separate so they are easy to swap in demos.
system_message = "You are a helpful coding assistant."
user_prompt = "Tell me a Python joke."

# Gemini takes the system instruction in the model constructor, not in generate_content()
# This differs from OpenAI/Anthropic where system prompts are passed per-request
gemini = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=system_message
)

response = gemini.generate_content(user_prompt)
print(response.text)
