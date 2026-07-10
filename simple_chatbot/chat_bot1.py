# Run with: python simple_chatbot/chat_bot1.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY from .env and create the SDK client.
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# This behavior instruction is added to every request.
system_message = "You are a helpful assistant."

def chat(messages, user_message):
    """Send one user turn, save the assistant turn, and return updated history."""
    # Keep only user/assistant turns in history; add the system message at request time.
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        # system message is prepended each call so it isn't stored in the mutable history list
        messages=[{"role": "system", "content": system_message}] + messages
    )

    # Extract the assistant text from the first completion candidate.
    reply = response.choices[0].message.content
    # Mutating and returning messages makes the state change explicit for the caller.
    messages.append({"role": "assistant", "content": reply})
    return reply, messages


# Demonstrate multi-turn memory: the model remembers context across calls
history = []

for user_message in [
    "Hi",
    "What is my name?",
    "My name is Walid",
    "What is my name?",
]:
    # The second "What is my name?" works only because history is preserved.
    print("User:", user_message)
    reply, history = chat(history, user_message)
    print("Bot:", reply)
