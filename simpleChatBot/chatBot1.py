import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_message = "You are a helpful assistant."

def chat(messages, user_message):
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        # system message is prepended each call so it isn't stored in the mutable history list
        messages=[{"role": "system", "content": system_message}] + messages
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply, messages


# Demonstrate multi-turn memory: the model remembers context across calls
history = []

reply, history = chat(history, "Hi")
print("Bot:", reply)

reply, history = chat(history, "What is my name?")
print("Bot:", reply)

reply, history = chat(history, "My name is Walid")
print("Bot:", reply)

reply, history = chat(history, "What is my name?")
print("Bot:", reply)
