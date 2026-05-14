import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_message = "You are a helpful assistant."

def call_gpt(history):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )
    return response.choices[0].message.content

def chat_loop():
    # history carries the full conversation so the model has memory across turns
    history = [{"role": "system", "content": system_message}]
    print("Chat started! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() in ["exit", "quit", "bye"]:
            print("Chat ended.")
            break

        history.append({"role": "user", "content": user_input})
        assistant_response = call_gpt(history)
        history.append({"role": "assistant", "content": assistant_response})

        print(f"Assistant: {assistant_response}\n")

if __name__ == "__main__":
    chat_loop()
