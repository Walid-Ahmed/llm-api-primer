import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def choose_system_message():
    """Let the user pick the assistant's personality before the chat begins."""
    options = {
        "1": "You are a helpful assistant.",
        "2": "You are a funny assistant. Always add humor or jokes.",
        "3": "You are a formal assistant. Always respond politely and professionally.",
        "4": "You are a concise assistant. Reply with short, to-the-point answers."
    }

    print("Choose assistant style:")
    for key, value in options.items():
        print(f"{key}. {value}")

    choice = input("Enter number (default 1): ").strip()
    return options.get(choice, options["1"])

def call_gpt(history):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )
    return response.choices[0].message.content

def chat_loop():
    system_message = choose_system_message()
    # history carries the full conversation so the model has memory across turns
    history = [{"role": "system", "content": system_message}]

    print("\nChat started! Type 'exit' to quit.\n")

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
