# Run with: python simple_chatbot/simple_chatbot.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_message = "You are a helpful assistant."

def call_gpt(history):
    """Send the full conversation history and return the assistant's reply."""
    # `history` is a list of chat messages in OpenAI's chat format:
    # {"role": "system" | "user" | "assistant", "content": "..."}
    # Sending the full list each time is what gives this simple chatbot memory.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )
    return response.choices[0].message.content

def chat_loop():
    # Start the conversation with one system message.
    # The system message sets the assistant's behavior for every turn.
    history = [{"role": "system", "content": system_message}]
    print("Chat started! Type 'exit' to quit.\n")

    while True:
        # input() pauses the program and waits for the user to type a message.
        user_input = input("You: ")

        if user_input.strip().lower() in ["exit", "quit", "bye"]:
            print("Chat ended.")
            break

        # Add the user's new message to the conversation history.
        history.append({"role": "user", "content": user_input})
        assistant_response = call_gpt(history)
        # Add the assistant reply too, otherwise the next turn will not remember it.
        history.append({"role": "assistant", "content": assistant_response})

        print(f"Assistant: {assistant_response}\n")

if __name__ == "__main__":
    chat_loop()
