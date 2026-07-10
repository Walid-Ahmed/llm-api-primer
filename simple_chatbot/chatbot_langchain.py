# Run with: python simple_chatbot/chatbot_langchain.py
import config  # loads OPENAI_API_KEY from .env and stores model names in config.py

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# ChatOpenAI is LangChain's wrapper around OpenAI chat models.
llm = ChatOpenAI(
    model=config.CHAT_MODEL,
    temperature=0.7,
)

# This list is the chatbot's memory.
# We send the whole list to the model every turn.
history = [
    SystemMessage(content="You are a helpful assistant.")
]


def chat(user_message):
    # Save the user's message in memory.
    history.append(HumanMessage(content=user_message))

    # Ask the model using the full conversation so far.
    response = llm.invoke(history)

    # Save the assistant's reply too, so the next turn can remember it.
    history.append(AIMessage(content=response.content))

    return response.content


for user_message in [
    "Hello, who are you?",
    "Can you remind me what I just asked?",
]:
    print("User:", user_message)
    print("Assistant:", chat(user_message))
