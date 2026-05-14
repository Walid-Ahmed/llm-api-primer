import config  # loads API keys and sets up the OpenAI client via config.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    # MessagesPlaceholder injects the session's message history at this position
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

chain = prompt | llm

# In-memory store keyed by session_id — each session keeps its own history
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# RunnableWithMessageHistory automatically loads and saves history around each invoke
conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

configuration = {"configurable": {"session_id": "user1"}}

print(conversation_chain.invoke({"input": "Hello, who are you?"}, configuration))
print(conversation_chain.invoke({"input": "Can you remind me what I just asked?"}, configuration))
