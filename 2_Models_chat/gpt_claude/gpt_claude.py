import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

load_dotenv(override=True)

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openAI_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

gpt_model = "gpt-4.1-mini"
claude_model = "claude-3-5-haiku-latest"

# Give each model a contrasting personality so the conversation is interesting
gpt_system = (
    "You are a chatbot who is very argumentative; "
    "you disagree with anything in the conversation and you challenge everything, in a snarky way."
)

claude_system = (
    "You are a very polite, courteous chatbot. You try to agree with "
    "everything the other person says, or find common ground. "
    "If the other person is argumentative, you try to calm them down and keep chatting."
)

# Seed messages start the conversation without requiring an API call
gpt_messages = ["Hi there"]
claude_messages = ["Hi"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "conversation_log.txt")
MSG_LOG_FILE = os.path.join(BASE_DIR, "messages_log.txt")

def log_to_file(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_messages_to_file(label: str, messages: list):
    with open(MSG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n--- {label} ---\n")
        for m in messages:
            f.write(f"{m['role'].upper()}: {m['content']}\n")
        f.write("\n" + "=" * 40 + "\n")

def call_gpt():
    # GPT sees its own turns as "assistant" and Claude's turns as "user"
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude_msg in zip(gpt_messages, claude_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": claude_msg})
    log_messages_to_file("GPT Input Messages", messages)
    completion = openAI_client.chat.completions.create(model=gpt_model, messages=messages)
    return completion.choices[0].message.content

def call_claude():
    # Claude sees GPT's turns as "user" and its own turns as "assistant"
    # The role mapping is intentionally the mirror image of call_gpt()
    messages = []
    for gpt, claude_msg in zip(gpt_messages, claude_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": claude_msg})
    messages.append({"role": "user", "content": gpt_messages[-1]})
    log_messages_to_file("Claude Input Messages", [{"role": "system", "content": claude_system}] + messages)
    message = claude.messages.create(
        model=claude_model,
        system=claude_system,
        messages=messages,
        max_tokens=500
    )
    return message.content[0].text

# Reset logs each run
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== Conversation Log ===\n\n")
with open(MSG_LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== Messages Log (system/user/assistant) ===\n\n")

print(f"GPT:\n{gpt_messages[0]}\n")
print(f"Claude:\n{claude_messages[0]}\n")
log_to_file(f"GPT: {gpt_messages[0]}")
log_to_file(f"Claude: {claude_messages[0]}")

for i in range(5):
    gpt_next = call_gpt()
    print(f"GPT:\n{gpt_next}\n")
    gpt_messages.append(gpt_next)
    log_to_file(f"GPT: {gpt_next}")

    claude_next = call_claude()
    print(f"Claude:\n{claude_next}\n")
    claude_messages.append(claude_next)
    log_to_file(f"Claude: {claude_next}")

print(f"\nConversation saved to: {LOG_FILE}")
print(f"Messages (with roles) saved to: {MSG_LOG_FILE}")
