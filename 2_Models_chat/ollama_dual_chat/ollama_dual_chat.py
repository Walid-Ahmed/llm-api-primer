import os
import ollama

# -------------------------------
# Models (make sure you pulled them with: ollama pull llama3.2 && ollama pull gemma:2b)
# -------------------------------
model_1 = "llama3.2"   # Snarky argumentative bot
model_2 = "phi3"       # Polite bot

# -------------------------------
# System prompts (personalities)
# -------------------------------
system_prompt_1 = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

system_prompt_2 = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

# -------------------------------
# Conversation history
# -------------------------------
bot1_messages = ["Hi there"]
bot2_messages = ["Hi"]

# -------------------------------
# Log files
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONV_LOG = os.path.join(BASE_DIR, "conversation_log.txt")
MSG_LOG = os.path.join(BASE_DIR, "messages_log.txt")

def log_conv(text: str):
    """Append dialogue lines to conversation log."""
    with open(CONV_LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_msgs(label: str, msgs: list):
    """Append raw messages to message log."""
    with open(MSG_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n--- {label} ---\n")
        for m in msgs:
            f.write(f"{m['role'].upper()}: {m['content']}\n")
        f.write("\n" + "="*40 + "\n")

# -------------------------------
# Call Ollama wrapper
# -------------------------------
def call_ollama(model, messages):
    response = ollama.chat(model=model, messages=messages)
    return response['message']['content']

# -------------------------------
# Bot 1 (snarky)
# -------------------------------
def call_bot1():
    msgs = [{"role": "system", "content": system_prompt_1}]
    # Bot 1 treats its prior lines as assistant turns and Bot 2 as the user.
    for b1, b2 in zip(bot1_messages, bot2_messages):
        msgs.append({"role": "assistant", "content": b1})
        msgs.append({"role": "user", "content": b2})
    log_msgs(f"Bot1 ({model_1}) input", msgs)
    return call_ollama(model_1, msgs)

# -------------------------------
# Bot 2 (polite)
# -------------------------------
def call_bot2():
    msgs = [{"role": "system", "content": system_prompt_2}]
    # Bot 2 gets the mirrored role mapping so the dialogue stays coherent.
    for b1, b2 in zip(bot1_messages, bot2_messages):
        msgs.append({"role": "user", "content": b1})
        msgs.append({"role": "assistant", "content": b2})
    msgs.append({"role": "user", "content": bot1_messages[-1]})
    log_msgs(f"Bot2 ({model_2}) input", msgs)
    return call_ollama(model_2, msgs)

# -------------------------------
# Main loop
# -------------------------------
def main():
    # Reset logs
    with open(CONV_LOG, "w", encoding="utf-8") as f:
        f.write("=== Conversation Log ===\n\n")
    with open(MSG_LOG, "w", encoding="utf-8") as f:
        f.write("=== Messages Log (system/user/assistant) ===\n\n")

    # Initial greetings
    print(f"Bot1:\n{bot1_messages[0]}\n")
    print(f"Bot2:\n{bot2_messages[0]}\n")
    log_conv(f"Bot1: {bot1_messages[0]}")
    log_conv(f"Bot2: {bot2_messages[0]}")

    rounds = 5
    for i in range(rounds):
        # Bot1 responds
        b1_next = call_bot1()
        print(f"Bot1:\n{b1_next}\n")
        bot1_messages.append(b1_next)
        log_conv(f"Bot1: {b1_next}")

        # Bot2 responds
        b2_next = call_bot2()
        print(f"Bot2:\n{b2_next}\n")
        bot2_messages.append(b2_next)
        log_conv(f"Bot2: {b2_next}")

    print(f"\nConversation saved to: {CONV_LOG}")
    print(f"Messages (with roles) saved to: {MSG_LOG}")

if __name__ == "__main__":
    main()
