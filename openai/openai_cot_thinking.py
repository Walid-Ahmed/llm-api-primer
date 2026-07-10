# Run with: python openai/openai_cot_thinking.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROBLEM = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"

# ── 1. Step-by-step explanation via prompt (works with any model) ─────────
# This does NOT enable a special hidden reasoning mode.
# It simply asks the model to show its reasoning in the visible answer.
# Useful for learning, but the reasoning text becomes part of the output.
print("=== Step-by-step explanation via prompt (gpt-4o-mini) ===")
cot_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            # This prompt requests a visible step-by-step explanation.
            # It is different from `reasoning_effort`, which uses hidden reasoning.
            "content": "Think step by step before giving your final answer.",
        },
        {"role": "user", "content": PROBLEM},
    ],
)
print(cot_response.choices[0].message.content)

# ── 2. Built-in Thinking via o-series reasoning model ─────────────────────
# o4-mini reasons internally before answering.
# reasoning_effort: "low" | "medium" | "high"  (controls depth of thought)
# This is better when you want the model to spend more reasoning tokens
# without exposing that internal reasoning to the caller.
print("\n=== Built-in Thinking (o4-mini, effort=medium) ===")
thinking_response = client.chat.completions.create(
    model="o4-mini",
    reasoning_effort="medium",
    messages=[
        {"role": "user", "content": PROBLEM},
    ],
)

msg = thinking_response.choices[0].message

# msg.content is the final answer ONLY — the internal reasoning is NOT returned.
# Unlike Anthropic's extended thinking (which can return thinking blocks),
# OpenAI keeps reasoning fully hidden server-side. The only trace is the
# reasoning_tokens count in usage.completion_tokens_details.
print("Answer:", msg.content)

# Token usage:
#   prompt     = tokens in the input prompt/messages you sent
#   reasoning  = hidden internal thinking tokens used by the reasoning model
#   completion = all completion-side tokens, including hidden reasoning tokens
if (usage := thinking_response.usage) and usage.completion_tokens_details:
    print(f"Tokens — prompt: {usage.prompt_tokens}, "
          f"reasoning: {usage.completion_tokens_details.reasoning_tokens}, "
          f"completion: {usage.completion_tokens}")
