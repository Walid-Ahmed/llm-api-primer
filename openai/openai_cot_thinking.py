import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROBLEM = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"

# ── 1. Chain-of-Thought via prompt (works with any model) ──────────────────
# This demonstrates visible reasoning in the model's answer.
# It is useful for learning, but the reasoning text becomes part of the output.
print("=== CoT via prompt (gpt-4o-mini) ===")
cot_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
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

# Token usage — includes reasoning tokens consumed internally
if (usage := thinking_response.usage) and usage.completion_tokens_details:
    print(f"Tokens — prompt: {usage.prompt_tokens}, "
          f"reasoning: {usage.completion_tokens_details.reasoning_tokens}, "
          f"completion: {usage.completion_tokens}")
