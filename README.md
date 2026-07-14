# Using LLM APIs — Primer

A collection of minimal examples for calling different LLM providers via their Python SDKs. Each snippet follows the same pattern: load a key, send a prompt, print the response.

---

## Setup

```bash
git clone https://github.com/Walid-Ahmed/llm-api-primer.git
cd llm-api-primer
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys — the `.env` file is git-ignored and should never be committed:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
```

---

## Recommended Learning Order

Go through the files in this order if you are learning LLM APIs from scratch:

1. [`openai/openai_basic.py`](openai/openai_basic.py) — start with the smallest OpenAI chat request.
2. [`openai/gpt.py`](openai/gpt.py) — separate the system prompt, user prompt, and message list.
3. [`openai/open_ai_stream.py`](openai/open_ai_stream.py) — stream tokens as they arrive instead of waiting for the full answer.
4. [`openai/openai_responses.py`](openai/openai_responses.py) — compare Chat Completions with the newer Responses API.
5. [`openai/openai_cot_thinking.py`](openai/openai_cot_thinking.py) — compare a requested explanation with built-in reasoning and an optional reasoning summary.
6. [`anthropic/anthropic_claude.py`](anthropic/anthropic_claude.py) — learn Claude's message format and separate `system` parameter.
7. [`anthropic/claude_stream.py`](anthropic/claude_stream.py) — stream Claude responses.
8. [`Anthropic/anthropic_thinking.py`](Anthropic/anthropic_thinking.py) — print Claude's visible thinking, final answer, and thinking-token usage.
9. [`google/google_gemini.py`](google/google_gemini.py) — see how Gemini handles system instructions.
10. [`deepseek.py`](deepseek.py) — reuse the OpenAI client with a different provider via `base_url`.
11. [`ollama_api.py`](ollama_api.py) — run a local model with Ollama.
12. [`simple_chatbot/chat_bot1.py`](simple_chatbot/chat_bot1.py) — understand multi-turn memory with a simple history list.
13. [`simple_chatbot/simple_chatbot.py`](simple_chatbot/simple_chatbot.py) — run an interactive terminal chatbot.
14. [`simple_chatbot/simple_chatbot2.py`](simple_chatbot/simple_chatbot2.py) — choose the assistant personality before chatting.
15. [`simple_chatbot/chatbot_langchain.py`](simple_chatbot/chatbot_langchain.py) — compare the plain Python chatbot with LangChain's history wrapper.
16. [`two_model_chat/gpt_claude/gpt_claude.py`](two_model_chat/gpt_claude/gpt_claude.py) — watch GPT and Claude talk to each other.
16. [`two_model_chat/ollama_dual_chat/ollama_dual_chat.py`](two_model_chat/ollama_dual_chat/ollama_dual_chat.py) — repeat the dual-chat pattern using local Ollama models.
17. [`notebooks/`](notebooks/) — every script above also has a notebook version, at the same relative path (e.g. `openai/gpt.py` → `notebooks/openai/gpt.ipynb`). Use whichever form you prefer for experimentation.

Suggested path: first understand one provider, then streaming, then reasoning, then chat memory, then multi-model conversations.

---

## Templates by Provider

### Anthropic — Claude

```python
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Tell me a joke"}]
)

print(response.content[0].text)
```

**With streaming:**

```python
result = client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    temperature=0.7,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Tell me a joke"}],
)

with result as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

**With extended thinking:** See [Anthropic/anthropic_thinking.py](Anthropic/anthropic_thinking.py).

```python
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=2048,
    thinking={"type": "enabled", "budget_tokens": 1024},
    messages=[{"role": "user", "content": "Solve this problem"}],
)

for block in response.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking)
    elif block.type == "text":
        print("Answer:", block.text)

print(
    "Thinking tokens:",
    response.usage.output_tokens_details["thinking_tokens"],
)
```

The returned thinking block may be summarized and is not guaranteed to contain the exact raw internal reasoning. Thinking tokens are included in billed output tokens, and `budget_tokens` must be lower than `max_tokens`.

**Models:** `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`

---

### OpenAI — GPT

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke"}
    ]
)

print(response.choices[0].message.content)
```

**With streaming:**

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke"}
    ],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```

**Models:** `gpt-4o`, `gpt-4o-mini`, `gpt-4.1-mini`

---

### OpenAI — Explanations & Built-in Reasoning

Three related approaches. See [openai/openai_cot_thinking.py](openai/openai_cot_thinking.py).

**Requested step-by-step explanation** (any model) — asks the model to include an explanation in its visible reply. This is generated answer text, not access to its raw internal reasoning:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Think step by step before giving your final answer."},
        {"role": "user", "content": "Your question here"},
    ],
)
print(response.choices[0].message.content)  # includes the reasoning steps
```

**Built-in reasoning** — a reasoning model spends internal reasoning tokens before answering. OpenAI recommends the Responses API for reasoning workloads:

```python
response = client.responses.create(
    model="gpt-5-nano",  # Lowest-cost hosted OpenAI reasoning model
    reasoning={"effort": "medium"},
    input="Your question here",
)
# output_text contains the final answer, not the raw internal reasoning.
print(response.output_text)
# response.usage.output_tokens_details.reasoning_tokens
```

**Optional reasoning summary** — supported models can return a model-generated summary. It is not raw chain-of-thought:

```python
response = client.responses.create(
    model="gpt-5-nano",
    reasoning={"effort": "medium", "summary": "auto"},
    input="Your question here",
)
for item in response.output:
    if item.type == "reasoning":
        for summary_part in item.summary:
            print(summary_part.text)
```

**Sample output:**

```text
=== Requested step-by-step explanation (gpt-4o-mini) ===
Let's define the cost of the ball as x dollars. According to the problem, the bat costs $1.00 more
than the ball, which can be expressed as x + 1.00 dollars.

Now, we can set up the equation based on the total cost:
  x + (x + 1.00) = 1.10
  2x + 1.00 = 1.10
  2x = 0.10
  x = 0.05

Thus, the cost of the ball is 0.05 dollars, or 5 cents.
The ball costs $0.05 (5 cents).

=== Built-in reasoning (gpt-5-nano, effort=medium) ===
Answer: The ball costs $0.05 (5 cents).
Tokens — input: <varies>, reasoning: <varies>, output: <varies>
```

The token breakdown shows that reasoning tokens were used internally. Raw reasoning does not appear in `response.output_text`; an optional summary must be requested explicitly. Exact token counts vary by request and response.

**Model:** `gpt-5-nano`

---

### OpenAI — Responses API (newer)

`responses.create()` is the modern OpenAI API (2025+). Key differences from `chat.completions`:

- `instructions` replaces the `system` message role
- Output is at `response.output_text` instead of `response.choices[0].message.content`
- Natively supports built-in tools like `web_search` and `file_search`

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a helpful assistant.",
    input="Tell me a joke",
)

print(response.output_text)
```

See [openai/openai_responses.py](openai/openai_responses.py) for the full example.

---

### Google — Gemini

```python
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gemini = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction="You are a helpful coding assistant."
)

response = gemini.generate_content("Tell me a Python joke.")
print(response.text)
```

**Models:** `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`

---

### DeepSeek (via OpenAI-compatible client)

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke"}
    ]
)

print(response.choices[0].message.content)
```

**Models:** `deepseek-chat`, `deepseek-reasoner`

---

### Ollama — Local Models

```python
import ollama

response = ollama.chat(
    model="llama3.2",   # any model you have pulled locally
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke"}
    ]
)

print(response['message']['content'])
```

Pull a model first: `ollama pull llama3.2`

**Models:** `llama3.2`, `phi3`, `mistral`, `gemma3` — see `ollama list`

---

## Multi-Model Examples

### GPT vs Claude — Dual Conversation

Two models hold a back-and-forth conversation, each with a distinct personality. See [two_model_chat/gpt_claude/gpt_claude.py](two_model_chat/gpt_claude/gpt_claude.py).

```text
GPT  → argumentative, snarky
Claude → polite, finds common ground
```

### Ollama Dual Chat

Two local Ollama models conversing with each other. See [two_model_chat/ollama_dual_chat/ollama_dual_chat.py](two_model_chat/ollama_dual_chat/ollama_dual_chat.py).

---

## Project Structure

```text
using_llm_api/
├── anthropic/
│   ├── anthropic_claude.py      # Basic Claude call
│   ├── anthropic_thinking.py    # Extended thinking + token usage
│   └── claude_stream.py         # Streaming Claude call
├── openai/
│   ├── openai_basic.py          # Basic GPT call
│   ├── open_ai_stream.py        # Streaming GPT call
│   ├── openai_cot_thinking.py   # Requested explanation + built-in reasoning
│   └── gpt.py
├── google/
│   └── google_gemini.py         # Gemini call
├── deepseek.py                  # DeepSeek via OpenAI client
├── ollama_api.py                # Local Ollama call
├── simple_chatbot/              # Simple chatbot examples
├── two_model_chat/
│   ├── gpt_claude/              # GPT ↔ Claude conversation
│   └── ollama_dual_chat/        # Local model ↔ model conversation
└── notebooks/                   # Jupyter notebooks
```
