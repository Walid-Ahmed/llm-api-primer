# Using LLM APIs — Primer

A collection of minimal examples for calling different LLM providers via their Python SDKs. Each snippet follows the same pattern: load a key, send a prompt, print the response.

---

## Setup

```bash
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
5. [`openai/openai_cot_thinking.py`](openai/openai_cot_thinking.py) — compare visible chain-of-thought prompting with hidden o-series reasoning.
6. [`anthropic/anthropic_claude.py`](anthropic/anthropic_claude.py) — learn Claude's message format and separate `system` parameter.
7. [`anthropic/claude_stream.py`](anthropic/claude_stream.py) — stream Claude responses.
8. [`google/google_gemini.py`](google/google_gemini.py) — see how Gemini handles system instructions.
9. [`deepseek.py`](deepseek.py) — reuse the OpenAI client with a different provider via `base_url`.
10. [`ollama_api.py`](ollama_api.py) — run a local model with Ollama.
11. [`simple_chatbot/chat_bot1.py`](simple_chatbot/chat_bot1.py) — understand multi-turn memory with a simple history list.
12. [`simple_chatbot/simple_chatbot.py`](simple_chatbot/simple_chatbot.py) — run an interactive terminal chatbot.
13. [`simple_chatbot/simple_chatbot2.py`](simple_chatbot/simple_chatbot2.py) — choose the assistant personality before chatting.
14. [`simple_chatbot/chatbot_langchain.py`](simple_chatbot/chatbot_langchain.py) — compare the plain Python chatbot with LangChain's history wrapper.
15. [`two_model_chat/gpt_claude/gpt_claude.py`](two_model_chat/gpt_claude/gpt_claude.py) — watch GPT and Claude talk to each other.
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

### OpenAI — Chain-of-Thought & Thinking

Two complementary reasoning approaches. See [openai/openai_cot_thinking.py](openai/openai_cot_thinking.py).

**CoT via prompt** (any model) — the model reasons visibly inside its reply:

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

**Built-in Thinking** (o-series models) — the model reasons internally before answering:

```python
response = client.chat.completions.create(
    model="o4-mini",
    reasoning_effort="medium",  # "low" | "medium" | "high"
    messages=[{"role": "user", "content": "Your question here"}],
)
# msg.content is the final answer ONLY — reasoning is hidden server-side
# and never returned in the response body. The only trace is the token count:
# response.usage.completion_tokens_details.reasoning_tokens
print(response.choices[0].message.content)
```

> **Note:** This is the key difference from Anthropic's extended thinking — Claude can optionally return its thinking content blocks so you can read the reasoning; OpenAI keeps it fully hidden.

**Sample output:**

```text
=== CoT via prompt (gpt-4o-mini) ===
Let's define the cost of the ball as x dollars. According to the problem, the bat costs $1.00 more
than the ball, which can be expressed as x + 1.00 dollars.

Now, we can set up the equation based on the total cost:
  x + (x + 1.00) = 1.10
  2x + 1.00 = 1.10
  2x = 0.10
  x = 0.05

Thus, the cost of the ball is 0.05 dollars, or 5 cents.
The ball costs $0.05 (5 cents).

=== Built-in Thinking (o4-mini, effort=medium) ===
Answer: The ball costs $0.05 (5 cents).
Tokens — prompt: 38, reasoning: 128, output: 23
```

The token breakdown confirms the thinking happened: **128 reasoning tokens** were consumed silently before the final answer — none of that content appears in `msg.content`. Exact token counts vary by model and response.

**Models:** `o4-mini`, `o3`, `o3-mini`, `o1`

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
│   └── claude_stream.py         # Streaming Claude call
├── openai/
│   ├── openai_basic.py          # Basic GPT call
│   ├── open_ai_stream.py        # Streaming GPT call
│   ├── openai_cot_thinking.py   # CoT via prompt + built-in thinking (o-series)
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
