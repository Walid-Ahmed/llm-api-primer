# Using LLM APIs — Primer

A collection of minimal examples for calling different LLM providers via their Python SDKs. Each snippet follows the same pattern: load a key, send a prompt, print the response.

---

## Setup

```bash
pip install anthropic openai google-generativeai ollama python-dotenv
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

Two models hold a back-and-forth conversation, each with a distinct personality. See [2_Models_chat/gpt_claude/gpt_claude.py](2_Models_chat/gpt_claude/gpt_claude.py).

```text
GPT  → argumentative, snarky
Claude → polite, finds common ground
```

### Ollama Dual Chat

Two local Ollama models conversing with each other. See [2_Models_chat/ollama_dual_chat/ollama_dual_chat.py](2_Models_chat/ollama_dual_chat/ollama_dual_chat.py).

---

## Project Structure

```text
using_llm_api/
├── Anthropic/
│   ├── anthropic_claude.py      # Basic Claude call
│   └── calude2_stream.py        # Streaming Claude call
├── openai/
│   ├── openai_basic.py          # Basic GPT call
│   ├── open_ai_stream.py        # Streaming GPT call
│   └── gpt.py
├── Google/
│   └── google_gemini.py         # Gemini call
├── deepseek.py                  # DeepSeek via OpenAI client
├── ollama_api.py                # Local Ollama call
├── simpleChatBot/               # Simple chatbot examples
├── 2_Models_chat/
│   ├── gpt_claude/              # GPT ↔ Claude conversation
│   └── ollama_dual_chat/        # Local model ↔ model conversation
└── notebooks/                   # Jupyter notebooks
```
