import ollama

# Ollama runs models locally — no API key needed.
# Pull a model first with: ollama pull llama3.2
response = ollama.chat(
    model="llama3.2",  # swap with any model you have pulled, e.g. "phi3", "gemma3"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke"}
    ]
)

print(response['message']['content'])
