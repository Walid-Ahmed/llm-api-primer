import ollama

# Ollama runs models locally — no API key needed.
# Pull a model first with: ollama pull llama3.2
# On macOS, pulled models are stored locally under: ~/.ollama/models
# They are managed by Ollama, not saved inside this project folder.
# Use `ollama list` to see downloaded models and `ollama rm llama3.2` to remove one.
response = ollama.chat(
    model="llama3.2",  # swap with any model you have pulled, e.g. "phi3", "gemma3"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke"}
    ]
)

print(response['message']['content'])
