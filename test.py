from ollama import Client

client = Client()

response = client.chat(
    model="mistral",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response['message']['content'])

