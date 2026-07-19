from langchain_ollama import ChatOllama

# Load the model
llm = ChatOllama(model="qwen3:8b")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = llm.invoke(question)

    print("\nAI:", response.content)
    print("-" * 50)