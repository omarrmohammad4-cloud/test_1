from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b")

system_prompt = """
You are an AI Assistant for an e-commerce platform.

Your responsibilities are:
- Answer merchant questions.
- Generate professional product descriptions.
- Recommend products.
- Explain business information.
- Always respond professionally.
"""

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    prompt = f"""
{system_prompt}

User Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nAI:")
    print(response.content)
    print("-" * 60)