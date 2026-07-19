import ollama

from extractor import MerchantRequest


system_instruction = f"""
You are an information extraction engine.

Your task is to extract structured inventory requests.

Return ONLY valid JSON.

Never explain.

Never add markdown.

If a value is missing, return null.

Follow this schema:
{MerchantRequest.model_json_schema()}

========================
Example 1

Input:
Please restock 20 gaming keyboards.

Output:
{{
    "action_type":"restock",
    "item_name":"gaming keyboards",
    "quantity":20,
    "urgency_level":"normal"
}}

========================
Example 2

Input:
Remove 5 office chairs.

Output:
{{
    "action_type":"remove_stock",
    "item_name":"office chairs",
    "quantity":5,
    "urgency_level":"normal"
}}

========================
Example 3

Input:
We urgently need another 100 USB-C cables.

Output:
{{
    "action_type":"restock",
    "item_name":"USB-C cables",
    "quantity":100,
    "urgency_level":"critical"
}}

Schema to follow:
{MerchantRequest.model_json_schema()}
"""


def extract_with_few_shot(raw_input: str) -> str:
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": raw_input},
        ],
        format=MerchantRequest.model_json_schema(),
        options={"temperature": 0.1},
    )
    return response["message"]["content"]


if __name__ == "__main__":
    raw_input = "Hey, we need a restock of 50 mechanical keyboards for the downtown branch ASAP. Thanks."
    print(f"Processing input: '{raw_input}'...\n")
    print("Extracted Data:")
    print(extract_with_few_shot(raw_input))