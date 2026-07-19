from pydantic import BaseModel
from typing import Optional, Literal
import ollama


class MerchantRequest(BaseModel):
    action_type: str
    item_name: str
    quantity: int
  
    urgency_level: Optional[Literal["low", "normal", "high", "critical"]] = "normal"


system_instruction = f"""
You are a strict data extraction system for an administrative merchant tool. 
Your only job is to extract inventory request details from raw text and output them as valid JSON.
Never include conversational filler.
Do not include markdown formatting outside of the JSON block.
If you cannot find a required value, output null.

Schema to follow:
{MerchantRequest.model_json_schema()}
"""


raw_input = "Hey, we need a restock of 50 mechanical keyboards for the downtown branch ASAP. Thanks."

print(f"Processing input: '{raw_input}'...\n")


response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": raw_input}
    ],
    format=MerchantRequest.model_json_schema(),
    options={"temperature": 0.1} 
)


print("Extracted Data:")
merchant_request = MerchantRequest.model_validate_json(
    response["message"]["content"]
)

print(merchant_request)
print(merchant_request.action_type)
print(merchant_request.item_name)
print(merchant_request.quantity)
print(merchant_request.urgency_level)