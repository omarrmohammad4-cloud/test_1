"""Inventory extraction pipeline with schema enforcement and model constraints.

This script shows two constraint layers:
1) prompt-level constraints in `system_instruction`
2) schema and validation constraints in `MerchantRequest` and `parse_inventory`

Add more prompt-level constraints inside `system_instruction` if you want the model
output to be stricter, or add more Python-side validation in `parse_inventory`.
"""

from pydantic import BaseModel
from typing import Optional, Literal
import ollama


class MerchantRequest(BaseModel):
    action_type: str
    item_name: str
    quantity: int

    urgency_level: Optional[Literal["low", "normal", "high", "critical"]] = "normal"


# Add more prompt-level constraints here if you want the model to be stricter.
# Examples: require certain field ordering, reject extra fields, enforce exact terminology,
# or explicitly require the presence of a specific field name.
system_instruction = f"""
You are a strict data extraction system for an administrative merchant tool.
Your only job is to extract inventory request details from raw text and output them as valid JSON.
Never include conversational filler.
Do not include markdown formatting outside of the JSON block.
If you cannot find a required value, output null.

Schema to follow:
{MerchantRequest.model_json_schema()}
"""


def extract_inventory(raw_input: str) -> str:
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": raw_input}
        ],
        format=MerchantRequest.model_json_schema(),
        options={"temperature": 0.1}
    )
    return response["message"]["content"]


def parse_inventory(raw_json: str) -> MerchantRequest:
    # Validate the model response against the schema.
    # This is the second safety layer after the prompt-level constraints.
    # Add more Python-side constraints here if you need stricter validation.
    # Examples:
    # - ensure quantity > 0
    # - reject unknown item names
    # - normalize or sanitize field values
    merchant_request = MerchantRequest.model_validate_json(raw_json)
    if merchant_request.quantity <= 0:
        raise ValueError("Quantity must be greater than zero")
    return merchant_request


def main() -> None:
    raw_input = "Hey, we need a restock of 50 mechanical keyboards for the downtown branch ASAP. Thanks."
    print(f"Processing input: '{raw_input}'...\n")

    raw_output = extract_inventory(raw_input)
    print("Extracted Data:")
    print(raw_output)
    print()

    merchant_request = parse_inventory(raw_output)
    print("Validated merchant request:")
    print(merchant_request)
    print("item_name:", merchant_request.item_name)
    print("quantity:", merchant_request.quantity)


if __name__ == "__main__":
    main()