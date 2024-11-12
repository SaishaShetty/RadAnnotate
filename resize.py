import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
)

print(torch.cuda.get_device_name(0))
base_model = "/home/spshetty/RadAnnotate/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Set a default chat template (e.g., "assistant")
tokenizer.chat_template = {
    "system": "System: {content}",
    "user": "User: {content}",
    "assistant": "Assistant: {content}"
}

def format_chat_template(row):
    # Manually format the text
    formatted_text = f"System: {row.get('Instruction', '')}\nUser: {row.get('Report (Input)', '')}\nAssistant: {row.get('Annotated Report (Output)', '')}"
    
    row["text"] = formatted_text
    return row

def test_format_chat_template(row):
    formatted_row = format_chat_template(row)
    print(f"Formatted Text: {formatted_row['text']}")
    
    # Tokenize the formatted text
    tokenized_output = tokenizer(formatted_row['text'], return_tensors='pt')
    print(f"Tokenized Output: {tokenized_output['input_ids']}")
    
    # Decode the tokens to ensure proper tokenization
    decoded_text = tokenizer.decode(tokenized_output['input_ids'][0])
    print(f"Decoded Text: {decoded_text}")

# Example row
example_row = {
    "Instruction": "Annotate the report.",
    "Report (Input)": "Patient shows signs of pneumonia.",
    "Annotated Report (Output)": "OBS-DP: pneumonia"
}

# Run the test
test_format_chat_template(example_row)
