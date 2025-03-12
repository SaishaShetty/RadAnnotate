import json
from transformers import AutoTokenizer

# Path to your JSON file (adjust path if needed)
json_file = "/home/spshetty/RadAnnotate/finetune/v3/train_set_v3.json"

# Load JSON file
with open(json_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)  # List of dicts (each dict is one training example)

print(f"Loaded {len(dataset)} records.")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Analyze token lengths
lengths = []
for example in dataset:  # Loop through each report
    instruction = example['instruction'].strip()
    input_text = example['input'].strip()
    output_json = json.dumps(example['output'], ensure_ascii=False)  # Serialize output dict to JSON string

    # Prepare full input as it will be during training
    text = f"<s>[INST] {instruction} {input_text} [/INST] {output_json} </s>"

    # Tokenize without truncation to get actual length
    tokenized = tokenizer(text, truncation=False)["input_ids"]
    lengths.append(len(tokenized))  # Store token count

# Statistics
max_len = max(lengths)
avg_len = sum(lengths) / len(lengths)
percent_above_4096 = sum(1 for l in lengths if l > 512) / len(lengths) * 100

# Print results
print(f"Max token length: {max_len}")
print(f"Average token length: {avg_len:.2f}")
print(f"Percentage of samples exceeding 4096 tokens: {percent_above_4096:.2f}%")
