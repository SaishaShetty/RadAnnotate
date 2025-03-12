import json

# Load JSON file
with open('/home/spshetty/RadAnnotate/finetune/v3/train.json', 'r') as f:
    data = json.load(f)

# Instruction to add
instruction = "You are given a clinical sentence. Label each word or phrase based on the provided schema (ANAT-DP, OBS-DP, OBS-U, etc.). Output the mapping of words to their corresponding labels."

# Process data: filter empty labels and reformat
processed_data = []
for item in data:
    if item['labels']:  # Check if labels is not empty
        processed_item = {
            "instruction": instruction,
            "input": item["sentence"],
            "output": item["labels"]
        }
        processed_data.append(processed_item)

# Save the transformed data back to JSON file
with open('processed_data.json', 'w') as f:
    json.dump(processed_data, f, indent=4)

print("Processed data saved to 'processed_data.json'")
