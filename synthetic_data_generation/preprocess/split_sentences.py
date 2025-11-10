import json
import re

# Load input
with open("/home/spshetty/RadAnnotate/data_generation/misc/finetune_train.json", "r") as f:
    data = json.load(f)
result = []

for item in data:
    report = item.get("Report", "")
    annotations = item.get("Annotated Report", {})

    # Build normalized label map: token.lower() → (original_token, label)
    label_map = {}
    for ann in annotations.values():
        token = ann["tokens"]
        label = ann["label"]
        label_map[token.lower()] = (token, label)

    # Split report into sentences
    sentences = re.split(r'(?<=[.!?])\s+', report)

    for sent in sentences:
        tokens = re.findall(r"\w+", sent)
        labels = {}

        for token in tokens:
            t_lower = token.lower()
            if t_lower in label_map:
                original_token, label = label_map[t_lower]
                labels[original_token] = label

        if labels:  # Only include sentences with at least one label
            result.append({
                "Report": sent,
                "Labels": labels
            })

# Save to file
with open("/home/spshetty/RadAnnotate/data_generation/nf_new_data/sentence_split_output.json", "w") as f:
    json.dump(result, f, indent=2)
