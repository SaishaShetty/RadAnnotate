import json

input_file = "/home/spshetty/RadAnnotate/finetune/v4/testing/test_sentence_label.json"   
output_file = "/home/spshetty/RadAnnotate/finetune/v4/testing/gold_data/test_set.json"
with open(input_file, "r") as f:
    data = json.load(f)

converted = []

for item in data:
    report = item.get("Report", "")
    entity_dict = item.get("entities", {})

    formatted_entities = []
    for token, label in entity_dict.items():
        formatted_entities.append({
            "entity_type": label,
            "entity_value": token
        })

    converted.append({
        "Report": report,
        "entities": formatted_entities
    })

with open(output_file, "w") as f:
    json.dump(converted, f, indent=2)

print(f"Converted {len(converted)} reports and saved to {output_file}")