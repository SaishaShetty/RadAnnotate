import json

input_file = "test_set_648.json"   
output_file = "test_set.json"
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