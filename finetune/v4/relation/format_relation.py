import json

# Load your annotated dataset
with open("/home/spshetty/RadAnnotate/data_generation/finetune_train.json", "r") as f:
    raw_data = json.load(f)

# Convert a single example using gold relations
def convert_from_gold_relations(entry):
    report = entry["Report"]
    annotations = entry["Annotated Report"]

    entity_map = {
        ent_id: {
            "tokens": ann["tokens"],
            "label": ann["label"],
            "relations": ann.get("relations", [])
        }
        for ent_id, ann in annotations.items()
    }

    relations = []
    for head_id, head_info in entity_map.items():
        for rel in head_info["relations"]:
            relation_type, tail_id = rel
            if tail_id in entity_map:
                tail_info = entity_map[tail_id]
                relations.append({
                    "head": head_info["tokens"],
                    "tail": tail_info["tokens"],
                    "relation": relation_type
                })

    return {
        "instruction": "Extract valid relations between labeled entities in the report.",
        "input": report,
        "output": json.dumps(relations)
    }

# Process all entries
formatted_data = []
for entry in raw_data:
    try:
        formatted_data.append(convert_from_gold_relations(entry))
    except Exception as e:
        print(f"❌ Failed to process: {entry.get('Report', '')[:60]}... → {e}")

# Save to JSONL
with open("relation_gold_finetune.jsonl", "w") as f:
    for item in formatted_data:
        f.write(json.dumps(item) + "\n")

print(f"✅ Created {len(formatted_data)} relation fine-tune examples using annotated relations.")
