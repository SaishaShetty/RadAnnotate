import json

# === Path to your JSON file ===

#/home/spshetty/RadAnnotate/finetune/v4/testing_syn/final_annotated_reports_all_3.json
# === Load the JSON ===
import json

# === File paths ===
syn_real_path = "/home/spshetty/RadAnnotate/finetune/v4/testing_syn/final_annotated_reports_(syn+real_3).json"
real_path = "/home/spshetty/RadAnnotate/finetune/v4/testing_syn/final_annotated_reports_all_3.json"

output_syn_real = "cleaned_syn+real.json"
output_real = "cleaned_real.json"

with open(syn_real_path, "r") as f:
    syn_real_data = json.load(f)

with open(real_path, "r") as f:
    real_data = json.load(f)

# === Collect bad indices from either file
bad_indices = set()

for i, entry in enumerate(syn_real_data):
    entities = entry.get("entities", [])
    if not entities or (isinstance(entities, dict) and "error" in entities):
        bad_indices.add(i)

for i, entry in enumerate(real_data):
    entities = entry.get("entities", [])
    if not entities or (isinstance(entities, dict) and "error" in entities):
        bad_indices.add(i)

# === Clear "entities" at those indices in both files
for i in bad_indices:
    syn_real_data[i]["entities"] = []
    real_data[i]["entities"] = []

# === Save aligned outputs
with open(output_syn_real, "w") as f:
    json.dump(syn_real_data, f, indent=2)

with open(output_real, "w") as f:
    json.dump(real_data, f, indent=2)

print(f"✅ Set entities to [] at {len(bad_indices)} common bad indices.")
print(f"📝 Saved aligned syn+real to: {output_syn_real}")
print(f"📝 Saved aligned real to: {output_real}")