import json
from ollama import chat

def consultRelationDetector(report_text, entity_list):
    entity_map = {ent["entity_value"]: ent["entity_type"] for ent in entity_list}

    prompt = f"""
<s><INST>
You are given a clinical radiology sentence and a list of labeled entities. Your task is to extract relationships **only** between the provided entities, based on the rules below.

Sentence:
"{report_text}"

Entities:
{json.dumps(entity_map, indent=2)}

✳️ **Instructions**:
- Only use the given entity values. Do not create new phrases or alter them.
- Do not invent entities or relations that are not explicitly present.
- Output only valid relations between entity pairs using this JSON format:

[
  {{
    "head": "<entity1>",
    "tail": "<entity2>",
    "relation": "<relation_type>"
  }}
]

📏 **Allowed Relations**:
1. `suggestive_of` — between two OBS-* entities.
2. `modify` — between:
   - two OBS-* entities, or
   - two ANAT-DP entities.
3. `located_at` — only between OBS-* → ANAT-DP.

🚫 **Restrictions**:
- Use only the entity_value as-is — no merging, paraphrasing, or modification.
- Do not include any relation unless it strictly follows the pairing rules.
- If no valid relations exist, return an empty list: `[]`.

🔁 Example of correct logic:
- ✅ OBS-DP → ANAT-DP → `located_at`
- ✅ OBS-DP → OBS-DP → `suggestive_of` or `modify`
- ✅ ANAT-DP → ANAT-DP → `modify`
- ❌ ANAT-DP → OBS-DP → Invalid
- ❌ Any relation with an entity not in the list → Invalid

Return only the result.
</INST>
"""


    response = chat(model="llama2", messages=[{"role": "user", "content": prompt}])

    try:
        return json.loads(response.message.content)
    except Exception as e:
        print("Failed to parse relation response:", e)
        print("Raw response:\n", response.message.content)
        return []

# Load entire list of data entries
with open("/home/spshetty/RadAnnotate/finetune/v4/testing/res/annotated_reports_test.json", "r") as f:
    data = json.load(f)

# Use only the second entry
sample = data[1]

# Run relation extraction
relations = consultRelationDetector(sample["Report"], sample["entities"])

# Display results
print("\nExtracted Relations for data[1]:")
print(json.dumps(relations, indent=2))
