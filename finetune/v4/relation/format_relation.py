import json
import re

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

def fuzzy_match(token, sentence):
    token = token.lower()
    sentence = sentence.lower()
    return token in sentence or any(word in sentence for word in token.split())

def get_entity_map(entity_block):
    return {
        eid: {
            "tokens": ent["tokens"],
            "label": ent["label"],
            "relations": ent.get("relations", [])
        }
        for eid, ent in entity_block.items()
    }

def extract_relations_for_sentence(sentence, entity_map):
    matched_entities = {
        eid: ent for eid, ent in entity_map.items()
        if fuzzy_match(ent["tokens"], sentence)
    }

    relations = []
    for head_id, head in matched_entities.items():
        for rel in head.get("relations", []):
            rel_type, tail_id = rel
            if tail_id in matched_entities:
                relations.append({
                    "head": head["tokens"],
                    "tail": entity_map[tail_id]["tokens"],
                    "relation": rel_type
                })

    return matched_entities, relations

def generate_samples(report_text, entity_block, instruction="Extract valid relations between labeled entities in the report."):
    samples = []
    sentences = split_into_sentences(report_text)
    entity_map = get_entity_map(entity_block)

    for sentence in sentences:
        matched_entities, matched_relations = extract_relations_for_sentence(sentence, entity_map)
        if not matched_entities:
            continue

        entity_dict = {e["tokens"]: e["label"] for e in matched_entities.values()}
        sample = {
            "instruction": instruction,
            "input": f"Report: {sentence}\nEntities: {json.dumps(entity_dict)}",
            "output": json.dumps(matched_relations)
        }
        samples.append(sample)

    return samples

# === MAIN SCRIPT ===
input_path = "/home/spshetty/RadAnnotate/finetune/v4/test_set.json"
output_path = "test_new.jsonl"

with open(input_path) as f:
    raw_data = json.load(f)  # ✅ it's a dict, not a list

all_samples = []

for file_id, entry in raw_data.items():
    report = entry.get("text", "")
    if not report:
        print(f"⚠️ No 'text' in {file_id}")
        continue

    labelers = [k for k in entry if k.startswith("labeler")]
    if not labelers:
        print(f"⚠️ No labelers in {file_id}")
        continue

    for labeler in labelers:
        entity_block = entry[labeler].get("entities", {})
        if not entity_block:
            print(f"⚠️ No entities in {labeler} for {file_id}")
            continue

        sentence_samples = generate_samples(report, entity_block)
        if not sentence_samples:
            print(f"🔍 No matched samples in {file_id} from {labeler}")
        all_samples.extend(sentence_samples)

# Write to JSONL
with open(output_path, "w") as f:
    for item in all_samples:
        f.write(json.dumps(item) + "\n")

print(f"✅ Done. Wrote {len(all_samples)} sentence-level prompts to {output_path}")
