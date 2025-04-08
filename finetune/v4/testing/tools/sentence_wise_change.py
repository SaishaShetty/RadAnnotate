import json
import re

def simple_sentence_split(text):
    # Naive sentence splitter (based on period)
    sentences = re.split(r'\.\s+|\.$', text)
    return [s.strip() + '.' for s in sentences if s.strip()]

def convert_to_simple_labels(data):
    output = []

    for doc_id, doc_data in data.items():
        text = doc_data.get("text", "")
        entities = doc_data.get("labeler_1", {}).get("entities", {})

        if not text or not entities:
            continue

        tokens = text.strip().split()
        sentences = simple_sentence_split(text)

        token_ptr = 0  # tracks position of each sentence

        for sentence in sentences:
            sentence_tokens = sentence.strip().split()
            start_ix = token_ptr
            end_ix = start_ix + len(sentence_tokens) - 1

            sentence_labels = {}

            for ent in entities.values():
                ent_token = ent.get("tokens", "").strip()
                ent_label = ent.get("label", "").strip()
                ent_start = ent.get("start_ix", -1)

                if ent_token and start_ix <= ent_start <= end_ix:
                    sentence_labels[ent_token] = ent_label

            if sentence_labels:
                output.append({
                    "sentence": sentence.strip(),
                    "labels": sentence_labels
                })

            token_ptr += len(sentence_tokens)

    return output

# === Load JSON and convert ===
input_path = "/home/spshetty/RadAnnotate/finetune/v4/test_set.json"  # Change to your real file
output_path = "sentence_wise_output.json"

with open(input_path, "r") as infile:
    data = json.load(infile)

converted = convert_to_simple_labels(data)

with open(output_path, "w") as out:
    json.dump(converted, out, indent=2)
