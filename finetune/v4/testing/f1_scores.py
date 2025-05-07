import json
import csv

def parse_entities(raw_ents):
    entities = set()
    for e in raw_ents:
        if isinstance(e, dict):
            label = e.get("entity_type", "").strip()
            val = e.get("entity_value", "").strip()
            if label and val:
                entities.add((label, val))
        elif isinstance(e, str):
            parts = e.strip().split(" ", 1)
            if len(parts) == 2:
                entities.add((parts[0].strip(), parts[1].strip()))
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            entities.add((str(e[0]).strip(), str(e[1]).strip()))
    return entities

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        return [
            {
                "report": item["report"],
                "entities": parse_entities(item.get("entities", []))
            }
            for item in data
        ]

def compute_precision_recall_f1(true_set, pred_set):
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1

# Load files
true_data = load_json("/home/spshetty/RadAnnotate/finetune/v4/testing/gold_data/test_set.json")
pred_data = load_json("/home/spshetty/RadAnnotate/finetune/v4/testing/FINAL/syn+real(2k)_res.json")

assert len(true_data) == len(pred_data), "Mismatch in number of reports!"

# Output file
csv_file = "/home/spshetty/RadAnnotate/finetune/v4/testing/results/syn+real(4k).csv"
total_f1 = 0
included_count = 0

with open(csv_file, "w", newline='', encoding="utf-8") as csvfile:
    fieldnames = ['report', 'ans', 'pred', 'precision', 'recall', 'f1_score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for t, p in zip(true_data, pred_data):
        report_text = t["report"]
        true_ents = t["entities"]
        pred_ents = p["entities"]

        if not pred_ents:
            # Skip if prediction is empty
            continue

        precision, recall, f1 = compute_precision_recall_f1(true_ents, pred_ents)
        total_f1 += f1
        included_count += 1

        ans_str = ", ".join([f"{x[0]}: {x[1]}" for x in sorted(true_ents)])
        pred_str = ", ".join([f"{x[0]}: {x[1]}" for x in sorted(pred_ents)])

        writer.writerow({
            "report": report_text,
            "ans": ans_str,
            "pred": pred_str,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        })

# Final stats
avg_f1 = total_f1 / included_count if included_count else 0
print(f"Saved to {csv_file}")
print(f"Average F1 Score (excluding empty predictions): {avg_f1:.4f} on {included_count} samples")
