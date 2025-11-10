import json
from sklearn.metrics import precision_score, recall_score, f1_score

# === Load the result file ===
with open("results.json", "r") as f:
    data = json.load(f)

y_true_all = []
y_pred_all = []

strict_correct = 0
total_reports = 0

# === Process each report ===
for entry in data:
    true_entities = entry.get("true_labels", [])
    pred_entities = entry.get("predicted_labels", [])

    # Handle case when model_output is a raw string (due to JSON decode error)
    if not isinstance(pred_entities, list):
        pred_entities = []

    # Convert to set of (type, value)
    true_set = set((e["entity_type"], e["entity_value"].lower()) for e in true_entities)
    pred_set = set((e["entity_type"], e["entity_value"].lower()) for e in pred_entities)

    # --- Strict Accuracy check (report-level exact match) ---
    if true_set == pred_set:
        strict_correct += 1
    total_reports += 1

    # --- Entity-level metrics ---
    all_entities = true_set.union(pred_set)
    for entity in all_entities:
        y_true_all.append(1 if entity in true_set else 0)
        y_pred_all.append(1 if entity in pred_set else 0)

# === Compute Scores ===
precision = precision_score(y_true_all, y_pred_all)
recall = recall_score(y_true_all, y_pred_all)
f1 = f1_score(y_true_all, y_pred_all)
strict_accuracy = strict_correct / total_reports if total_reports > 0 else 0

print("Evaluation Results:")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"Strict Accuracy: {strict_accuracy:.4f}")
