import json
import os
import csv
import numpy as np

# --- CONFIGURATION ---
CSV_PATH = "final_thresholds_report_v2.csv"
COMBINED_TEST_FILE = "/home/spshetty/RadAnnotate/automation_pipeline/test_combined_folds_v2.json"
TARGET_TO_AUDIT = "0.95" 
MAJORITY_RATIO = 0.95 # As per your example

def load_thresholds_for_target(csv_path, target_acc):
    cal_thrs = {}
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    
    # Sort to handle fallbacks
    all_rows.sort(key=lambda x: float(x['Target_Acc']))
    entities = sorted(list(set(row['Entity'] for row in all_rows)))

    for ent in entities:
        last_val = 0.0
        for row in all_rows:
            if row['Entity'] == ent:
                if row['Cal_Thr'] != '--':
                    last_val = float(row['Cal_Thr'])
                if row['Target_Acc'] == target_acc:
                    cal_thrs[ent] = last_val
                    break
    return cal_thrs

def audit_rejections():
    with open(COMBINED_TEST_FILE, 'r') as f:
        data = json.load(f)

    thrs = load_thresholds_for_target(CSV_PATH, TARGET_TO_AUDIT)
    print(f"🔎 Auditing CAL Rejections for Target {TARGET_TO_AUDIT}")
    print(f"Thresholds: {thrs}\n")

    rejection_reasons = {"OBS-U": 0}
    for ent in thrs.keys(): rejection_reasons[ent] = 0

    audited_samples = []

    for item in data:
        preds = item.get("model_output", [])
        gold = item.get("true_labels", [])
        
        # Reason 1: OBS-U
        if any(p["entity_type"] == "OBS-U" for p in preds):
            rejection_reasons["OBS-U"] += 1
            continue

        # Reason 2: Threshold Failures
        failed_entities = []
        for p in preds:
            ent_type = p["entity_type"]
            score = float(p.get("cal_conf", 0))
            required = thrs.get(ent_type, 1.0)
            if score < required:
                failed_entities.append((ent_type, score, required))
        
        # If any failed, calculate the impact
        if failed_entities:
            # Increment count for the specific entities that failed
            unique_fails = set([f[0] for f in failed_entities])
            for f_ent in unique_fails:
                rejection_reasons[f_ent] += 1
            
            # Save a sample for analysis
            audited_samples.append({
                "id": item.get("index"),
                "fails": failed_entities,
                "total_acc": 1.0 # placeholder for Jaccard if needed
            })

    # --- PRINT SUMMARY ---
    print(f"{'ENTITY TYPE':<15} | {'REJECTION COUNT':<18} | {'% of Total Rejections'}")
    print("-" * 60)
    total_rej = sum(rejection_reasons.values())
    for ent, count in rejection_reasons.items():
        pct = (count / total_rej * 100) if total_rej > 0 else 0
        print(f"{ent:<15} | {count:<18} | {pct:.1f}%")

    # --- PRINT SAMPLES ---
    print(f"\n{'='*20} EXAMPLES OF REJECTED REPORTS {'='*20}")
    for sample in audited_samples[:5]:
        print(f"\nID: {sample['id']}")
        for ent, score, req in sample['fails']:
            print(f"  {ent}: Score {score:.3f} is BELOW threshold {req:.2f}")

if __name__ == "__main__":
    audit_rejections()