import json, os, csv
import numpy as np

# --- CONFIGURATION ---
SPLIT_ROOT = "/home/spshetty/new_calib/outputs/splits"
ENTITIES = ["ANAT-DP", "OBS-DP", "OBS-DA", "OBS-U"]

# Using 0.00 to 1.00 in 0.01 increments
# We round to 2 decimals to prevent floating point errors (e.g., 0.7000000001)
THRESHOLDS = [round(t, 2) for t in np.arange(0, 1.01, 0.01)]

def get_metrics_at_thresholds(data, conf_key):
    """Computes accuracy and coverage for all thresholds."""
    results = {}
    total_reports = len(data)
    
    for thr in THRESHOLDS:
        accepted_reports = 0
        total_preds_accepted = 0
        correct_preds_accepted = 0

        for item in data:
            confs = [
                p.get(conf_key) 
                for p in item.get("model_output", []) 
                if p.get(conf_key) is not None
            ]

            # REJECTION: If any prediction in report < threshold, skip report
            if not confs or any(round(c, 4) < thr for c in confs):
                continue

            accepted_reports += 1
            gold = {(g["entity_value"], g["entity_type"]) for g in item.get("true_labels", [])}

            for p in item.get("model_output", []):
                total_preds_accepted += 1
                if (p["entity_value"], p["entity_type"]) in gold:
                    correct_preds_accepted += 1

        acc = (correct_preds_accepted / total_preds_accepted) if total_preds_accepted > 0 else 1.0
        cov = accepted_reports / total_reports
        results[thr] = (acc, cov)
    
    return results

def run():
    csv_filename = "full_sensitivity_results.csv"
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Entity", "Threshold", "Raw_Acc", "Raw_Cov", "Cal_Acc", "Cal_Cov"])

        for ent in ENTITIES:
            ent_dir = os.path.join(SPLIT_ROOT, ent)
            
            raw_path = os.path.join(ent_dir, "val.json")
            cal_path = os.path.join(ent_dir, "val_calibrated.json")

            if not os.path.exists(raw_path) or not os.path.exists(cal_path):
                continue

            print(f"\n" + "="*70)
            print(f"ENTITY: {ent}")
            print(f"{'THR':<6} | {'RAW ACC':<8} | {'RAW COV':<8} | {'CAL ACC':<8} | {'CAL COV':<8}")
            print("-" * 70)

            raw_data = json.load(open(raw_path))
            cal_data = json.load(open(cal_path))

            raw_metrics = get_metrics_at_thresholds(raw_data, "confidence")
            cal_metrics = get_metrics_at_thresholds(cal_data, "cal_conf")

            for thr in THRESHOLDS:
                r_acc, r_cov = raw_metrics[thr]
                c_acc, c_cov = cal_metrics[thr]
                
                # Write to CSV
                writer.writerow([ent, thr, r_acc, r_cov, c_acc, c_cov])
                
                # PRINT EVERY STEP
                print(f"{thr:>4.2f}   | {r_acc:>7.1%} | {r_cov:>7.1%} | {c_acc:>7.1%} | {c_cov:>7.1%}")

    print(f"\nFull table printed and saved to {csv_filename}")

if __name__ == "__main__":
    run()