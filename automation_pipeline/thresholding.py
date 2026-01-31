import json, os, csv
import numpy as np

# --- CONFIGURATION ---
SPLIT_ROOT = "/home/spshetty/new_calib/outputs/splits"
ENTITIES = ["ANAT-DP", "OBS-DP", "OBS-DA", "OBS-U"]
THRESHOLDS = [round(t, 2) for t in np.arange(0, 1.01, 0.01)]

def validate_alignment(raw_data, cal_data, entity_name):
    """
    Checks if raw and calibrated datasets are perfectly aligned.
    """
    if len(raw_data) != len(cal_data):
        raise ValueError(
            f"CRITICAL ERROR for {entity_name}: Row count mismatch! "
            f"Raw: {len(raw_data)}, Calibrated: {len(cal_data)}"
        )
    
    # Check every report ID to ensure perfect synchronization
    for i, (r_item, c_item) in enumerate(zip(raw_data, cal_data)):
        # Try to find an ID in common keys
        r_id = r_item.get("id") or r_item.get("report_id") or "No ID"
        c_id = c_item.get("id") or c_item.get("report_id") or "No ID"
        
        if r_id != c_id:
            raise ValueError(
                f"ID MISMATCH at index {i} for {entity_name}!\n"
                f"  Raw ID: {r_id}\n"
                f"  Cal ID: {c_id}"
            )
    
    # Print sample for peace of mind
    first_id = raw_data[0].get("id") or raw_data[0].get("report_id") or "N/A"
    last_id = raw_data[-1].get("id") or raw_data[-1].get("report_id") or "N/A"
    print(f"✅ Alignment Verified for {entity_name} ({len(raw_data)} reports).")
    print(f"   Sample IDs match: Start={first_id}, End={last_id}")

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
                print(f"⚠️  Skipping {ent}: Files not found.")
                continue

            # Load Data
            with open(raw_path, 'r') as f_raw, open(cal_path, 'r') as f_cal:
                raw_data = json.load(f_raw)
                cal_data = json.load(f_cal)

            # 1. VALIDATION STEP
            try:
                validate_alignment(raw_data, cal_data, ent)
            except ValueError as e:
                print(f"❌ {e}")
                continue 

            print(f"\n" + "="*70)
            print(f"ENTITY: {ent}")
            print(f"{'THR':<6} | {'RAW ACC':<8} | {'RAW COV':<8} | {'CAL ACC':<8} | {'CAL COV':<8}")
            print("-" * 70)

            # 2. Compute Metrics
            raw_metrics = get_metrics_at_thresholds(raw_data, "confidence")
            cal_metrics = get_metrics_at_thresholds(cal_data, "cal_conf")

            for thr in THRESHOLDS:
                r_acc, r_cov = raw_metrics[thr]
                c_acc, c_cov = cal_metrics[thr]
                
                writer.writerow([ent, thr, r_acc, r_cov, c_acc, c_cov])
                
                # Logic check: if coverage is very low, accuracy might be unstable
                print(f"{thr:>4.2f}   | {r_acc:>7.1%} | {r_cov:>7.1%} | {c_acc:>7.1%} | {c_cov:>7.1%}")

    print(f"\n🏁 Finished. Results saved to {csv_filename}")

if __name__ == "__main__":
    run()