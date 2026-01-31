import json
import os
import csv
import numpy as np

# --- CONFIGURATION ---
# Pointing to your new Folds output directory
INPUT_DIR = "/home/spshetty/new_calib/folds_outputs_v2/val_calibrated"
ENTITIES = ["ANAT-DP", "OBS-DP", "OBS-DA", "OBS-U"]
TARGET_ACCURACIES = [0.80, 0.85, 0.90, 0.95]
OUTPUT_CSV = "final_thresholds_report_v2.csv"

# --- MAJORITY RULE ---
# 1.00 means ALL entities in a report must be above the threshold to automate
REQUIRED_RATIO = 1.00

def report_accuracy(true_labels, pred_labels):
    """Calculates accuracy for a single report."""
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    TP = len(true_set & pred_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)
    denom = TP + FP + FN
    return TP / denom if denom > 0 else None

def find_thresholds_for_targets(data, conf_key, majority_ratio):
    """Sweeps thresholds to find the best coverage for target accuracies."""
    results = {f"{t:.2f}": None for t in TARGET_ACCURACIES}
    
    # Sweep thresholds from 0.0 to 1.0
    for thr in np.arange(0.00, 1.01, 0.01):
        thr = round(float(thr), 2)
        report_scores = []
        accepted_count = 0

        for item in data:
            preds = item.get("model_output", [])
            if not preds: continue

            # Count how many entities pass the current threshold
            # Using round(p.get(conf_key), 4) for floating point safety
            above_thr_count = sum(1 for p in preds if (p.get(conf_key) is not None) and (round(p.get(conf_key), 4) >= thr))
            actual_ratio = above_thr_count / len(preds)

            # Check if this report meets the Majority Rule (e.g., 100%)
            if actual_ratio < majority_ratio:
                continue

            accepted_count += 1
            gold_vals = [g["entity_value"] for g in item.get("true_labels", [])]
            pred_vals = [p["entity_value"] for p in preds]

            acc = report_accuracy(gold_vals, pred_vals)
            if acc is not None:
                report_scores.append(acc)

        avg_acc = np.mean(report_scores) if report_scores else 0.0
        coverage = accepted_count / len(data) if len(data) > 0 else 0.0

        # Record the FIRST (lowest) threshold that hits our target accuracy
        for t_val in TARGET_ACCURACIES:
            t_str = f"{t_val:.2f}"
            if results[t_str] is None and avg_acc >= t_val:
                results[t_str] = {"thr": thr, "acc": avg_acc, "cov": coverage}
                
    return results

def run():
    all_rows = []
    print(f"\n" + "="*95)
    print(f"🚀 FINAL THRESHOLD OPTIMIZATION (Majority Ratio: {int(REQUIRED_RATIO*100)}%)")
    print(f"="*95)
    
    # Get all calibrated files
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_val.json")]
    
    for filename in sorted(files):
        ent = filename.split("_")[0]
        if ent not in ENTITIES: continue
        
        file_path = os.path.join(INPUT_DIR, filename)
        data = json.load(open(file_path))

        # We run the SAME data through the optimizer twice:
        # Once using the 'confidence' key (Raw)
        # Once using the 'cal_conf' key (Calibrated)
        raw_results = find_thresholds_for_targets(data, "confidence", REQUIRED_RATIO)
        cal_results = find_thresholds_for_targets(data, "cal_conf", REQUIRED_RATIO)

        print(f"\nENTITY: {ent}")
        print(f"{'Target Acc':<12} | {'RAW Thr':<8} | {'RAW Cov':<10} | {'CAL Thr':<8} | {'CAL Cov':<10} | {'Gain'}")
        print("-" * 95)

        for t_val in TARGET_ACCURACIES:
            target = f"{t_val:.2f}"
            r = raw_results[target]
            c = cal_results[target]

            r_thr = f"{r['thr']:.2f}" if r else "--"
            r_cov = r['cov'] if r else 0.0
            c_thr = f"{c['thr']:.2f}" if c else "--"
            c_cov = c['cov'] if c else 0.0
            
            gain = (c_cov - r_cov)
            
            r_cov_str = f"{r_cov:>8.1%}" if r else f"{'--':>8}"
            c_cov_str = f"{c_cov:>8.1%}" if c else f"{'--':>8}"
            gain_str = f"{gain:>+7.1%}" 

            print(f"{target:<12} | {r_thr:<8} | {r_cov_str}   | {c_thr:<8} | {c_cov_str}   | {gain_str}")
            all_rows.append([ent, target, r_thr, r_cov, c_thr, c_cov, gain])

    # Save to CSV for your final report
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Entity", "Target_Acc", "Raw_Thr", "Raw_Cov", "Cal_Thr", "Cal_Cov", "Gain_Pct"])
        writer.writerows(all_rows)

    print(f"\nOptimization complete. Final report saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run()