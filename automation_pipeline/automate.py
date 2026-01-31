import json
import os
import csv
import numpy as np

# --- CONFIGURATION ---
CSV_PATH = "final_thresholds_report_v2.csv"
COMBINED_TEST_FILE = "/home/spshetty/RadAnnotate/automation_pipeline/test_combined_folds_v2.json"

# The rules to test
MAJORITY_RATIOS = [0.90, 0.95]

def get_available_targets(csv_path):
    targets = set()
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.add(row['Target_Acc'])
    return sorted(list(targets))

def load_thresholds_with_fallback(csv_path, target_acc):
    raw_thrs, cal_thrs = {}, {}
    with open(csv_path, mode='r') as f:
        all_rows = list(csv.DictReader(f))
    
    entities = sorted(list(set(row['Entity'] for row in all_rows)))
    all_rows.sort(key=lambda x: float(x['Target_Acc']))

    for ent in entities:
        last_known_raw, last_known_cal = 0.0, 0.0
        for row in all_rows:
            if row['Entity'] == ent:
                if row['Raw_Thr'] != '--': last_known_raw = float(row['Raw_Thr'])
                if row['Cal_Thr'] != '--': last_known_cal = float(row['Cal_Thr'])
                if row['Target_Acc'] == target_acc:
                    raw_thrs[ent] = last_known_raw
                    cal_thrs[ent] = last_known_cal
                    break
    return raw_thrs, cal_thrs

def resolve_overlaps(preds, conf_key):
    value_groups = {}
    for p in preds:
        val = p["entity_value"].strip().lower()
        value_groups.setdefault(val, []).append(p)
    resolved = []
    for val, group in value_groups.items():
        best = max(group, key=lambda x: x.get(conf_key, 0))
        resolved.append(best)
    return resolved

def compute_jaccard(preds, gold_labels):
    pred_set = {(p["entity_value"], p["entity_type"]) for p in preds}
    gold_set = {(g["entity_value"], g["entity_type"]) for g in gold_labels}
    TP = len(pred_set & gold_set)
    FP = len(pred_set - gold_set)
    FN = len(gold_set - pred_set)
    denom = TP + FP + FN
    return TP / denom if denom > 0 else (1.0 if not gold_set and not pred_set else 0.0)

def simulate(data, thr_map, conf_key, majority_ratio):
    auto_accs, review_accs = [], []
    for item in data:
        raw_preds = item.get("model_output", [])
        gold = item.get("true_labels", [])
        
        if any(p["entity_type"] == "OBS-U" for p in raw_preds):
            acc = compute_jaccard(resolve_overlaps(raw_preds, conf_key), gold)
            review_accs.append(acc)
            continue

        resolved = resolve_overlaps(raw_preds, conf_key)
        if not resolved:
            acc = 1.0 if not gold else 0.0
            review_accs.append(acc)
            continue

        above_thr_count = 0
        for p in resolved:
            thr = thr_map.get(p["entity_type"], 1.0)
            if float(p.get(conf_key, 0)) >= thr:
                above_thr_count += 1
        
        ratio = above_thr_count / len(resolved)
        acc = compute_jaccard(resolved, gold)
        
        if ratio >= majority_ratio:
            auto_accs.append(acc)
        else:
            review_accs.append(acc)
                
    return auto_accs, review_accs

def run_all_simulations():
    targets = get_available_targets(CSV_PATH)
    if not os.path.exists(COMBINED_TEST_FILE):
        print(f"❌ Error: Combined file {COMBINED_TEST_FILE} not found.")
        return

    with open(COMBINED_TEST_FILE, 'r') as f: 
        test_data = json.load(f)

    for ratio in MAJORITY_RATIOS:
        print(f"\n{'#'*130}")
        print(f"{'SIMULATION: ' + str(int(ratio*100)) + '% MAJORITY RULE':^130}")
        print(f"{'#'*130}")

        for t in targets:
            r_thrs, c_thrs = load_thresholds_with_fallback(CSV_PATH, t)
            
            r_auto, r_rev = simulate(test_data, r_thrs, "confidence", ratio)
            c_auto, c_rev = simulate(test_data, c_thrs, "cal_conf", ratio)

            total = len(test_data)

            # String formats for the thresholds to keep the table clean
            r_thr_str = ", ".join([f"{k}:{v:.2f}" for k, v in r_thrs.items()])
            c_thr_str = ", ".join([f"{k}:{v:.2f}" for k, v in c_thrs.items()])

            print(f"\n[TARGET: {t}]")
            print(f"{'Model':<6} | {'Auto %':<8} | {'Auto-Acc':<10} | {'Rev-Acc':<10} | {'Thresholds (ANAT, DA, DP, U)'}")
            print("-" * 130)
            
            print(f"{'RAW':<6} | {len(r_auto)/total:>7.1%} | {np.mean(r_auto) if r_auto else 0:>10.4f} | {np.mean(r_rev) if r_rev else 0:>10.4f} | {r_thr_str}")
            print(f"{'CAL':<6} | {len(c_auto)/total:>7.1%} | {np.mean(c_auto) if c_auto else 0:>10.4f} | {np.mean(c_rev) if c_rev else 0:>10.4f} | {c_thr_str}")
            print("-" * 130)

if __name__ == "__main__":
    run_all_simulations()