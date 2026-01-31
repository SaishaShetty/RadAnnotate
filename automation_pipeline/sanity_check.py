import json
import numpy as np

# --- CONFIGURATION ---
COMBINED_TEST_FILE = "/home/spshetty/RadAnnotate/automation_pipeline/test_combined_folds_v2.json"
THRS = {'ANAT-DP': 0.97, 'OBS-DA': 0.91, 'OBS-DP': 0.92, 'OBS-U': 0.0}
MAJORITY_RATIO = 0.95

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

def get_detailed_comparison(preds, gold_labels):
    pred_set = {(p["entity_type"], p["entity_value"].strip().lower()) for p in preds}
    gold_set = {(g["entity_type"], g["entity_value"].strip().lower()) for g in gold_labels}
    tp, fp, fn = pred_set & gold_set, pred_set - gold_set, gold_set - pred_set
    denom = len(tp) + len(fp) + len(fn)
    acc = len(tp) / denom if denom > 0 else (1.0 if not gold_set and not pred_set else 0.0)
    return acc, sorted(list(gold_set)), sorted(list(fp)), sorted(list(fn))

def run_comprehensive_audit():
    with open(COMBINED_TEST_FILE, 'r') as f:
        data = json.load(f)

    accepted_list, rejected_list = [], []

    for item in data:
        raw_preds = item.get("model_output", [])
        gold = item.get("true_labels", [])
        resolved = resolve_overlaps(raw_preds, "cal_conf")
        has_obsu = any(p["entity_type"] == "OBS-U" for p in raw_preds)
        
        above_count = sum(1 for p in resolved if float(p.get("cal_conf", 0)) >= THRS.get(p["entity_type"], 1.0))
        ratio = above_count / len(resolved) if resolved else 1.0
        is_automated = (ratio >= MAJORITY_RATIO) and not has_obsu

        acc, gt, fp, fn = get_detailed_comparison(resolved, gold)
        res = {"id": item.get("index"), "acc": acc, "gt": gt, "fp": fp, "fn": fn}

        if is_automated: accepted_list.append(res)
        else: rejected_list.append(res)

    # --- 1. SUMMARY ---
    print("\n" + "="*80)
    print(f"{'FINAL PERFORMANCE SUMMARY':^80}")
    print("="*80)
    print(f"{'Category':<20} | {'Count':<10} | {'Avg Acc'}")
    print("-" * 45)
    print(f"{'Accepted (Auto)':<20} | {len(accepted_list):<10} | {np.mean([x['acc'] for x in accepted_list]):.4f}")
    print(f"{'Rejected (Review)':<20} | {len(rejected_list):<10} | {np.mean([x['acc'] for x in rejected_list]):.4f}")
    print("="*80)

    # --- 2. ACCEPTED SECTION ---
    print(f"\n\n>>> SECTION 1: ACCEPTED FOR AUTOMATION (Total: {len(accepted_list)})")
    print("="*80)
    for x in accepted_list[:15]: # Printing first 15 for brevity
        print(f"AUTO ID: {x['id']} | Accuracy: {x['acc']:.2%}")
        print(f"  GT: {x['gt']}")
        if x['fp']: print(f"  HALLUCINATED: {x['fp']}")
        if x['fn']: print(f"  MISSED: {x['fn']}")
        print("-" * 80)

    # --- 3. REJECTED SECTION ---
    print(f"\n\n>>> SECTION 2: REJECTED FOR HUMAN REVIEW (Total: {len(rejected_list)})")
    print("="*80)
    for x in rejected_list[:15]: # Printing first 15 for brevity
        print(f"REVIEW ID: {x['id']} | Accuracy: {x['acc']:.2%}")
        print(f"  GT: {x['gt']}")
        if x['fp']: print(f"  HALLUCINATED: {x['fp']}")
        if x['fn']: print(f"  MISSED: {x['fn']}")
        print("-" * 80)

if __name__ == "__main__":
    run_comprehensive_audit()