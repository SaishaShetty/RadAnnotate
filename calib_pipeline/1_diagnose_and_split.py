import json, os
import numpy as np
from sklearn.model_selection import train_test_split
import config

def get_fp_stats(items, ent_name):
    total_preds_at_thresh = {t: 0 for t in config.THRESHOLDS}
    fp_counts_at_thresh = {t: 0 for t in config.THRESHOLDS}
    total_preds_overall = 0
    
    for item in items:
        gold = {(g["entity_value"], g["entity_type"]) for g in item.get("true_labels", [])}
        for p in item.get("model_output", []):
            total_preds_overall += 1
            c_val = float(p.get("confidence", 0))
            is_fp = (p["entity_value"], p["entity_type"]) not in gold
            
            for t in config.THRESHOLDS:
                if c_val >= t:
                    total_preds_at_thresh[t] += 1
                    if is_fp:
                        fp_counts_at_thresh[t] += 1
    
    print(f"\nUNIQUE DIAGNOSTIC (FILTERED): {ent_name}")
    print(f"   Unique Reports: {len(items)} | Total Predictions: {total_preds_overall}")
    print(f"   {'Threshold':<12} | {'Total Preds':<12} | {'FP Count':<10} | {'Error Rate (%)'}")
    print(f"   {'-'*55}")
    
    for t in config.THRESHOLDS:
        total = total_preds_at_thresh[t]
        fps = fp_counts_at_thresh[t]
        rate = (fps / total * 100) if total > 0 else 0
        print(f"   >{t:<11} | {total:<12} | {fps:<10} | {rate:.1f}%")

def run():
    for ent, path in config.ENTITY_FILES.items():
        if not os.path.exists(path):
            print(f"NOT FOUND: {path}")
            continue
        
        with open(path, 'r') as f:
            raw = json.load(f)
        
        # --- DEDUPLICATION & FILTERING ---
        seen_reports = set()
        clean_uniq_data = []
        dropped_count = 0

        for it in raw:
            report_text = (it.get("report") or "").strip()
            
            # 1. Check for duplicates
            if report_text in seen_reports:
                continue
                
            # 2. Check for missing confidence (The 2 problematic reports)
            has_null = any(p.get("confidence") is None for p in it.get("model_output", []))
            
            if has_null:
                print(f"⚠️  DROPPING report index {it.get('index')} due to missing confidence.")
                dropped_count += 1
                continue
            
            seen_reports.add(report_text)
            clean_uniq_data.append(it)

        print(f"{ent}: Kept {len(clean_uniq_data)} reports (Dropped {dropped_count} nulls)")

        # Run stats on the clean data
        get_fp_stats(clean_uniq_data, ent)
        
        # Split
        val, test = train_test_split(clean_uniq_data, test_size=config.TEST_SIZE, random_state=config.SEED)
        
        ent_dir = os.path.join(config.SPLIT_DIR, ent)
        os.makedirs(ent_dir, exist_ok=True)
        json.dump(val, open(os.path.join(ent_dir, "val.json"), "w"), indent=2)
        json.dump(test, open(os.path.join(ent_dir, "test.json"), "w"), indent=2)
        print(f"Saved clean splits to {ent_dir}")

if __name__ == "__main__":
    run()