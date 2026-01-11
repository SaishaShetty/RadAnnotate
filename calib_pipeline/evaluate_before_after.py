import os, json, pickle
import numpy as np

SPLIT_DIR = "outputs/splits"
CAL_DIR = "outputs/calibrators"
OUT_CAL_TEST_DIR = "outputs/calibrated"
ENTITIES = ["OBS-U", "OBS-DA"]
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

def collect_conf_and_labels(items, key="confidence"):
    conf, y = [], []
    for it in items:
        gold = {(g["entity_value"], g["entity_type"]) for g in it["true_labels"]}
        for p in it["model_output"]:
            c = p.get(key)
            if c is None:
                continue
            pred = (p["entity_value"], p["entity_type"])
            conf.append(float(c))
            y.append(1 if pred in gold else 0)
    return np.array(conf), np.array(y)

def brier(p, y):
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")

def fp_conf_list(items, key="confidence"):
    fp = []
    for it in items:
        gold = {(g["entity_value"], g["entity_type"]) for g in it["true_labels"]}
        for p in it["model_output"]:
            c = p.get(key)
            if c is None:
                continue
            pred = (p["entity_value"], p["entity_type"])
            if pred not in gold:
                fp.append(float(c))
    return fp

def main():
    os.makedirs(OUT_CAL_TEST_DIR, exist_ok=True)

    for ent in ENTITIES:
        test_path = f"{SPLIT_DIR}/{ent}/test.json"
        iso_path = f"{CAL_DIR}/{ent}_isotonic.pkl"

        test = json.load(open(test_path))
        iso = pickle.load(open(iso_path, "rb"))

        # --- compute BEFORE metrics on *the same test* ---
        test_conf, test_y = collect_conf_and_labels(test, key="confidence")
        b_before = brier(test_conf, test_y)

        # --- add cal_conf to the same test items ---
        for it in test:
            for p in it["model_output"]:
                c = p.get("confidence")
                p["cal_conf"] = None if c is None else float(iso.predict([float(c)])[0])

        # --- compute AFTER metrics on the same test items ---
        test_cal, test_y2 = collect_conf_and_labels(test, key="cal_conf")
        b_after = brier(test_cal, test_y2)

        # --- FP reduction analysis (same test, same FPs, different conf key) ---
        fp_raw = fp_conf_list(test, key="confidence")
        fp_cal = fp_conf_list(test, key="cal_conf")

        print(f"\n==================== {ent} ====================")
        print(f"TEST preds: {len(test_conf)} | Brier BEFORE: {b_before:.4f} | AFTER: {b_after:.4f} | Δ: {(b_before-b_after):.4f}")
        print(f"TEST FP count: {len(fp_raw)}")

        if len(fp_raw):
            for thr in THRESHOLDS:
                raw_cnt = sum(c >= thr for c in fp_raw)
                cal_cnt = sum(c >= thr for c in fp_cal)
                print(f"FP conf ≥ {thr}: BEFORE {raw_cnt}/{len(fp_raw)} ({100*raw_cnt/len(fp_raw):.2f}%) | AFTER {cal_cnt}/{len(fp_raw)} ({100*cal_cnt/len(fp_raw):.2f}%)")

        # --- save calibrated test (original test.json remains untouched on disk) ---
        out_ent_dir = os.path.join(OUT_CAL_TEST_DIR, ent)
        os.makedirs(out_ent_dir, exist_ok=True)
        out_path = os.path.join(out_ent_dir, "test_calibrated.json")
        json.dump(test, open(out_path, "w"), indent=2)
        print(f"Saved calibrated TEST → {out_path}")

if __name__ == "__main__":
    main()
