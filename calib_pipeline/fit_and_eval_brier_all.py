import os, json, pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression

SPLIT_DIR = "outputs/splits"
CAL_DIR = "outputs/calibrators"
CALIBRATED_DIR = "outputs/calibrated"

ENTITIES = ["ANAT-DP", "OBS-DP", "OBS-DA", "OBS-U"]

# Only these will be saved as test_calibrated.json
SAVE_CALIBRATED_FOR = {"OBS-U", "OBS-DA"}

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

def main():
    os.makedirs(CAL_DIR, exist_ok=True)
    os.makedirs(CALIBRATED_DIR, exist_ok=True)

    print("\n===== FIT ON VAL, EVAL BRIER ON TEST (ALL 4) =====")

    for ent in ENTITIES:
        val_path = f"{SPLIT_DIR}/{ent}/val.json"
        test_path = f"{SPLIT_DIR}/{ent}/test.json"

        val = json.load(open(val_path))
        test = json.load(open(test_path))

        # ---- Fit isotonic on VAL ----
        val_conf, val_y = collect_conf_and_labels(val, key="confidence")
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_conf, val_y)

        # Save calibrator for reproducibility (all 4)
        iso_path = f"{CAL_DIR}/{ent}_isotonic.pkl"
        pickle.dump(iso, open(iso_path, "wb"))

        # ---- Evaluate on SAME TEST: before vs after ----
        test_conf, test_y = collect_conf_and_labels(test, key="confidence")
        b_before = brier(test_conf, test_y)

        # Add cal_conf IN MEMORY to the same test items (no resplit, no mismatch)
        for it in test:
            for p in it["model_output"]:
                c = p.get("confidence")
                p["cal_conf"] = None if c is None else float(iso.predict([float(c)])[0])

        test_cal, test_y2 = collect_conf_and_labels(test, key="cal_conf")
        b_after = brier(test_cal, test_y2)

        print(f"\n{ent}")
        print(f"  TEST preds: {len(test_conf)}")
        print(f"  Brier BEFORE: {b_before:.4f}")
        print(f"  Brier AFTER:  {b_after:.4f}")
        print(f"  Δ:            {(b_before - b_after):.4f}")

        # ---- Save calibrated test ONLY for OBS-U and OBS-DA ----
        if ent in SAVE_CALIBRATED_FOR:
            out_ent_dir = os.path.join(CALIBRATED_DIR, ent)
            os.makedirs(out_ent_dir, exist_ok=True)
            out_path = os.path.join(out_ent_dir, "test_calibrated.json")
            json.dump(test, open(out_path, "w"), indent=2)
            print(f"  Saved calibrated TEST → {out_path}")
        else:
            print("  (Not saving calibrated JSON for this entity — calibration not used.)")

if __name__ == "__main__":
    main()
