import json, os, numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import config

def collect(items):
    conf, y = [], []
    for it in items:
        gold = {(g["entity_value"], g["entity_type"]) for g in it.get("true_labels", [])}
        for p in it.get("model_output", []):
            # Using .get() ensures we don't crash if a key is missing
            c = p.get("confidence")
            if c is not None:
                correct = 1 if (p["entity_value"], p["entity_type"]) in gold else 0
                conf.append(float(c))
                y.append(correct)
    return np.array(conf), np.array(y)

def run():
    print(f"\nCalibrating entities in {config.SPLIT_DIR}...")
    
    for ent in os.listdir(config.SPLIT_DIR):
        ent_dir = os.path.join(config.SPLIT_DIR, ent)
        if not os.path.isdir(ent_dir): continue

        val = json.load(open(f"{ent_dir}/val.json"))
        test = json.load(open(f"{ent_dir}/test.json"))
        
        v_conf, v_y = collect(val)
        
        if len(v_conf) == 0:
            print(f"Skipping {ent}: No prediction data found.")
            continue

        # Smart Switch: Platt for small samples, Isotonic for large
        method = "Platt" if len(v_conf) < 200 else "Isotonic"
        
        if method == "Platt":
            model = LogisticRegression(C=1e10).fit(v_conf.reshape(-1, 1), v_y)
            pred_func = lambda x: model.predict_proba(x.reshape(-1, 1))[:, 1]
        else:
            # clip ensures we handle test scores outside the val set range
            model = IsotonicRegression(out_of_bounds="clip").fit(v_conf, v_y)
            pred_func = lambda x: model.predict(x)

        # Apply to Test Data
        for it in test:
            for p in it["model_output"]:
                raw_c = p.get("confidence")
                # Handle potential nulls even in test set
                score = float(raw_c) if raw_c is not None else 0.0
                p["cal_conf"] = float(pred_func(np.array([score]))[0])

        out_path = os.path.join(config.CALIB_DIR, f"{ent}_calibrated.json")
        json.dump(test, open(out_path, "w"), indent=2)
        print(f"{ent:<10} | Samples: {len(v_conf):<5} | Method: {method}")

if __name__ == "__main__":
    run()