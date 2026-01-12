import json, os, numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import config

def run():
    print(f"Generating Reliability Diagrams in {config.CALIB_DIR}...")
    
    for filename in sorted(os.listdir(config.CALIB_DIR)):
        ent = filename.replace("_calibrated.json", "")
        data = json.load(open(os.path.join(config.CALIB_DIR, filename)))
        
        y_true, conf_raw, conf_cal = [], [], []
        
        for it in data:
            gold = {(g["entity_value"], g["entity_type"]) for g in it.get("true_labels", [])}
            for p in it.get("model_output", []):
                y_true.append(1 if (p["entity_value"], p["entity_type"]) in gold else 0)
                conf_raw.append(float(p.get("confidence", 0)))
                conf_cal.append(float(p.get("cal_conf", 0)))

        # Calculate calibration curves
        # n_bins=5 or 10 is standard to see the distribution
        prob_true_raw, prob_pred_raw = calibration_curve(y_true, conf_raw, n_bins=5)
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, conf_cal, n_bins=5)

        # Plotting
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.plot(prob_pred_raw, prob_true_raw, marker='s', label=f'Raw (Uncalibrated)', color='red')
        plt.plot(prob_pred_cal, prob_true_cal, marker='o', label=f'Post-Calibration', color='blue')
        
        plt.title(f"Reliability Diagram: {ent}")
        plt.xlabel("Mean Predicted Confidence")
        plt.ylabel("Fraction of Positives (Actual Accuracy)")
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(config.CALIB_DIR, f"{ent}_calibration_plot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot for {ent}")

if __name__ == "__main__":
    run()