import json
import random

# === Step 1: Load full dataset ===
input_file = "/home/spshetty/RadAnnotate/data_generation/data/synthetic_data.json"     # <-- change path
output_file = "syn_data(2k).json"  # <-- save output

with open(input_file, "r") as f:
    data = json.load(f)

# === Step 2: Randomly select 1000 samples ===
random_1k = random.sample(data, k=2000)

# === Step 3: Save new subset ===
with open(output_file, "w") as f:
    json.dump(random_1k, f, indent=4)

print(f"✅ Saved 1000 random samples to {output_file}")
