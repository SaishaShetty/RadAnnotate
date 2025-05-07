import torch
import json
import codecs
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# === Load model and tokenizer ===
model_path = "/home/spshetty/RadAnnotate/finetune/v4/mistral-ner-v4/checkpoint-7809"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Load input ===
with open("/home/spshetty/RadAnnotate/finetune/v4/testing/gold_data/test_set.json", "r") as f:
    data = json.load(f)

results = []

for item in tqdm(data, desc="Processing reports"):
    report = item.get("Report", "")

    prompt = f"""
<s>[INST] "Extract all anatomical and observation entities (ANAT-DP, OBS-DP, OBS-DA, OBS-U) from the input report and return them as a list of JSON objects with entity_type, entity_value, start_position, and end_position."
Report: {report} [/INST]
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )

    # === Decode and extract model output after [/INST] ===
    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    if "[/INST]" in decoded:
        cleaned_output = decoded.split("[/INST]")[-1].strip()
    else:
        cleaned_output = decoded.strip()

    # === Try parsing output ===
    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    cleaned_output = decoded.split("[/INST]")[-1].strip()

# === Extract only JSON array from [ ... ] ===
    try:
        # Use regex to extract the first JSON array in the string
        match = re.search(r"\[.*?\]", cleaned_output)
        if match:
            json_part = match.group(0)
            prediction = json.loads(json_part)
        else:
            raise ValueError("No JSON array found in model output.")
    except Exception as e:
        print("\n⚠️  Failed to parse this output:")
        print("Raw:", repr(cleaned_output))
        print(f"Error: {e}\n")
        prediction = {"error": str(e), "raw_output": cleaned_output}

    # === Save each result regardless of parsing success ===
    results.append({
        "report": report,
        "true_labels": item.get("labels", []),
        "entities": prediction
    })

# === Write results to file ===
output_path = "/home/spshetty/RadAnnotate/finetune/v4/testing/res/annotated_reports_test_new.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Predictions saved to {output_path}")
