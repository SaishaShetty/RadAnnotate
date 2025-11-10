import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ------------------- Config -------------------
test_set_path = "/home/spshetty/RadAnnotate/finetune/v4/evaluation_all_models/gold_data/test_set.json"
output_file = "res_with_labels.json"
#FIND ENTITY PROMPT
prompt_file = "prompt.txt"
checkpoint_interval = 10

model_name = "Qwen/Qwen2.5-7B"
adapter_path = "./NER_qwen2.5_adapter"

# ------------------- Load Prompt -------------------
if not os.path.exists(prompt_file):
    raise FileNotFoundError(f"Prompt file '{prompt_file}' not found.")
with open(prompt_file, "r") as f:
    instruction = f.read().strip()

# ------------------- Load Tokenizer & Model -------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# ------------------- Load Test Set -------------------
with open(test_set_path, "r") as f:
    test_data = json.load(f)

# ------------------- Resume from Checkpoint -------------------
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        results = json.load(f)
    processed_indices = set(entry["index"] for entry in results)
    print(f"Resuming from checkpoint. {len(results)} reports already processed.")
else:
    results = []
    processed_indices = set()
    print("Starting new inference run.")

# ------------------- Inference Loop -------------------
for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Processing Reports"):
    if idx in processed_indices:
        continue

    report = item.get("Report", "").strip()
    true_labels = item.get("entities", [])

    if not report:
        continue

    prompt = f"Task: {instruction}\n\nInput: {report}\n\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_output = decoded.split("Output:")[-1].strip()

    try:
        parsed = json.loads(generated_output)
    except json.JSONDecodeError:
        parsed = None

    results.append({
        "index": idx,
        "report": report,
        "true_labels": true_labels,
        "model_output": parsed if parsed is not None else generated_output
    })

    # Save periodically
    if len(results) % checkpoint_interval == 0 or idx == len(test_data) - 1:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Checkpoint saved at {len(results)} reports")

# ------------------- Done -------------------
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"All reports processed. Final output saved to {output_file}")
