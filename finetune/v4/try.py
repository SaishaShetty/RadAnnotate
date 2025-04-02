import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Step 1: Load model and tokenizer ===
model_path = "/home/spshetty/RadAnnotate/finetune/v4/mistral-ner-v4/checkpoint-4000"  # e.g., "./outputs/checkpoint-4000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

# === Step 2: Prepare your prompt ===
prompt = """
<s>[INST] You are given a clinical radiology report. Your task is to annotate the report using the following schema:

Entity Types: 
- ANAT-DP: Anatomy - Definitely Present
- ANAT-DA: Anatomy - Definitely Absent
- OBS-DP: Observation - Definitely Present
- OBS-DA: Observation - Definitely Absent

Output format (JSON):
{
  "Report": "Text of the report",
  "Labels": [
    {
      "entity_type": "ANAT-DP",
      "entity_value": "right lower lobe",
      "start_position": 38,
      "end_position": 54
    },
    ...
  ]
}

Report: The right lung has opacity . [/INST]
"""

# === Step 3: Tokenize and run inference ===
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,          # deterministic output
        temperature=0.0,
        return_dict_in_generate=True,
        output_scores=False
    )

# === Step 4: Decode and print output ===
generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print("=== Generated Output ===\n")
print(generated_text)
