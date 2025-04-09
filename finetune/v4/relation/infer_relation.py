import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Step 1: Load fine-tuned model and tokenizer ===
model_path = "/home/spshetty/RadAnnotate/finetune/v4/relation/mistral-ner-v4-relation/checkpoint-3100"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Step 2: Define prompt ===
input = "Report: There are no new areas of consolidation .\nEntities: {\"new\": \"OBS-DA\", \"areas\": \"OBS-DA\", \"consolidation\": \"OBS-DA\"}"
prompt = f"""<s>[INST] Extract valid relations between labeled entities in the report.
{input} [/INST]"""

# === Step 3: Tokenize and generate ===
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

# === Step 4: Decode ===
generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

# === Step 5: Extract output JSON after [/INST]
# Helpful if model echoes instruction + prompt before the output
response = generated_text.split("[/INST]")[-1].strip()

print("=== Generated Output ===")
print(response)
