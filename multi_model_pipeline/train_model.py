import os
import json
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# -----------------------------
# GPU Setup
# -----------------------------
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# WandB Initialization
# -----------------------------
wandb.init(
    project="v4",
    name="qwen2.5-ner",
    config={
        "learning_rate": 2e-4,
        "batch_size": 1,
        "epochs": 3,
        "model": "Qwen/Qwen2.5-7B"
    }
)

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
model_name = "Qwen/Qwen2.5-7B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------------
# Load JSONL Data
# -----------------------------
with open("formatted_obs_da.jsonl") as f:
    train_data = [json.loads(line) for line in f if line.strip()]

with open("dev.jsonl") as f:
    eval_data = [json.loads(line) for line in f if line.strip()]

# -----------------------------
# Format Prompts
# -----------------------------
def format_for_base_model(row):
    instruction = row.get("instruction", "")
    input_text = row.get("input", "").strip()
    output = row.get("output", "")
    if isinstance(output, (dict, list)):
        output = json.dumps(output)

    prompt = f"""Task: {instruction}

Input: {input_text}

Output: {output}{tokenizer.eos_token}"""
    return {"text": prompt}

train_dataset = Dataset.from_list(train_data).map(format_for_base_model)
eval_dataset = Dataset.from_list(eval_data).map(format_for_base_model)

# -----------------------------
# Tokenization
# -----------------------------
MAX_LENGTH = 1024

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)

# -----------------------------
# LoRA Configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

# -----------------------------
# Prepare Model for Training
# -----------------------------
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
print(f"Total parameters: {model.num_parameters():,}")

# -----------------------------
# Training Arguments
# -----------------------------
training_arguments = TrainingArguments(
    output_dir="./qwen2.5_ner_output",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_strategy="steps",
    logging_steps=25,
    load_best_model_at_end=True,
    save_total_limit=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    dataloader_drop_last=True,
    report_to="wandb",
    run_name="qwen2.5-base-ner-lora",
    max_grad_norm=1.0,
    remove_unused_columns=False,
    group_by_length=True,
    dataloader_num_workers=2,
)

# -----------------------------
# Trainer Initialization
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_arguments,
    tokenizer=tokenizer
)

print("Starting training...")
trainer.train()

# -----------------------------
# Save Model and Adapter
# -----------------------------
print("Saving full fine-tuned model...")
trainer.save_model("./NER_qwen2.5_final")
tokenizer.save_pretrained("./NER_qwen2.5_final")

print("Saving LoRA adapter only...")
model.save_pretrained("./NER_qwen2.5_adapter")

print("Training completed successfully.")
wandb.finish()
