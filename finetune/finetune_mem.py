from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
import torch
from transformers import EarlyStoppingCallback
import wandb

# Step 1: Initialize Weights & Biases
wandb.init(
    project="mistral-finetuning",
    name="mistral-7b-v0.1-lora",
    config={
        "learning_rate": 5e-5,
        "batch_size": 4,
        "epochs": 3,
        "model": "mistralai/Mistral-7B-v0.1"
    }
)

# Step 2: Load the Model and Tokenizer
model_name = "mistralai/Mistral-7B-v0.1"

# QLoRA Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model config and model
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Resize token embeddings if necessary
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA Config for Trainable Adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Attach LoRA adapters to the model
model = get_peft_model(model, lora_config)

# Load datasets
train_dataset = load_dataset('json', data_files={'train': 'finetune_data.json'}, split='train')

def format_chat_template(row):
    formatted_text = f"System: {row.get('instruction', '')}\nUser: {row.get('input', '')}\nAssistant: {row.get('output', '')}"
    row["text"] = formatted_text
    return row

train_dataset = train_dataset.map(format_chat_template, num_proc=4)

# Training arguments
training_arguments = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    fp16=True,
    group_by_length=True,
    load_best_model_at_end=True
)

# Train the model with LoRA adapters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    dataset_text_field="text"
)

trainer.train()

# Save the model
model.save_pretrained("./mistral-finetuned")
tokenizer.save_pretrained("./mistral-finetuned")

# Test the model


wandb.finish()
