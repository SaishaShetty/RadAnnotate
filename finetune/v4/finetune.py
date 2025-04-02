import torch
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer
import wandb
import random

# Clear GPU cache
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize WandB
wandb.init(
    project="v3",
    name="mistral-7b-v0.1-lora-v2",
    config={
        "learning_rate": 5e-6,
        "batch_size": 1,
        "epochs": 4,
        "model": "mistralai/Mistral-7B-v0.1"
    }
)

# Load Model
model_name = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

# Load and split dataset
with open("/home/spshetty/RadAnnotate/finetune/v4/mix_data_cleaned_fixed_sanitized.jsonl") as f:
    data = [json.loads(line) for line in f if line.strip()]

random.shuffle(data)
split_index = int(0.9 * len(data))
train_data = data[:split_index]
eval_data = data[split_index:]

train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# Format dataset for training (text with instruction)
def format_chat_template(row):
    instruction = row.get("instruction", "")
    input_text = row.get("input", "").strip()
    output = json.dumps(row.get("output")) if isinstance(row.get("output"), (dict, list)) else row.get("output", "").strip()

    return {
        "text": f"<s>[INST] {instruction}\n{input_text} [/INST] {output} </s>"
    }

train_dataset = train_dataset.map(format_chat_template, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(format_chat_template, remove_columns=eval_dataset.column_names)

# LoRA config
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    modules_to_save=["embed_tokens", "lm_head"]
)

# Prepare model for training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Tokenization and masking instruction for loss
MAX_TOKENS = 512

def tokenize_function(examples):
    texts = examples["text"]
    
    # Tokenize entire batch
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="longest",
        max_length=MAX_TOKENS,
        return_tensors="pt"
    )

    input_ids = tokenized["input_ids"]
    labels = input_ids.clone()

    for i, text in enumerate(texts):
        instruction_end = text.find("[/INST]") + len("[/INST]")
        if instruction_end > 0:
            # Only tokenize once per instruction prefix
            inst_tokens = tokenizer(
                text[:instruction_end],
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"]
            labels[i, :inst_tokens.shape[1]] = -100

    tokenized["labels"] = labels
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Training arguments
training_arguments = TrainingArguments(
    output_dir="./mistral-ner-v4",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",           # Changed from 'epoch'
    eval_steps=100,                        # Evaluate every 100 steps (adjust based on dataset size)
    save_strategy="steps",                 # Save checkpoint by steps
    save_steps=100,                        # Save every 100 steps
    logging_strategy="steps",             # Log every n steps
    logging_steps=50,                      # Log every 50 steps
    load_best_model_at_end=True,
    save_total_limit=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=6,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=False,
    bf16=True,
    logging_dir="./logs",
    max_grad_norm=5,
    remove_unused_columns=False
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_arguments,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train(resume_from_checkpoint=False)

# Save final model
model.save_pretrained("mistral-ner-v4-lora-only")
tokenizer.save_pretrained("mistral-ner-v4-lora-only")

wandb.finish()