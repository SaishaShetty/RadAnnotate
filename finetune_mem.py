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


# Paths
base_model = "/home/spshetty/RadAnnotate/Meta-Llama-3-8B"
new_model = "finetuned_models/instr/epoch_10_early_stop"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model config and model
config = AutoConfig.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Resize token embeddings if necessary before LORA
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config for trainable adapters
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Attach LoRA adapters to the model
model = get_peft_model(model, peft_config)

# Load datasets
train_dataset = load_dataset('json', data_files={'train': '/home/spshetty/RadAnnotate/data/instr_data/finetune_instr_data.json'}, split='train')
test_dataset = load_dataset('json', data_files={'test': '/home/spshetty/RadAnnotate/data/instr_data/finetune_instr_dev.json'}, split='test')

# Function to format dataset for chat format
tokenizer.chat_template = {
    "system": "System: {content}",
    "user": "User: {content}",
    "assistant": "Assistant: {content}"
}

def format_chat_template(row):
    
    formatted_text = f"System: {row.get('Instruction', '')}\nUser: {row.get('Report (Input)', '')}\nAssistant: {row.get('Annotated Report (Output)', '')}"
    
    row["text"] = formatted_text
    return row
"""def format_chat_template(row):

    row_json = [
        {"role": "system", "content": str(row.get("Instruction", ""))},  # Use the instruction directly from the JSON
        {"role": "user", "content": str(row.get("Report (Input)", ""))},
        {"role": "assistant", "content": str(row.get("Annotated Report (Output)", ""))}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row
    """
train_dataset = train_dataset.map(format_chat_template, num_proc=4)
test_dataset = test_dataset.map(format_chat_template, num_proc=4)

# Training arguments with a single batch size and learning rate to avoid excessive memory usage
learning_rate = 5e-5
batch_size = 2

# Training arguments
# Training arguments with load_best_model_at_end set to True
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=10,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=10,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=learning_rate,
    fp16=True,
    bf16=False,
    group_by_length=True,
    load_best_model_at_end=True  # This ensures the best model is loaded at the end
)

# Initialize early stopping callback with patience parameter
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,  # Stop training if no improvement after 2 evaluations
    early_stopping_threshold=0.01  # Minimum improvement to reset patience
)

# Train the model with LoRA adapters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    dataset_text_field="text",
    callbacks=[early_stopping]  # Add early stopping to the callbacks list
)

print(f"Training with batch size: {batch_size} and learning rate: {learning_rate}")

trainer.train()

# Optionally, print detailed memory summary
print("Detailed memory summary after training:")
print(torch.cuda.memory_summary(device=torch.device('cuda'), abbreviated=False))