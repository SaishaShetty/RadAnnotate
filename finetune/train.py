from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

# Paths
base_model = "/home/spshetty/RadAnnotate/Meta-Llama-3-8B"
new_model = "finetuned_models/w/o_lora/adapter_model"

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

# Print the initial sizes of the tokenizer vocab and model embeddings
print(f"Initial Tokenizer vocab size: {len(tokenizer)}")
print(f"Initial Model input embeddings size: {model.get_input_embeddings().num_embeddings}")

# Resize token embeddings if necessary (must be done before applying LoRA)
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    print("Resizing token embeddings to match tokenizer size.")
    model.resize_token_embeddings(len(tokenizer))

# Check sizes again after resizing
print(f"Resized Model input embeddings size: {model.get_input_embeddings().num_embeddings}")

# Setup chat format (before LoRA)
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Load datasets
train_dataset = load_dataset('json', data_files={'train': '/home/spshetty/RadAnnotate/train_data/finetune_data.json'}, split='train')
test_dataset = load_dataset('json', data_files={'test': '/home/spshetty/RadAnnotate/test_data/finetune_data_test.json'}, split='test')

# Format dataset with a chat template
def format_chat_template(row):
    row_json = [
        {"role": "user", "content": str(row.get("Report", ""))},
        {"role": "assistant", "content": str(row.get("Annotated Report", ""))}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# Apply formatting to the datasets
train_dataset = train_dataset.map(format_chat_template, num_proc=4)
test_dataset = test_dataset.map(format_chat_template, num_proc=4)

# Training arguments
# Experiment with different learning rates, batch sizes, or epoch numbers
learning_rates = [2e-4]
batch_sizes = [4]

# Loop through learning rates and batch sizes for experiments
for lr in learning_rates:
    for bs in batch_sizes:
        training_arguments = TrainingArguments(
            output_dir=new_model,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            gradient_accumulation_steps=2,
            optim="paged_adamw_32bit",
            num_train_epochs=5,
            evaluation_strategy="steps",
            eval_steps=100,
            logging_steps=10,
            warmup_steps=10,
            logging_strategy="steps",
            learning_rate=lr,
            fp16=False,
            bf16=False,
            group_by_length=True
        )
       
        # Trainer setup
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False,
            dataset_text_field="text",
        )
        
        # Start training
        trainer.train()

        # Print a message to show progress
        print(f"Training complete for learning rate {lr} and batch size {bs}")
