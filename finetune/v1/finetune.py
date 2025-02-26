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

# Clear GPU cache
torch.cuda.empty_cache()

# Initialize WandB for tracking
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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  

train_dataset = load_dataset('json', data_files='/home/spshetty/RadAnnotate/finetune/v1/data/finetune_instr_data.json')['train']
eval_dataset = load_dataset('json', data_files='/home/spshetty/RadAnnotate/finetune/v1/data/dev_instr.json')['train']

#print("Initial Train Dataset Columns:", train_dataset.column_names)
#print("Initial Eval Dataset Columns:", eval_dataset.column_names)

def format_chat_template(row):
    return {
        "text": f"System: {row.get('instruction', '')}\nUser: {row.get('input', '')}\nAssistant: {row.get('output', '')}"
    }

train_dataset = train_dataset.map(format_chat_template, num_proc=4, remove_columns=["instruction", "input", "output"])
eval_dataset = eval_dataset.map(format_chat_template, num_proc=4, remove_columns=["instruction", "input", "output"])


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("Final Train Dataset Columns:", train_dataset.column_names)
print("Final Eval Dataset Columns:", eval_dataset.column_names)

lora_config = LoraConfig(
    r=8,  
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"], 
    modules_to_save=["embed_tokens"],  
)

model.gradient_checkpointing_enable() 
model = prepare_model_for_kbit_training(model) 
model = get_peft_model(model, lora_config)  

model.get_input_embeddings().requires_grad_(True)

training_arguments = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    num_train_epochs=3,  
    learning_rate=5e-5,
    evaluation_strategy="steps",  
    eval_steps=50,               
    save_strategy="steps",       
    save_steps=50,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=True,
    group_by_length=True,
    load_best_model_at_end=True,
    remove_unused_columns=False  
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,  
    eval_dataset=eval_dataset,  
    args=training_arguments,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

trainer.train()

model.save_pretrained("./mistral-finetuned")
tokenizer.save_pretrained("./mistral-finetuned")

wandb.finish()
