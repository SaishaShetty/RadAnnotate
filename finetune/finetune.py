from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

base_model = "/home/spshetty/RadAnnotate/Meta-Llama-3-8B"
dataset_name = "/home/spshetty/RadAnnotate/train_data"
testing = "/home/spshetty/RadAnnotate/dev/finetune_dev.json"
new_model = "new"
torch_dtype = torch.float16
attn_implementation = "eager"
# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation,
    ignore_mismatched_sizes = True
)


tokenizer = AutoTokenizer.from_pretrained(base_model)
#model = AutoModelForCausalLM.from_pretrained(base_model, ignore_mismatched_sizes=True)

model.resize_token_embeddings(len(tokenizer))
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
model = get_peft_model(model, peft_config)

#Importing the dataset

from datasets import load_dataset

#train_dataset = load_dataset('json', data_files={'train': dataset_name}, split='train')
train_dataset = load_dataset('json', data_files={'train': '/home/spshetty/RadAnnotate/train_data/finetune_data.json'}, split='train')

test_dataset = load_dataset('json', data_files={'test': '/home/spshetty/RadAnnotate/test_data/finetune_data_test.json'}, split='test')

def format_chat_template(row):
    row_json = [
        {"role": "user", "content": str(row.get("Report", ""))},
        {"role": "assistant", "content": str(row.get("Annotated Report", ""))}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

train_dataset = train_dataset.map(
    format_chat_template,
    num_proc=4,
)

test_dataset  = test_dataset.map(
    format_chat_template,
    num_proc=4
)
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=100,  
    logging_steps=10,  
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True
)

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


trainer.train()