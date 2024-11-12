import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

# Paths to model and tokenizer
base_model = "/home/spshetty/RadAnnotate/Meta-Llama-3-8B"
new_model = "/home/spshetty/RadAnnotate/finetuned_models/epoch_5/checkpoint-1060"

# QLoRA config for loading in 4-bit quantization (optional)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model config and model (for inference, no need for training configs)
config = AutoConfig.from_pretrained(base_model)

# Load the model with quantization config (if necessary)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",  # Adjust based on GPU availability
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained(base_model)

if tokenizer.pad_token is None:
    # Option 1: Use `eos_token` as `pad_token`
    tokenizer.pad_token = tokenizer.eos_token
    
    # Option 2 (Alternative): Add a new `[PAD]` token explicitly
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))  # Resize if new tokens are added

# Load the tokenizer


# Check for mismatch in the tokenizer and model embedding size
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Model embedding size before resizing: {model.get_input_embeddings().num_embeddings}")

# Resize token embeddings to match the tokenizer's vocabulary size if needed
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    print(f"Resizing model token embeddings from {model.get_input_embeddings().num_embeddings} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

# Check the model's embedding size after resizing
print(f"Model embedding size after resizing: {model.get_input_embeddings().num_embeddings}")

# Example inference input (raw report that needs annotation)
def format_chat_prompt(report):
    # Mimicking a user-assistant conversation style format with explicit instructions
    return f"User: Here is the patient's report: {report}. Can you please provide the annotated report?\nAssistant:"


new_data = [
    "FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The lungs are clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
]
# Prepare the input prompts in the user-assistant format
input_prompts = [format_chat_prompt(report) for report in new_data]

# Tokenize the input prompts (reports)
inputs = tokenizer(input_prompts, return_tensors="pt", padding=True, truncation=True)

# Generate the annotated reports using the fine-tuned model (inference)
output_ids = model.generate(
    inputs['input_ids'], 
    max_length=512,  # Adjust max length based on expected report size
    num_return_sequences=1,  # Number of completions to generate
    do_sample=False,         # Set to True for non-deterministic output
    temperature=0.7          # Controls randomness of sampling
)

# Decode the generated output (annotated report)
generated_texts = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]

# Display the original and annotated reports
for i, generated_text in enumerate(generated_texts):
    print(f"Original Report {i+1}: {new_data[i]}")
    print(f"Annotated Report {i+1}: {generated_text}")
    print("-" * 80)
