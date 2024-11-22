import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# Paths to model and tokenizer
base_model = "/home/spshetty/RadAnnotate/Meta-Llama-3-8B"
new_model = "/home/spshetty/RadAnnotate/finetuned_models/instr/epoch_10/checkpoint-1060"

# QLoRA config for loading in 4-bit quantization (optional)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load the model with quantization config (if necessary)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Handle missing pad token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check for mismatch in the tokenizer and model embedding size
if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))

def generate_annotation(report):
    # Create the input prompt using the same format as training
    formatted_text = f"System: You are given a clinical radiology report. Your task is to annotate the report using the following guidelines: Our schema defines two broad entity types: Observation and Anatomy. The Observation entity type includes two uncertainty levels: Definitely Present and Definitely Absent. In total, we have four entities: ANAT-DA (Anatomy - Definitely Absent), ANAT-DP (Anatomy - Definitely Present), OBS-DA (Observation - Definitely Absent), OBS-DP (Observation - Definitely Present).The schema also defines three relationships between entities: suggestive_of, located_at, and modify.\nUser: {report}\nAssistant:"

    # Tokenize the input report
    inputs = tokenizer(formatted_text, return_tensors="pt").to("cuda")

    # Generate the annotated report
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,  # Disable sampling for deterministic output
        num_return_sequences=1
    )

    # Decode the generated text
    annotated_report = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return annotated_report

if __name__ == "__main__":
    # Example report for inference
    report = "Patient shows signs of pneumonia and shortness of breath."

    # Generate the annotated report
    annotated_report = generate_annotation(report)
    print("Generated Annotated Report:")
    print(annotated_report)
