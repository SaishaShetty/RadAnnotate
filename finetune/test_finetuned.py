import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Paths to model and tokenizer
base_model_path = "/home/spshetty/RadAnnotate/finetune/mixtral-7b"  # Base Mixtral model path
peft_checkpoint_path  = "/home/spshetty/RadAnnotate/finetune/mistral-finetuned/checkpoint-66"  # Fine-tuned Mixtral model

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Handle missing pad token
if tokenizer.pad_token is None:
    print("Setting pad_token to eos_token...")
    tokenizer.pad_token = tokenizer.eos_token

# Load the base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Resize embeddings to match the checkpoint size (e.g., 32002)
checkpoint_vocab_size = 32002  # Update this based on your checkpoint's tokenizer size
current_vocab_size = model.get_input_embeddings().num_embeddings
if checkpoint_vocab_size != current_vocab_size:
    print(f"Resizing embeddings from {current_vocab_size} to {checkpoint_vocab_size}...")
    model.resize_token_embeddings(checkpoint_vocab_size)

# Load LoRA adapters
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, peft_checkpoint_path)

# Function to generate annotations
def generate_annotation(report):
    """
    Generate an annotated clinical report using the Mixtral model.
    """
    prompt = f"System: Annotate the clinical report.\nUser: {report}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=2000,
        do_sample=False,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model
if __name__ == "__main__":
    test_report = "The lungs are clear without focal consolidation, or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
    print("Running Mixtral model test...")
    annotated_report = generate_annotation(test_report)
    print("\nGenerated Annotated Report:")
    print(annotated_report)
