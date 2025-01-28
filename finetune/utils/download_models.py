from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model name
model_name = "mistralai/Mistral-7B-v0.1"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save locally if needed
model.save_pretrained("./mixtral-7b")
tokenizer.save_pretrained("./mixtral-7b")