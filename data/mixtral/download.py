from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Initialize the text generation pipeline with the Mistral model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
pipe = pipeline("text-generation", model=model_name)
# Define chat messages with roles and content
messages = [
    {"role": "system", "content": "You are a radiological expert capable of generating diverse synthetic clinical reports and their annotations. Your task is to create variations in findings, conditions, and anatomical structures related to the chest to help build a comprehensive dataset."},
    {"role": "user", "content": "Explain the concept of artificial intelligence."},
]
# Generate response using the pipeline
response = pipe(messages, max_new_tokens=128)[0]['generated_text']
# Print the assistant's response
print(response)