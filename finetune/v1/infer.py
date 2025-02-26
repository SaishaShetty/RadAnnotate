import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
# Paths to model and tokenizer
fine_tuned_model_path = "/home/spshetty/RadAnnotate/finetune/v1/mistral-finetuned/checkpoint-201"  # Fine-tuned model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

print("Loading fine-tuned model...")
model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,
    torch_dtype=torch.float16,
    device_map="auto"  # Load efficiently to GPU
)

model = model.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_annotation(report):
    """
    Generate an annotated clinical report using the fine-tuned model.
    Ensures proper JSON formatting and removes invalid entries.
    """
    instruction = "Annotate the given clinical radiology report by identifying relevant entities (ANAT-DP, ANAT-DA, OBS-DP, OBS-DA) and their relations (suggestive_of, located_at, modify). Output the annotations in JSON format."
    prompt = f"""
    {instruction}
    
    Clinical Report:
    "{report}"
    
    JSON Output:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=10000,
        do_sample=False,
        num_return_sequences=1
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_part = response[start:end]

        json_part = json_part.replace("'", '"')
        parsed_output = json.loads(json_part)

        parsed_output = {k: v for k, v in parsed_output.items() if v is not None}
        return parsed_output

    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw output:", response)
        return {}

if __name__ == "__main__":
    test_report = "The lungs are clear without focal consolidation, or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process ."
    print("Running fine-tuned Mixtral model test...")
    annotated_report = generate_annotation(test_report)
    
    print("\nGenerated Annotated Report (Cleaned):")
    print(json.dumps(annotated_report, indent=4))
