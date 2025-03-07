import torch
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load stored reports & build FAISS index
json_file_path = "/home/spshetty/RadAnnotate/data_generation/mixtral/dataset/finetune_train.json"

if not os.path.exists(json_file_path):
    print(f"Warning: {json_file_path} not found! Creating an empty dataset.")
    with open(json_file_path, "w") as file:
        json.dump([], file)

try:
    with open(json_file_path, "r") as file:
        stored_reports_with_annotations = json.load(file)
except json.JSONDecodeError as e:
    print(f"JSON Parsing Error: {e}")
    exit(1)

stored_texts = [item["Report"] for item in stored_reports_with_annotations if "Report" in item]

if not stored_texts:
    print("No valid 'Report' entries found in the JSON file! Exiting.")
    exit(1)


report_embeddings = np.array(embedding_model.encode(stored_texts), dtype="float32")
dimension = report_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  
index.add(report_embeddings)

stored_reports_dict = {i: stored_reports_with_annotations[i] for i in range(len(stored_reports_with_annotations))}

def retrieve_similar_reports(report, k=5):
    """Retrieve relevant reports based on key terms."""
    query_text = report
    query_embedding = embedding_model.encode([query_text],dtype = "float32")
    _, indices = index.search(query_embedding, k)

    valid_indices = [idx for idx in indices[0] if idx >= 0 and idx < len(stored_reports_with_annotations)]

    return [stored_reports_dict[idx] for idx in valid_indices]


# Load fine-tuned Mistral model
fine_tuned_model_path = "/home/spshetty/RadAnnotate/finetune/v1/mistral-finetuned-v2/checkpoint-1084"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

print("Loading fine-tuned model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

model = model.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_annotation(report):
    """
    Generate an annotated clinical report using the fine-tuned Mistral model with RAG.
    Retrieves similar clinical reports and their annotations to improve output.
    """

    retrieved_reports = retrieve_similar_reports(report, k=1) 

    # Step 2: Construct prompt using fine-tuned model + RAG
    prompt = f"""
    <s><INST>
    [System]
    You are an expert in clinical report annotation.

    ###  Retrieved Context (for reference only, do NOT introduce new words from here unless present in the report):
    {retrieved_reports}

    Your task is to annotate **only relevant anatomy and observation entities** using the following steps:

    ### Step 1: Label each token with one of the following categories:
    - **ANAT-DP**: Anatomy - Definitely Present (e.g., "heart", "aorta").
    - **ANAT-DA**: Anatomy - Definitely Absent (e.g., "no pericardium").
    - **OBS-DP**: Observation - Definitely Present (e.g., "effusion").
    - **OBS-DA**: Observation - Definitely Absent (e.g., "no edema").

    ### Step 2: Identify Relations Between Entities:
    - **suggestive_of (Observation → Observation)**: e.g., "scoliosis" suggestive_of "asymmetry".
    - **located_at (Observation → Anatomy)**: Whene.g., "Normal" located_at "cardiomediastinal".
    - **modify (Observation → Observation or Anatomy → Anatomy)**: e.g., "left" modify "apex".

    *Important:* All directional terms should have a modify relation with their corresponding anatomy.

    ### Step 3: Format the output strictly as valid JSON:
    {{
        "entities": {{
            "1": {{
                "tokens": "string",
                "label": "string",
                "start_ix": int,
                "end_ix": int,
                "relations": [
                    "relation_type",
                    "entity_id"
                ]
            }}
        }}
    }}
    ### input:
    "{report}"

    ### **Strict Formatting Rules**
        - Annotate only the words that are present in "{report}". 
        - The output **MUST** be in **valid JSON format**.
        - **DO NOT** include any explanations or extra text.
        - The output must start with **"{{"** and end with **"}}"**.
        - If output is invalid JSON, 
   [/INST]
    ### output:
    </s>
    """

    # Step 3: Tokenize and Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():  
        outputs = model.generate(
            **inputs,
            max_new_tokens=10000,  
            do_sample=False,
            num_return_sequences=1
        )

    # Step 4: Extract and Parse JSON Output
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_part = response[start:end]

        json_part = json_part.replace("'", '"')
        parsed_output = json.loads(json_part)
        return parsed_output

    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw output:", response)
        return {}

if __name__ == "__main__":
    test_report = " The lungs are grossly clear . There is no focal consolidation , large effusion or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process"
    print("Running fine-tuned Mistral model test with RAG...")
    annotated_report = generate_annotation(test_report)
    
    print("\nGenerated Annotated Report (Cleaned):")
    print(json.dumps(annotated_report, indent=4))
