import faiss
import numpy as np
import json
import os
import torch
from sentence_transformers import SentenceTransformer
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import random

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = MistralTokenizer.from_file("/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model/tokenizer.model.v3")

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

def retrieve_similar_reports(user_terms, k=2, randomize=True):
    """Retrieve relevant reports based on key terms, with optional randomization."""
    query_text = " ".join(user_terms) 
    query_embedding = embedding_model.encode([query_text])
    _, indices = index.search(np.array(query_embedding, dtype="float32"), 20)  

    valid_indices = [idx for idx in indices[0] if idx >= 0 and idx < len(stored_reports_with_annotations)]
    
    if randomize:
        random.shuffle(valid_indices)  

    return [stored_reports_dict[idx] for idx in valid_indices[:k]] 

USER_TERMS_POOL = ["system","function","recurrent","walls","contusion","absent"]

def get_random_user_terms(pool, min_terms=1, max_terms=2):
    """Randomly select a subset of user terms for diversity"""
    num_terms = random.randint(min_terms, max_terms) 
    return random.sample(pool, num_terms)  

def mixtral_generate_data(
    total_samples=1, 
    batch_size=1, 
    output_file="/home/spshetty/RadAnnotate/data_generation/nf_new_data/nf_v2.json",
    user_terms_pool=USER_TERMS_POOL
):
    global model
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_results = 0  # Track total reports generated

    for batch_start in range(0, total_samples, batch_size):
        torch.cuda.empty_cache()

        if "model" in globals():
            del model  

        model = Transformer.from_folder('/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model', dtype=torch.float16)
        user_terms = get_random_user_terms(user_terms_pool, min_terms=2, max_terms=3)  # Pick between 2 to 4 terms

        print(f"🔑 Using User Terms for this batch: {user_terms}")

        # Retrieve 3 Relevant Reports Based on User Key Terms
        retrieved_reports = retrieve_similar_reports(user_terms, k=3, randomize=True)

        # --- Prompt construction logic (same as your provided one) ---
        prompt = f"""
        <s><INST>
        [System]
            You are a radiology expert specializing in generating structured synthetic reports.

            **User-Provided Key Terms:**  
            {', '.join(user_terms)}

            **Retrieved Annotated Reports for Context:**  
            {json.dumps(retrieved_reports, indent=5)}

            **Your Task:**  
            1. **Generate Annotations Based on Individual Tokens:**  
                - Label each token with one of the following categories:  
                    - **ANAT-DP**: Anatomy - Definitely Present (e.g., "heart," "aorta").  
                    - **OBS-U**: Observation - Uncertain (e.g., "potentially left basal pneumonia").  
                    - **OBS-DP**: Observation - Definitely Present (e.g., "effusion").  
                    - **OBS-DA**: Observation - Definitely Absent (e.g., "no edema").  
                - Use the retrieved reports to ensure clinical relevance.  
                - Do not assume **adjectives** as anatomical structures (e.g., "pulmonary" is not an anatomy).  
                - Avoid annotating **auxiliary words** like "no" or "mild."  

            2. **Construct a Report Using Only the Tokens in Annotations:**  
                - The final report **must** include every annotated token.  
                - No additional words or phrases outside of the given tokens.  
                - The report should be diverse and not repetitive.  

            3. **Output Format:**  
                - Strict **JSON format** as a list of dictionaries:  
                    - **`Labels`**: A dictionary of token-level annotations.  
                    - **`Report`**: The generated textual report.  
                - The generated output **must strictly follow the retrieved report structure**:
                {{
                "Report": "Pulmonary vasculature is normal .",
                "labels": {{
                    "Pulmonary": "ANAT-DP",
                    "vasculature": "ANAT-DP",
                    "normal": "OBS-DP"
                }}
            }}
                
            4. **Validation Rules:**  
                - Ensure **all annotations appear in the report**.  
                - **No missing tokens.**  
                - Validate that **relations reference valid tokens**.  

            5. **Realism in the Report:**  
                - Ensure **every observation token has a logical anatomical location**.  
                - Avoid **ambiguous** or **clinically illogical** reports.  
                - Maintain the **diversity** of reports.  

        [/System]
        **Important**: Your output **must strictly start with `{{` and end with `}}`** only.  
        Do **not** include extra text, explanations, or descriptions outside the JSON structure.  
        If you fail to comply, the output will be considered **invalid**.  
        After every report generated, add a `,` to separate entries. 
        Output only the json file '[]'. Start the response with '[' and end with ']' 

        [/INST]
        """
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        try:
            out_tokens, _ = generate(
                [tokens], 
                model, 
                max_tokens=10000,
                temperature=0.6,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
        except RuntimeError as e:
            print(f"Runtime Error: {e}")
            torch.cuda.empty_cache()
            continue  

        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        try:
            parsed_res = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"JSON Error: {e}")
            parsed_res = {"error": "Invalid JSON output"}
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as file:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, list): 
                        existing_data = []
            except json.JSONDecodeError:
                existing_data = [] 
        else:
            existing_data = []
        if isinstance(parsed_res, list):
            existing_data.extend(parsed_res)
        else:
            existing_data.append(parsed_res)
        with open(output_file, "w") as file:
            json.dump(existing_data, file, indent=4)

        total_results += len(parsed_res) if isinstance(parsed_res, list) else 1

    print(f"✅ Generated {total_results} reports and appended to {output_file}")

mixtral_generate_data(total_samples=30, batch_size=1, output_file="/home/spshetty/RadAnnotate/data_generation/nf_new_data/nf_v6.json")
