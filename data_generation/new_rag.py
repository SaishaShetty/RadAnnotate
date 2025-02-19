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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = MistralTokenizer.from_file("/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model/tokenizer.model.v3")

json_file_path = "/home/spshetty/RadAnnotate/data_generation/mixtral/dataset/finetune_train.json"

# ✅ Ensure JSON File Exists
if not os.path.exists(json_file_path):
    print(f"⚠️ Warning: {json_file_path} not found! Creating an empty dataset.")
    with open(json_file_path, "w") as file:
        json.dump([], file)

# ✅ Load JSON Dataset
try:
    with open(json_file_path, "r") as file:
        stored_reports_with_annotations = json.load(file)
except json.JSONDecodeError as e:
    print(f"❌ JSON Parsing Error: {e}")
    exit(1)

# ✅ Extract "Report" Texts Safely
stored_texts = [item["Report"] for item in stored_reports_with_annotations if "Report" in item]

if not stored_texts:
    print("❌ No valid 'Report' entries found in the JSON file! Exiting.")
    exit(1)

# ✅ Convert reports into FAISS embeddings
report_embeddings = np.array(embedding_model.encode(stored_texts), dtype="float32")

# ✅ Ensure FAISS uses CPU (to prevent CUDA OOM)
dimension = report_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # ✅ FAISS in CPU mode
index.add(report_embeddings)

stored_reports_dict = {i: stored_reports_with_annotations[i] for i in range(len(stored_reports_with_annotations))}

print(f"✅ FAISS index built with {len(stored_reports_with_annotations)} reports.")


def retrieve_similar_reports(user_terms, k=4):
    """Retrieve relevant reports based on key terms."""
    query_text = " ".join(user_terms)  # Create a query from user-defined terms
    query_embedding = embedding_model.encode([query_text])
    _, indices = index.search(np.array(query_embedding, dtype="float32"), k)
    
    # ✅ Ensure valid indices (no negative or out-of-range values)
    valid_indices = [idx for idx in indices[0] if idx >= 0 and idx < len(stored_reports_with_annotations)]
    
    return [stored_reports_dict[idx] for idx in valid_indices]


def mixtral_generate_data(user_terms, total_samples=1, batch_size=1, output_file="/home/spshetty/RadAnnotate/data_generation/outputs/check.json"):
    global model
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as file:
        file.write("[\n")

    total_results = 0  

    for batch_start in range(0, total_samples, batch_size):
        torch.cuda.empty_cache()  # Clear GPU memory before every run

        # Reload model to avoid cache issues
        if "model" in globals():
            del model  # Delete model if it exists to avoid memory issues

        model = Transformer.from_folder('/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model',dtype=torch.float16)

        # **Retrieve 4 Relevant Reports Based on User Key Terms**
        retrieved_reports = retrieve_similar_reports(user_terms, k=4)

        prompt = f"""
        <s><INST>
        [System]
            You are a radiology expert specializing in generating structured synthetic reports.

            **User-Provided Key Terms:**  
            {', '.join(user_terms)}

            **Retrieved Annotated Reports for Context:**  
            {json.dumps(retrieved_reports, indent=4)}


            **Your Task:**  
            1. **Generate Annotations Based on Individual Tokens:**  
                - Label each token with one of the following categories:  
                    - **ANAT-DP**: Anatomy - Definitely Present (e.g., "heart," "aorta").  
                    - **ANAT-DA**: Anatomy - Definitely Absent (e.g., "no pericardium").  
                    - **OBS-DP**: Observation - Definitely Present (e.g., "effusion").  
                    - **OBS-DA**: Observation - Definitely Absent (e.g., "no edema").  
                - Use the retrieved reports to ensure clinical relevance.  
                - Do not assume **adjectives** as anatomical structures (e.g., "pulmonary" is not an anatomy).  
                - Avoid annotating **auxiliary words** like "no" or "mild."  

            2. **Construct a Report Using Only the Tokens in Annotations:**  
                - The final report **must** include every annotated token.  
                - No additional words or phrases outside of the given tokens.  
                - Ensure correct **clinical phrasing** while maintaining logical relations.  
                - The report should be diverse and not repetitive.  

            3. **Incorporate Relations into the Report:**  
                - Establish meaningful relationships among tokens:  

                - **`suggestive_of (Observation → Observation):`**  
                - **Example:** "Moderate scoliosis, causing asymmetry of the ribcage."  
                - `"scoliosis"` is **suggestive_of** `"asymmetry"`.  

                - **`located_at (Observation → Anatomy):`**  
                - **Example:** "The cardiomediastinal contours appear normal."  
                - `"Normal"` is **located_at** `"cardiomediastinal"`.  

                - **`modify (Observation → Observation)` or `(Anatomy → Anatomy):`**  
                - **Example:** "The left apex is slightly obscured."  
                - `"left"` is **modify** `"apex"`.  

                - Directional terms (e.g., "left", "right") **must** have a `"modify"` relation with the corresponding anatomy.   

            4. **Output Format:**  
                - Strict **JSON format** as a list of dictionaries:  
                    - **`Annotations`**: A dictionary of token-level annotations and their relations, sorted by `start_ix`.  
                    - **`Report`**: The generated textual report.  
                - The generated output **must strictly follow the retrieved report structure**:
                {{
                    "Annotations": {{
                        "1": {{
                            "tokens": "Lungs",
                            "label": "ANAT-DP",
                            "start_ix": 36,
                            "end_ix": 36,
                            "relations": []
                        }},
                        "2": {{
                            "tokens": "clear",
                            "label": "OBS-DP",
                            "start_ix": 38,
                            "end_ix": 38,
                            "relations": [
                            [
                                "located_at",
                                "1"
                            ]
                            ]
                        }},
                        "3": {{
                            "tokens": "Normal",
                            "label": "OBS-DP",
                            "start_ix": 40,
                            "end_ix": 40,
                            "relations": [
                            [
                                "located_at",
                                "4"
                            ],
                            [
                                "located_at",
                                "5"
                            ],
                            [
                                "located_at",
                                "7"
                            ]
                            ]
                        }},
                        "4": {{
                            "tokens": "cardiomediastinal",
                            "label": "ANAT-DP",
                            "start_ix": 41,
                            "end_ix": 41,
                            "relations": []
                        }},
                        "5": {{
                            "tokens": "hilar",
                            "label": "ANAT-DP",
                            "start_ix": 43,
                            "end_ix": 43,
                            "relations": []
                        }},
                        "6": {{
                            "tokens": "silhouettes",
                            "label": "ANAT-DP",
                            "start_ix": 44,
                            "end_ix": 44,
                            "relations": [
                            [
                                "modify",
                                "4"
                            ],
                            [
                                "modify",
                                "5"
                            ]
                            ]
                        }},
                        "7": {{
                            "tokens": "pleural",
                            "label": "ANAT-DP",
                            "start_ix": 46,
                            "end_ix": 46,
                            "relations": []
                        }},
                        "8": {{
                            "tokens": "surfaces",
                            "label": "ANAT-DP",
                            "start_ix": 47,
                            "end_ix": 47,
                            "relations": [
                            [
                                "modify",
                                "7"
                            ]
                            ]
                        }}
                    }} 
                "Report": "Patient has been extubated . Lungs are clear . Normal cardiomediastinal and hilar silhouettes and pleural surfaces ."
                }}

            5. **Validation Rules:**  
                - Ensure **all annotations appear in the report**.  
                - **No missing tokens.**  
                - Validate that **relations reference valid tokens**.  
                - Check for **logical consistency** (e.g., `effusion located_at lungs`, but NOT `effusion located_at trachea`).  

            6. **Realism in the Report:**  
                - Ensure **every observation token has a logical anatomical location**.  
                - Avoid **ambiguous** or **clinically illogical** reports.  
                - Maintain the **diversity** of reports.  

            7. **Final Validation:**  
                - Ensure all annotations **match** the report content.  
                - Logical **relations must be correct** and medically accurate.  

        [/System]
        **Important**: Your output **must strictly start with `{{` and end with `}}`** only.  
        Do **not** include extra text, explanations, or descriptions outside the JSON structure.  
        If you fail to comply, the output will be considered **invalid**.  
        After every report generated, add a `,` to separate entries.  

        [/INST]
        """


        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        try:
            out_tokens, _ = generate(
                [tokens], 
                model, 
                max_tokens=5000,  
                temperature=0.6,  
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
        except RuntimeError as e:
            print(f"Runtime Error: {e}")
            torch.cuda.empty_cache()
            continue  

        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        try:
            print(result)
            parsed_res = json.loads(result)  
            print(parsed_res)
        except json.JSONDecodeError as e:
            print(f"JSON Error: {e}")
            parsed_res = {"error": "Invalid JSON output"}

        with open(output_file, "a") as file:
            json.dump(parsed_res, file, indent=4)
            if batch_start + batch_size < total_samples:
                file.write(",\n")

        total_results += len(parsed_res) if isinstance(parsed_res, list) else 1

    with open(output_file, "a") as file:
        file.write("\n]")

    print(f"Generated {total_results} reports and saved to {output_file}")

# ✅ Run the synthetic report generation with RAG, user provides key terms
user_terms = ["heart", "aorta"]  
mixtral_generate_data(user_terms, total_samples=1, batch_size=1, output_file="/home/spshetty/RadAnnotate/data_generation/outputs/test.json")
