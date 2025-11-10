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

# -----------------------------
# Setup
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = MistralTokenizer.from_file(
    "/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model/tokenizer.model.v3"
)

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


# -----------------------------
# Helper Functions
# -----------------------------
def retrieve_similar_reports(user_terms, k=2, randomize=True):
    """Retrieve relevant reports based on key terms, with optional randomization."""
    query_text = " ".join(user_terms)
    query_embedding = embedding_model.encode([query_text])
    _, indices = index.search(np.array(query_embedding, dtype="float32"), 20)
    valid_indices = [idx for idx in indices[0] if 0 <= idx < len(stored_reports_with_annotations)]
    if randomize:
        random.shuffle(valid_indices)
    return [stored_reports_dict[idx] for idx in valid_indices[:k]]


# -----------------------------
# Check term_pool.txt for some keywords used to generate synthetic data
# -----------------------------
USER_TERMS_POOL = ["lungs", "pneumonia"]

def get_random_user_terms(pool, min_terms=1, max_terms=2):
    num_terms = random.randint(min_terms, max_terms)
    return random.sample(pool, num_terms)


def load_prompt_template(template_path):
    """Load the base prompt text."""
    with open(template_path, "r") as f:
        return f.read()


# -----------------------------
# Main Function
# -----------------------------
def mixtral_generate_data(
    total_samples=1,
    batch_size=1,
    output_file="output.json",
    prompt_template_path="prompts/syn_data_generation_prompt.txt",
    user_terms_pool=USER_TERMS_POOL,
):
    global model
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    prompt_log_dir = os.path.join(os.path.dirname(output_file), "prompt_logs")
    os.makedirs(prompt_log_dir, exist_ok=True)

    # Load the base prompt template once
    base_prompt = load_prompt_template(prompt_template_path)
    total_results = 0

    for batch_start in range(0, total_samples, batch_size):
        torch.cuda.empty_cache()
        if "model" in globals():
            del model
        model = Transformer.from_folder(
            "/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model", dtype=torch.float16
        )

        user_terms = get_random_user_terms(user_terms_pool, min_terms=2, max_terms=3)
        print(f"🧠 Using User Terms for this batch: {user_terms}")

        retrieved_reports = retrieve_similar_reports(user_terms, k=3, randomize=True)
        retrieved_reports_str = json.dumps(retrieved_reports, indent=5)

        # Fill template placeholders
        prompt = base_prompt.format(
            user_terms=", ".join(user_terms),
            retrieved_reports=retrieved_reports_str
        )

        # Save each prompt
        prompt_file_path = os.path.join(prompt_log_dir, f"prompt_batch_{batch_start}.txt")
        with open(prompt_file_path, "w") as f:
            f.write(f"User Terms: {', '.join(user_terms)}\n\n")
            f.write(prompt)
        print(f"📝 Saved prompt to {prompt_file_path}")

        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        try:
            out_tokens, _ = generate(
                [tokens],
                model,
                max_tokens=10000,
                temperature=0.6,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
            )
        except RuntimeError as e:
            print(f"Runtime Error: {e}")
            torch.cuda.empty_cache()
            continue

        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        try:
            parsed_res = json.loads(result)
        except json.JSONDecodeError:
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


# -----------------------------
# Run Generator
# -----------------------------
mixtral_generate_data(
    total_samples=50,
    batch_size=1,
    output_file="syn_data.json"
)
