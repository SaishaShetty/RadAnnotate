import faiss
import numpy as np
import random
import json
import torch
from sentence_transformers import SentenceTransformer
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from collections import Counter

# ✅ Load Sentence Transformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load Mixtral Tokenizer & Model
tokenizer = MistralTokenizer.from_file("/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model/tokenizer.model.v3")
model = Transformer.from_folder('/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model')

# ✅ Sample Reports for FAISS (Pretend these are real medical reports)
stored_reports = [
    "The aorta appears normal without significant dilation.",
    "The pulmonary arteries are mildly enlarged with no effusion.",
    "Cardiac silhouette is within normal limits.",
    "Heart size is within normal range, no significant abnormalities.",
    "Aortic arch is mildly prominent, likely age-related."
]


report_embeddings = np.array(embedding_model.encode(stored_reports), dtype="float32")

# ✅ Initialize FAISS Index
dimension = report_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  
index.add(report_embeddings)

# ✅ Store Reports for Lookup
stored_reports_dict = {i: stored_reports[i] for i in range(len(stored_reports))}


# ✅ Step 1: Retrieve Similar Reports & Extract Extra Tokens
def retrieve_similar_reports(query_text, k=2):
    """Retrieve similar reports from FAISS and extract additional tokens."""
    query_embedding = embedding_model.encode([query_text])
    _, indices = index.search(np.array(query_embedding), k)

    retrieved_reports = [stored_reports_dict[idx] for idx in indices[0]]

    # Extract extra tokens from retrieved reports
    extra_tokens = extract_tokens_from_reports(retrieved_reports)
    
    return retrieved_reports, extra_tokens


# ✅ Step 2: Extract & Mix Tokens from Retrieved Reports
def extract_tokens_from_reports(retrieved_reports):
    """Extract anatomical and observation tokens from retrieved reports."""
    anatomy_tokens = ["heart", "aorta", "arteries", "trachea", "lungs", "diaphragm"]
    observation_tokens = ["effusion", "enlargement", "collapse", "opacity", "thickening", "edema"]
    
    extracted_tokens = []

    for report in retrieved_reports:
        for word in report.split():
            clean_word = word.strip(",.").lower()
            if clean_word in anatomy_tokens or clean_word in observation_tokens:
                extracted_tokens.append(clean_word)

    # Remove duplicates and mix-match
    extracted_tokens = list(set(extracted_tokens))
    random.shuffle(extracted_tokens)

    return extracted_tokens


# ✅ Step 3: Generate Annotations & Assign Relations
def generate_tokens():
    """Generates a random number of token annotations (7-15) & allows repetitions."""
    anatomy_tokens = ["heart", "aorta", "arteries", "trachea", "lungs", "diaphragm"]
    observation_tokens = ["effusion", "enlargement", "collapse", "opacity", "thickening", "edema"]
    
    num_tokens = random.randint(7, 15)  # 🔥 Now dynamic token count
    selected_anatomy = random.choices(anatomy_tokens, k=random.randint(3, num_tokens // 2))  # ✅ Allow repetitions
    selected_observations = random.choices(observation_tokens, k=num_tokens - len(selected_anatomy))  # ✅ Allow repetitions
    
    annotations = {}
    index = 1
    for token in selected_anatomy:
        annotations[str(index)] = {
            "tokens": token, 
            "label": "ANAT-DP", 
            "start_ix": index * 5, 
            "end_ix": index * 5 + len(token), 
            "relations": []
        }
        index += 1
    for token in selected_observations:
        annotations[str(index)] = {
            "tokens": token, 
            "label": "OBS-DP", 
            "start_ix": index * 5, 
            "end_ix": index * 5 + len(token), 
            "relations": []
        }
        index += 1

    return annotations



def assign_relations(annotations):
    """Ensures logical relationships & allows repeated tokens."""
    obs_tokens = [ann for ann in annotations.values() if ann["label"].startswith("OBS")]
    anat_tokens = [ann for ann in annotations.values() if ann["label"].startswith("ANAT")]

    token_usage_count = Counter()  # ✅ Track token occurrences

    for ann in annotations.values():
        assigned_relation_type = None  # Track relation type to avoid mixing
        
        # ✅ Allow repeated tokens
        token_usage_count[ann["tokens"]] += 1

        # ✅ `located_at (Observation → Anatomy)`
        if ann["label"].startswith("OBS") and anat_tokens and assigned_relation_type is None:
            related_anatomy = random.choice(anat_tokens)
            ann["relations"].append(["located_at", related_anatomy["tokens"]])
            assigned_relation_type = "located_at"

        # ✅ `suggestive_of (Observation → Observation)`
        if ann["label"] == "OBS-DP" and len(obs_tokens) > 1 and assigned_relation_type is None:
            related_observation = random.choice([obs for obs in obs_tokens if obs["tokens"] != ann["tokens"]])
            ann["relations"].append(["suggestive_of", related_observation["tokens"]])
            assigned_relation_type = "suggestive_of"

        # ✅ `modify (Observation → Observation)`
        if ann["label"].startswith("OBS") and assigned_relation_type is None:
            related_obs = random.choice([obs for obs in obs_tokens if obs["tokens"] != ann["tokens"]])
            ann["relations"].append(["modify", related_obs["tokens"]])
            assigned_relation_type = "modify"

        # ✅ `modify (Anatomy → Anatomy)`
        if ann["label"].startswith("ANAT") and assigned_relation_type is None:
            related_anat = random.choice([a for a in anat_tokens if a["tokens"] != ann["tokens"]])
            ann["relations"].append(["modify", related_anat["tokens"]])
            assigned_relation_type = "modify"

    return annotations, token_usage_count



def generate_report_with_mixtral(annotations, token_usage_count, max_retries=3):
    """Generates a structured clinical report ensuring correct token repetition and consistency."""

    all_tokens = list(token_usage_count.keys())

    # Construct a relation mapping
    relation_map = {}
    for ann in annotations.values():
        for relation in ann["relations"]:
            relation_type, target = relation
            if ann["tokens"] not in relation_map:
                relation_map[ann["tokens"]] = []
            relation_map[ann["tokens"]].append(f"{relation_type} {target}")

    attempt = 0

    while attempt < max_retries:
        print(f"🌀 Attempt {attempt + 1} to generate a valid report...")

        # Construct strict instruction prompt
        prompt = f"""
    [System] You are a radiology expert generating structured clinical reports.

    **Your Task:**
    - Generate a structured radiology report.
    - Use the following words when relevant: {', '.join(all_tokens)}
    - Use relations (`located_at`, etc.) **only when logical**.
    
    **Example Report:**
    "The aorta appears mildly thickened with opacity in the trachea. The arteries are normal."

    Now, generate a structured report.
[/System]
"""

        # Encode input for Mixtral
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        # Generate output using Mixtral
        out_tokens, _ = generate(
            [tokens], 
            model, 
            max_tokens=2000,  
            temperature=0.5,  
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
        )

        # Decode output
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        missing_tokens = [t for t in all_tokens if abs(result.lower().count(t.lower()) - token_usage_count[t]) > 1]
        if not missing_tokens:  
            return result  # ✅ Success: Return valid output



        print(f"⚠️ Warning: Incorrect token repetition in report: {missing_tokens}")

        attempt += 1

    print("❌ Failed to generate a valid report after multiple attempts.")
    return "ERROR: Could not generate a valid report."


def generate_synthetic_report():
    """Complete pipeline to generate synthetic reports using FAISS + Mixtral."""

    # 1️⃣ Retrieve similar reports + extract extra tokens
    retrieved_reports, extra_tokens = retrieve_similar_reports("Generate a synthetic clinical report.", k=2)

    annotations = generate_tokens()
    annotations, token_usage_count = assign_relations(annotations)
    report = generate_report_with_mixtral(annotations, token_usage_count)  # ✅ Correct

    output = {
        "Annotations": annotations,
        "Report": report
    }

    return json.dumps(output, indent=4)

synthetic_report = generate_synthetic_report()
print(synthetic_report)
