from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import csv

def compare_entire_entries(synthetic_file, real_file, output_csv):
    with open(synthetic_file, "r") as f:
        synthetic_data = json.load(f)
    with open(real_file, "r") as f:
        real_data = json.load(f)

    synthetic_serialized = [json.dumps(entry, sort_keys=True) for entry in synthetic_data]
    real_serialized = [json.dumps(entry, sort_keys=True) for entry in real_data]

    vectorizer = TfidfVectorizer()
    all_entries = synthetic_serialized + real_serialized
    entry_vectors = vectorizer.fit_transform(all_entries)

    synthetic_vectors = entry_vectors[:len(synthetic_serialized)]
    real_vectors = entry_vectors[len(synthetic_serialized):]

    similarity_matrix = cosine_similarity(synthetic_vectors, real_vectors)

    with open(output_csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(["Synthetic Index", "Closest Real Index", "Max Similarity", "Average Similarity"])

        for i, similarities in enumerate(similarity_matrix):
            max_similarity = max(similarities)  # Max similarity with any real entry
            avg_similarity = sum(similarities) / len(similarities)  # Average similarity
            closest_real_idx = similarities.tolist().index(max_similarity)  # Index of closest real entry
            csvwriter.writerow([i, closest_real_idx, max_similarity, avg_similarity])

synthetic_file = "mixtral/outputs/check1.0.json"  # Path to synthetic data JSON file
real_file = "mixtral/dataset/finetune_train.json"  # Path to real data JSON file
output_csv = "synthetic_similarity_results.csv"  # Output CSV file

compare_entire_entries(synthetic_file, real_file, output_csv)
