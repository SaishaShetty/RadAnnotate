import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

if embedding_model is None:
    raise RuntimeError("❌ Sentence Transformer model failed to load!")

# ✅ Get stored embeddings dynamically
stored_texts = ["Example report 1", "Example report 2"]  # Example reports
report_embeddings = np.array(embedding_model.encode(stored_texts), dtype="float32")

# ✅ Extract FAISS dimension dynamically
dimension = report_embeddings.shape[1]

# ✅ Initialize FAISS on CPU
index = faiss.IndexFlatL2(dimension)  # ✅ Uses detected dimension
index.add(report_embeddings)

print(f"✅ FAISS initialized on CPU with dimension: {dimension}")
