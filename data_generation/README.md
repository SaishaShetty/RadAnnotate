# **Synthetic Data Generation and Validation**

This module of **RadAnnotate** expands a limited set of annotated radiology reports by generating and validating synthetic data using **Mixtral-8x7B-Instruct-v0.1** and **Qwen2.5-32B**.

---

## **Overview**

<p align="center">
  <img src="/Users/saishashetty/Desktop/RadAnnotate/data_generation/img/syn_data.jpg" width="500"/>
</p>

*Figure: RAG-Enhanced Synthetic Data Generation Pipeline*

---

## **1. Sentence-Level Split**
Since the [RadGraph dataset](https://physionet.org/content/radgraph/1.0.0/) contains only 425 clinical reports (chest X-ray modality), each multi-sentence report is split into individual sentences to increase the amount of training data.  
This preprocessing step creates a larger pool of shorter, clinically coherent sentences, forming the base for synthetic generation.
This preprocessing step creates a larger pool of shorter, clinically coherent sentences, forming the base for synthetic generation.

---

## **2. Synthetic Data Generation**
- **Model:** Mixtral-8x7B-Instruct-v0.1  
- **Objective:** Create clinically realistic one-sentence radiology reports with token-level entity annotations.  
- **Process:**
  - Extract frequently occurring anatomy and observation terms from the sentence-level dataset.  
  - Retrieve similar reports using RAG with FAISS and MiniLM embeddings.  
  - Use retrieved examples as few-shot context for Mixtral to generate new annotated reports.  
  - Ensure outputs follow the RadGraph schema and preserve clinical plausibility.  
- **Output:** Synthetic annotated datasets used for downstream model training.

---

## **3. LLM Judge Validation**
- **Model:** Qwen2.5-32B (via Ollama)  
- **Objective:** Validate and refine generated labels to ensure clinical accuracy.  
- **Process:**
  - Prefilter invalid tokens and non-medical terms.  
  - Evaluate each word-label pair in context using structured validation rules.  
  - Remove hallucinated or irrelevant labels while retaining valid anatomical and observation entities.  
  - Produce cleaned, trustworthy annotations for fine-tuning.  

---

## **Key Features**
- Expands limited real data through sentence-level splitting  
- Retrieval-augmented synthetic generation for realistic medical language  
- LLM-based label validation for reliable annotations  
- Schema-aligned, reproducible data pipeline  

---

## **Contributor**
- [Saisha Shetty](https://github.com/SaishaShetty)
