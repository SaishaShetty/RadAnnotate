## Multi-Model Fine-Tuning and Inference Pipeline

This pipeline trains and evaluates four independent binary classifiers for clinical named entity recognition (NER) on radiology reports.  
Each model corresponds to one entity type from the RadGraph schema, ensuring label-specific learning and interpretability.

---

## Entity-Specific Models

- **ANAT-DP** – Anatomical structures or directions  
  Example: “left", "lung”, “midline”, “heart”  
- **OBS-DP** – Observations or findings confirmed as present  
  Example: “opacity”, “effusion”, “intact”  
- **OBS-DA** – Observations explicitly negated  
  Example: “no consolidation”, “absence of fracture”  
- **OBS-U** – Observations mentioned with uncertainty  
  Example: “possible effusion”, “may represent pneumonia”

---

## Pipeline Stages

### 1. Data Preparation
- The RadGraph gold dataset (425 reports) is used as the base dataset.  
- Each multi-sentence report is split into single sentences to increase the number of samples.  
- Negative samples are added per entity type to balance class frequency:  
  - High-frequency entities (ANAT-DP, OBS-DP) use fewer negatives.  
  - Low-frequency entities (OBS-DA, OBS-U) use proportionally more.  
- Synthetic negatives supplement scarce gold negatives to improve robustness.

### 2. Fine-Tuning
- Each entity type is fine-tuned independently using **Qwen2.5-7B**.  
- Uses **LoRA adapters** and **4-bit NF4 quantization** for efficient training.  
- Data follows an instruction-based format with keys: `instruction`, `input`, and `output`.  
- **Weights & Biases (W&B)** tracks experiments and logs metrics.

### 3. Inference
- Each model uses an entity-specific prompt stored in `prompt.txt`.  
- The inference script loads the prompt dynamically and performs sentence-level prediction.  
- The model output includes:  
  `entity_type`, `entity_value`, `start_position`, `end_position`.  
- Checkpoints are saved periodically during evaluation.

---

## Models Used
- Base model: `Qwen/Qwen2.5-7B`  
- LoRA adapters: one per entity type  
- Quantization: 4-bit NF4  

---

## Outputs

Outputs will be stored in this format : 

- Fine-tuned model weights → `NER_qwen2.5_final/`  
- LoRA adapter weights → `NER_qwen2.5_adapter/`  
- Inference results → `results.json` (includes `report`, `true_labels`, and `model_output`)

---

## Folder Structure
```
finetune/
│
├── train_model.py   # Fine-tuning script
├── inferenece.py    # Inference script
├── prompts/         # Entity-specific prompt

```

