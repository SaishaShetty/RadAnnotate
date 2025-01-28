
# **RadAnnotate: Synthetic Clinical Report Annotation Pipeline**

**RadAnnotate** is a two-phase project leveraging the **Mixtral-7B-v3** model to generate synthetic clinical reports and fine-tune it to annotate radiological clinical notes effectively.

## **Phases**

### **Phase 1: Synthetic Data Generation**
- **Objective**: Generate diverse and realistic clinical reports to address limited dataset availability.
- **Key Steps**:
  1. **Token Generation**: Generate anatomy and observation tokens.
  2. **Report Construction**: Build reports using tokens while ensuring medical plausibility.
  3. **Human-in-the-Loop Feedback**: Validate batches of 50 reports to refine prompts.
- **Output**: JSON files with synthetic reports stored in `data_generation/Output/`.

### **Phase 2: Fine-Tuning**
- **Objective**: Fine-tune the model to annotate radiological clinical notes with entity labels and relationships.
- **Key Steps**:
  1. Combine real and synthetic data for training.
  2. Train on schema-specific tasks (e.g., `ANAT-DP`, `OBS-DP`, `located_at` relationships).
  3. Evaluate performance on held-out datasets.
- **Output**: Fine-tuned model saved in `finetune/`.


## **Features**
- **Dynamic Token-Based Pipeline**: Generate tokens and construct reports.
- **Human-in-the-Loop Validation**: Iteratively improve data quality.
- **Schema-Driven Annotation**: Fine-tuned for radiological notes with entities and relationships.
- **Scalable Outputs**: JSON format for structured data storage.

## **Project Structure**
### **Data Generation**
- **`new_prompt.py`**: Generates tokens and reports.
- **`generate_output_first.py`**: Initial direct report generation.
- **Postprocess**:
  - **`reorder.py`**: Adjusts order of report sections and annotations.
- **Output**:
  - JSON files for generated reports.

### **Fine-Tuning**
- **`train.py`**: Fine-tunes Mixtral-7B-v3 for annotation.
- **`inferencing.py`**: Runs inference with the fine-tuned model.

### **Evaluation**
- **`sim_scores.py`**: Measures similarity between real and synthetic data.
- **Utils**:
  - **`length.py`**: Counts reports in JSON files.

## **Goals**
1. **Phase 1**: Generate ~500 synthetic reports with diversity and quality.
2. **Phase 2**: Fine-tune the model to annotate radiological clinical notes effectively.
3. Automate validation and scaling for broader applications.

## **Contributors**
- [Saisha Shetty](https://github.com/SaishaShetty)
