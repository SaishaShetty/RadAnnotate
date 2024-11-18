# RadAnnotate
# Synthetic Clinical Report Generation Pipeline

Generates synthetic clinical reports using the **LLaMA 3.2 (8B)** model with few-shot learning. The primary goal is to produce a large, diverse dataset of realistic clinical reports based on a small seed dataset of 462 examples.

---

## Workflow

1. **Seed Data Preparation**:  
   Select diverse examples from the training data to guide few-shot learning.  
   - Example Prompt:  
     ```
     Instruction: Generate a clinical report in the style of the examples below.

     Example 1:
     - <Report 1>
     Example 2:
     - <Report 2>

     Generate a new report:
     ```

2. **Report Generation**:  
   Use the seed examples and LLaMA 3.2 to generate synthetic clinical reports in batches. Adjust model parameters like temperature and top-p for diversity.

3. **Validation**:  
   - Use metrics like ROUGE-L or cosine similarity to compare generated reports to the seed examples.
   - Perform manual checks for medical coherence and quality.

4. **Automation**:  
   Automate the generation process with scripts to scale up to thousands of synthetic samples.

---

## Goals

- Generate **1,000+ synthetic clinical reports**.
- Ensure outputs are diverse, realistic, and medically coherent.

---

## Repository Structure

