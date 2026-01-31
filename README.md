<table>
  <tr>
    <td valign="middle">
      <a href="PAPER_LINK_HERE">
        <img src="img/Rad.png" width="42" alt="RadAnnotate logo"/>
      </a>
    </td>
    <td valign="middle">
      <strong>RadAnnotate : Large Language Models for Efficient and
Reliable Radiology Report Annotation</strong>
    </td>
  </tr>
</table>

<p>
  <a href="PAPER_LINK_HERE">
    <img src="https://img.shields.io/badge/Paper-PDF-blue?style=for-the-badge" alt="Paper"/>
  </a>
</p>


RadAnnotate is a research pipeline for **data-efficient and reliable annotation of radiology reports** using large language models. It combines **retrieval-augmented synthetic data generation**, **entity-specific annotation models**, and **confidence-based selective automation** to reduce expert labeling effort in low-resource clinical NLP settings.

---

## Overview

Manual annotation of radiology reports is expensive and slow—especially for rare and ambiguous findings. RadAnnotate is designed to generate clinically grounded synthetic reports, train lightweight entity-specific models, and selectively automate annotations using confidence thresholds.

The pipeline targets **RadGraph-style clinical entity annotation** and is optimized for scenarios with limited labeled data.

---

## Key Goals

- Build a **data-efficient clinical annotation pipeline** using real and synthetic data
- Improve performance for **low-frequency and uncertain clinical entities**
- Enable **safe partial automation** through confidence-aware routing
- Support scalable creation of structured radiology NLP datasets

---

## Annotation Schema

RadAnnotate follows a four-label entity schema:

- **ANAT-DP** — Anatomy, definitely present  
- **OBS-DP** — Observation, definitely present  
- **OBS-DA** — Observation, definitely absent  
- **OBS-U** — Observation, uncertain  

Each labeled span is grounded directly in the report text.

---

## Model Stack

- **Synthetic report generation:** Mixtral-7B-v3  
- **Annotation models:** Qwen-2.5 (entity-specific classifiers)

Each entity type is modeled independently rather than with a single multi-class model.

---

## Pipeline

### 1. Retrieval-augmented synthetic report generation
- Frequent clinical terms are extracted from real reports.
- Semantically similar gold reports are retrieved using embeddings.
- Mixtral generates one-sentence radiology reports together with token-level labels.

### 2. LLM-based quality filtering
A large LLM validates and cleans synthetic annotations by:
- removing invalid and stop-word labels,
- correcting negation and uncertainty errors,
- preventing hallucinated entities,
- enforcing consistency with the report text.

### 3. Entity-specific annotation models
Independent models are trained for:
- ANAT-DP
- OBS-DP
- OBS-DA
- OBS-U

This design reduces the effect of severe class imbalance and improves robustness for rare categories.

---

## Confidence-based selective automation

RadAnnotate performs annotation only when predictions are reliable:

- each entity model produces a confidence score,
- entity-specific confidence thresholds are learned,
- a report is automatically accepted only if its predicted entities meet their thresholds,
- reports containing low-confidence or uncertain predictions (especially OBS-U) are routed to expert review.

This enables practical partial automation while preserving clinical annotation quality.

---

## Intended use

RadAnnotate is intended for:

- radiology report annotation workflows,
- low-resource clinical NLP research,
- construction of structured datasets for downstream medical AI,
- research on trustworthy and selective automation in clinical pipelines.

