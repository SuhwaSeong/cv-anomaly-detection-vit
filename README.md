```md
# CV Anomaly Detection with Vision Transformer (ViT)

This project implements an end-to-end computer vision anomaly detection pipeline
using a Vision Transformer (ViT), covering data preparation, model training,
and deployment-ready inference on Azure Databricks.

The core objective is to demonstrate how a practical anomaly detection system
can be built starting from minimal real-world data under realistic constraints.

---

## Overview

- Input: a single publicly shared video containing a swan
- Approach: frame extraction + synthetic anomaly generation
- Model: Vision Transformer (ViT)
- Platform: Azure Databricks
- Focus: reproducible pipeline design rather than raw data distribution

This project emphasizes pipeline design, cloud-based execution, and deployment
considerations, rather than maximizing dataset size.

---

## Data Source

The original input was a short video clip containing a single swan.
The video was publicly shared and permitted for general use.

To keep the repository lightweight and code-focused, the raw video and derived
image data are not included. Instead, the data generation process is fully
documented and reproducible through the provided scripts.

---

## Local Preprocessing Pipeline

Local preprocessing was performed on a Windows machine before transferring
the processed data to Azure Databricks.

Key steps:
1. Extract image frames from the original video
2. Generate synthetic anomaly samples using irregular salt-and-pepper noise
3. Prepare labels and metadata for downstream training

Relevant scripts:
- `frames.py`: video-to-frame extraction
- `salt_pepper_noise.py`: synthetic anomaly generation
- `label.py`: labeling logic
- `llm.py`: auxiliary LLM-assisted processing

---

## Azure Databricks Pipeline

Model training and inference were conducted on Azure Databricks to leverage
managed cloud compute and scalable resources.

Key components:
- Data ingestion from prepared image datasets
- ViT-based model training with experiment tracking
- MLflow-backed model loading and inference validation
- Batch inference producing anomaly scores

Due to workspace tier constraints, managed serving endpoints were not used.
Instead, a notebook-based inference pipeline was implemented to validate
deployment-ready logic in a production-like environment.

---

## Deployment and Inference

The trained model was loaded from MLflow and executed in an inference-only
pipeline within Azure Databricks.

The deployment workflow includes:
- Model loading from registry
- Image preprocessing
- Forward pass and logits computation
- Softmax-based anomaly score calculation

This setup mirrors real-world deployment logic and can be extended to REST-based
serving (e.g., FastAPI) in other environments.

---

## Cost and Resource Decisions

To overcome resource limitations typically encountered in free-tier environments,
paid Azure Databricks compute was used for training and inference validation.

This decision reflects real-world trade-offs between cost, scalability, and
experimental flexibility in cloud-based machine learning workflows.

---

## Repository Structure

```

cv-anomaly-detection-vit/
├─ local_preprocessing/
│  ├─ frames.py
│  ├─ label.py
│  ├─ llm.py
│  └─ salt_pepper_noise.py
├─ README.md
└─ .gitignore

```

---

## Key Takeaways

- Demonstrated an end-to-end CV anomaly detection workflow
- Designed a pipeline starting from minimal real-world data
- Applied ViT models in a practical anomaly detection setting
- Validated deployment-ready inference on Azure Databricks
- Made explicit cost and resource decisions in a cloud environment
```

---
