# 🦢 ViT-Based Image Anomaly Detection on Azure Databricks

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-ViT-yellow)
![Azure](https://img.shields.io/badge/Azure-Cloud-blue)
![Databricks](https://img.shields.io/badge/Databricks-Spark-orange)
![MLflow](https://img.shields.io/badge/MLflow-ExperimentTracking-blue)
![License](https://img.shields.io/badge/Status-Portfolio_Project-lightgrey)

End-to-end computer vision anomaly detection pipeline built using Azure Databricks and Vision Transformer (ViT).

This repository demonstrates a cloud-native ML workflow from raw video frame extraction to deployment-ready anomaly scoring.

---

# 📌 Project Objective

The objective of this project was to design and implement a reproducible anomaly detection system with the following stages:

1. Extract frames from a single swan video
2. Generate synthetic anomaly samples
3. Construct a structured and balanced dataset
4. Fine-tune a Vision Transformer (ViT)
5. Track experiments using MLflow
6. Validate deployment behavior via inference logic

The focus extends beyond model accuracy to:

* Cloud-based data engineering
* Distributed image processing
* Secure credential management
* Deployment readiness
* Reproducible ML system design

---

# 🏗 System Architecture

```mermaid
flowchart LR

    subgraph Local_Environment
        A[Raw Swan Video]
        B[Frame Extraction<br/>frames.py]
        C[Noise Generation<br/>salt_pepper_noise.py]
        D[Labeling<br/>label.py]
    end

    subgraph Azure_Cloud
        E[Azure Data Lake]
        F[Databricks Mount]
        G[Spark ETL<br/>01_Ingestion_ETL.py]
        H[Distributed Augmentation<br/>02_Augmentation.py]
        I[ViT Fine-Tuning<br/>03_hf_deep_learning.py]
        J[MLflow Tracking]
        K[Inference Validation<br/>05_model_serving.py]
        L[Softmax]
        M[Anomaly Score]
    end

    A --> B --> C --> D --> E
    E --> F --> G --> H --> I
    I --> J
    I --> K
    K --> L --> M
```

---

# 🧰 Technology Stack

| Category            | Tools                             |
| ------------------- | --------------------------------- |
| Programming         | Python                            |
| Deep Learning       | PyTorch, HuggingFace Transformers |
| Model               | Vision Transformer (ViT)          |
| Data Processing     | Apache Spark, pandas UDF          |
| Cloud Storage       | Azure Data Lake                   |
| Cloud Compute       | Azure Databricks                  |
| Experiment Tracking | MLflow                            |
| Deployment Logic    | REST Endpoint Validation          |

---

# 📂 Repository Structure

```
cv-anomaly-detection-vit/
│
├── databricks_pipeline/
│   ├── 00_utils.py
│   ├── 01_Ingestion_ETL.py
│   ├── 02_Augmentation.py
│   ├── 03_hf_deep_learning.py
│   ├── 04_model_deployment.py
│   └── 05_model_serving.py
│
├── local_preprocessing/
│   ├── frames.py
│   ├── label.py
│   ├── llm.py
│   └── salt_pepper_noise.py
│
├── .gitignore
└── README.md
```

---

# 🔹 Local Preprocessing

## frames.py

Extracts individual frames from a video source.

## salt_pepper_noise.py

Implements irregular polygon-based anomaly synthesis with:

* Adjustable noise ratio
* Salt vs pepper control
* Multi-patch random generation

## label.py

Creates structured labels for normal vs abnormal classes.

## llm.py

Optional experimental LLM-assisted labeling logic.

---

# 🔹 Cloud Pipeline (Azure Databricks)

## 01_Ingestion_ETL.py

Spark-based ingestion pipeline:

* Load images from Azure Data Lake
* Join label metadata
* Crop and resize
* Convert to binary JPEG
* Save structured Parquet dataset

## 02_Augmentation.py

Distributed augmentation via pandas UDF:

* Flips
* Rotations
* Affine transforms
* Polygon-based anomaly patches

## 03_hf_deep_learning.py

Vision Transformer fine-tuning.

Model:

```
google/vit-base-patch16-224
```

Training:

* Input: 224 × 224 RGB
* Binary classification
* Early stopping
* MLflow tracking
* Model artifact logging

## 05_model_serving.py

Deployment-oriented inference validation:

* REST endpoint invocation
* Logits extraction
* Softmax computation
* Anomaly score calculation

Anomaly Score:

```
P(class = abnormal)
```

---

# 📊 Results

### Model Configuration

* Architecture: Vision Transformer (ViT)
* Epochs: 5
* Learning rate: 2e-5
* Weight decay: 0.01
* Early stopping enabled

### Inference Behavior (Normal Samples)

* Average anomaly score ≈ 0.00024
* Maximum anomaly score < 0.01
* Stable probability distribution

The model consistently assigns low anomaly probabilities to normal images, indicating stable inference behavior under deployment-like conditions.

---

# 🔐 Security Considerations

* No Azure SAS tokens stored in repository
* No Databricks personal access tokens hardcoded
* Authentication handled via environment variables or Databricks Secrets

This repository follows secure credential management practices.

---

# 🚀 Deployment Compatibility

The trained model is compatible with:

* Databricks Model Serving
* FastAPI REST API
* MLflow model serving
* Containerized deployment

Inference logic included in this repository validates production readiness.

---

# 💡 Engineering Highlights

* Distributed image processing with Spark
* Vectorized image transformation via pandas UDF
* MLflow experiment lifecycle tracking
* Cloud-native ML workflow
* Secure secret management
* Deployment-oriented validation

---

# 👩‍💻 Author

Suhwa Seong
M.Sc. Data Science
University of Europe for Applied Sciences

Focus areas:

* Computer Vision
* ML Engineering
* Cloud-based ML Systems
* Deployment & MLOps

---

# 🎯 Purpose of This Repository

This project demonstrates:

* End-to-end ML engineering capability
* Cloud-based data processing
* Experiment tracking and model lifecycle management
* Deployment readiness
* Secure and reproducible ML workflow design

---
