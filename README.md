# 🦢 ViT-Based Image Anomaly Detection on Azure Databricks

End-to-end computer vision anomaly detection pipeline built using:

* Azure Data Lake Storage
* Azure Databricks (Spark + MLflow)
* HuggingFace Vision Transformer (ViT)
* Distributed image augmentation
* Deployment-oriented inference validation

This project demonstrates a production-style ML workflow from raw video frames to deployment-ready anomaly scoring.

---

# 📌 Project Goal

The objective of this project was to:

1. Extract frames from a single swan video
2. Generate synthetic anomaly samples
3. Build a balanced training dataset
4. Fine-tune a Vision Transformer (ViT)
5. Track experiments using MLflow
6. Validate deployment behavior via inference endpoint logic

The focus was not only model performance, but also:

* Cloud-based data engineering
* Secure credential handling
* Deployment readiness
* Reproducible ML workflow design

---

# 🏗 High-Level Architecture

```
Video → Frame Extraction (Local)
           │
           ▼
Synthetic Anomaly Generation
           │
           ▼
Azure Data Lake Storage
           │
           ▼
Databricks Mount
           │
           ▼
Spark ETL + Augmentation
           │
           ▼
ViT Fine-Tuning (MLflow)
           │
           ▼
Notebook-based Inference
           │
           ▼
Anomaly Score (Softmax Probability)
```

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

# 🔹 Local Preprocessing (Video → Dataset Preparation)

This stage runs locally before uploading data to Azure.

## 📁 local_preprocessing/

### frames.py

Extracts image frames from a video source.

Purpose:

* Convert raw video into individual frame images
* Prepare input dataset for anomaly generation

---

### salt_pepper_noise.py

Generates irregular polygon-based anomaly patches.

Features:

* Random irregular polygon noise
* Salt vs pepper distribution control
* Adjustable noise amount
* Random multi-patch generation
* Output saved into `noisy_frames/`

This simulates abnormal samples for supervised training.

---

### label.py

Creates or processes label information.

Purpose:

* Assign normal vs abnormal labels
* Prepare structured dataset for cloud ingestion

---

### llm.py

Optional LLM-assisted labeling support.

Used for:

* Experimental automated labeling
* Metadata generation

---

# 🔹 Cloud Pipeline (Azure Databricks)

All cloud processing is inside:

```
databricks_pipeline/
```

---

## 00_utils.py

Utility functions for:

* Displaying images stored in DBFS
* Rendering binary image content
* Debugging preprocessing results

---

## 01_Ingestion_ETL.py

Data engineering pipeline:

* Load raw images from Azure Data Lake
* Join with labels
* Crop and resize images
* Convert to binary JPEG
* Save processed dataset as Parquet

Ensures structured and reproducible dataset creation.

---

## 02_Augmentation.py

Distributed augmentation using Spark and pandas UDF.

Techniques include:

* Horizontal / vertical flips
* Rotations (90°, 180°, 270°)
* Affine squash & skew
* Polygon-based noise patches

Purpose:

* Balance dataset
* Improve model generalization
* Simulate real-world anomalies

---

## 03_hf_deep_learning.py

Vision Transformer fine-tuning using HuggingFace.

Model:

```
google/vit-base-patch16-224
```

Training setup:

* 224 × 224 RGB input
* Binary classification (normal vs abnormal)
* Early stopping
* MLflow experiment tracking
* Model logging with signature

Tracked metrics:

* F1 score
* Loss
* Confusion matrix
* Validation metrics

---

## 04_model_deployment.py

Deployment preparation logic.

Includes:

* Model loading
* Signature validation
* Deployment configuration scaffolding

Designed for compatibility with:

* Databricks Model Serving
* REST endpoint inference

---

## 05_model_serving.py

Deployment-oriented inference validation.

Features:

* REST endpoint invocation
* Logits extraction
* Softmax computation
* Anomaly score calculation

Anomaly score definition:

```
Anomaly Score = P(class = abnormal)
```

Softmax probabilities are derived from model logits.

This notebook validates production readiness even when managed serving is limited.

---

# 🧠 Model Summary

Architecture: Vision Transformer (ViT)

Input shape:

```
(3, 224, 224)
```

Output:

```
2 logits → softmax → probabilities
```

Training configuration:

* Epochs: 5
* Learning rate: 2e-5
* Weight decay: 0.01
* Early stopping enabled

---

# 📊 Inference Behavior

On normal samples:

* Low anomaly scores
* Stable softmax confidence
* Consistent classification results

This confirms deployment-ready inference logic.

---

# 🔐 Security Design

Sensitive credentials such as:

* Azure SAS tokens
* Databricks personal access tokens

are not hardcoded.

Authentication must be provided via:

* Environment variables
* Databricks Secrets

No secrets are stored in this repository.

---

# 💡 Engineering Highlights

✔ Distributed image processing using Spark
✔ pandas UDF for vectorized image transformation
✔ MLflow experiment tracking
✔ Secure secret handling
✔ Deployment-oriented inference validation
✔ Cloud-native ML pipeline design

---

# 🚀 Deployment Options

This model can be deployed via:

* Databricks Model Serving
* FastAPI REST API
* MLflow model serving
* Container-based deployment

The repository includes endpoint invocation logic for integration testing.

---

# 📌 Future Improvements

* Threshold-based anomaly calibration
* UMAP visualization of feature space
* PatchCore baseline comparison
* CI/CD integration
* Dockerized API deployment

---

# 👩‍💻 Author

Suhwa Seong
M.Sc. Data Science
University of Europe for Applied Sciences

Focus areas:

* Computer Vision
* ML Engineering
* Cloud-based ML Systems
* Deployment and MLOps

---

# 🎯 Project Purpose

This repository demonstrates:

* End-to-end ML engineering capability
* Cloud data processing experience
* Model training and experiment tracking
* Deployment readiness
* Secure credential handling
* Production-style workflow design

---

중에서 하나 고르면 다음 단계로 정리해준다.
