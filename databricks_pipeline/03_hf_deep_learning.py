# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Prepare Dataset #
# MAGIC - Hugginh Face Dataset for Spark DataFrame
# MAGIC # 2. Environment Set Up #
# MAGIC - Dependencies and Device
# MAGIC # 3. Build Deep Learning Model (Transfer Learning) #
# MAGIC - Define Model
# MAGIC # 4. Initialize the Classifier Weights #
# MAGIC # 5. Process Input (Image Processing) #
# MAGIC # 6. Training Set Up #
# MAGIC - Training Parameters
# MAGIC # 7. Evaluation Metrics #
# MAGIC - Define Evaluation Metrics
# MAGIC # 8. Train (MLflow) #
# MAGIC - Define trainer
# MAGIC - Train
# MAGIC - Log metrics (train, evaluate)
# MAGIC - Signature (input, output)
# MAGIC - Log the model with MLflow
# MAGIC - Log the input dataset for lineage tracking

# COMMAND ----------

dbutils.fs.ls("/mnt")

# COMMAND ----------

# Verify schema and row count
spark.table("training_dataset_augmented").printSchema()
print("rows:", spark.table("training_dataset_augmented").count())

# COMMAND ----------

# DBTITLE 1,Prepare Dataset
from pyspark.sql.functions import col

# Base mount path
mount_path = "/mnt/cv_anomaly_e2e/"

# Input datasets
normal_data = f"{mount_path}gold_dataset_normal"
abnormal_data = f"{mount_path}gold_dataset_abnormal"

# Read parquet datasets
df_normal = spark.read.parquet(normal_data)
df_abnormal = spark.read.parquet(abnormal_data)

# Union by column names to avoid column order issues
df_augmented = df_normal.unionByName(df_abnormal, allowMissingColumns=True)

# Force label type to string to avoid Delta schema conflicts
df_augmented = df_augmented.withColumn("label", col("label").cast("string"))

# Target Delta table name
dataset_name = "training_dataset_augmented"

# Drop existing Delta table to remove old/incompatible schema
spark.sql(f"DROP TABLE IF EXISTS {dataset_name}")

# Write as a fresh Delta table
(
    df_augmented
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(dataset_name)
)

# COMMAND ----------

from datasets import Dataset

df = spark.read.table(dataset_name)
dataset = Dataset.from_spark(df)
num_labels = len(dataset.unique("label"))
print(f"{num_labels} labels found in the dataset")
display(dataset)
display(dataset.features)

# COMMAND ----------

# DBTITLE 1,Environment Set Up
import io
import time
import pandas as pd
from PIL import Image
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import mlflow.pytorch

# Add device selection logic
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device {device}")

# COMMAND ----------

# DBTITLE 1,Build Deep Learning
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.utils import logging

class WrappedViT(torch.nn.Module):
    def __init__(self, model):
        super(WrappedViT, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits  # Only return logits (a single tensor)

# Specify a pre-trained model
model_checkpoint = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(
    model_checkpoint,
    use_fast=True,
    do_resize=True,
    size=224
)

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
)

# Wrap the model to ensure it returns only logits
wrapped_model = WrappedViT(model)

# COMMAND ----------

# DBTITLE 1,Initialize Classifier Weights
model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
torch.nn.init.xavier_uniform_(model.classifier.weight)
model.classifier.bias.data.fill_(0)

model = model.to(device)

# COMMAND ----------

# DBTITLE 1,Process Input (Image)
# Define a preprocessing function to handle binary image data

# Images are resized/rescaled to the same resolution (224x224) and normalized across the RGB channels
# with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).

def preprocess(example):
    # Convert binary image data to PIL Image with RGB channel
    image = Image.open(io.BytesIO(example["image"])).convert("RGB")

    # Process the image using the image processor
    processed_image = image_processor(images=image, return_tensors="pt")
    
    # [1, 3, 224, 224] > [3, 224, 224]
    example['pixel_values'] = processed_image['pixel_values'].squeeze()
    example['pixel_values'] = example['pixel_values'].to(device)
    return example

# Apply the preprocessing function to the dataset
dataset = dataset.map(preprocess)

# Set the format of dataset to PyTorch tensors
dataset.set_format(type="torch", columns=["pixel_values", "label"])

#Split the dataset into training and validation sets
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# COMMAND ----------

train_ds_abnormal = sum(x == "0" for x in train_dataset["label"])
train_ds_normal   = sum(x == "1" for x in train_dataset["label"])

eval_ds_abnormal  = sum(x == "0" for x in eval_dataset["label"])
eval_ds_normal    = sum(x == "1" for x in eval_dataset["label"])

print(f"Training dataset: {train_ds_abnormal} abnormal, {train_ds_normal} normal")
print(f"Validation dataset: {eval_ds_abnormal} abnormal, {eval_ds_normal} normal")

# COMMAND ----------

from collections import Counter

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(eval_dataset))

# Convert scalar tensors to Python ints for correct counting
train_labels = [int(x.item()) if hasattr(x, "item") else int(x) for x in train_dataset["label"]]
eval_labels  = [int(x.item()) if hasattr(x, "item") else int(x) for x in eval_dataset["label"]]

train_counts = Counter(train_labels)
eval_counts  = Counter(eval_labels)

print("Train label counts (encoded):", train_counts)
print("Validation label counts (encoded):", eval_counts)

# 1 -> noisy (abnormal), 0 -> swan (normal)
print(
    f"Training_dataset: "
    f"{train_counts.get(1, 0)} abnormal (noisy), "
    f"{train_counts.get(0, 0)} normal (swan)"
)

print(
    f"Validation_dataset: "
    f"{eval_counts.get(1, 0)} abnormal (noisy), "
    f"{eval_counts.get(0, 0)} normal (swan)"
)

# Sanity check
print("Label example:", train_dataset[0]["label"], "type:", type(train_dataset[0]["label"]))

# COMMAND ----------

# DBTITLE 1,Training Set Up
from transformers import TrainingArguments
import os

# Set environment variable to avoid the warning
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    output_dir=f"/tmp/huggingface/{model_name}-finetuned-swan",
    remove_unused_columns=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    gradient_accumulation_steps=1,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    metric_for_best_model="f1"
    # ddp_find_unused_parameters=False,
)

# COMMAND ----------

# DBTITLE 1,Evaluation Metrics
import evaluate

accuracy = evaluate.load('f1')

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)    

# COMMAND ----------

def cast_label(example):
    example["label"] = int(example["label"])
    return example

train_dataset = train_dataset.map(cast_label)
eval_dataset  = eval_dataset.map(cast_label)

print("train labels:", set(train_dataset["label"]))
print("eval labels :", set(eval_dataset["label"]))

# COMMAND ----------

from transformers import AutoModelForImageClassification, Trainer

# 내가 원래 사용한 base checkpoint 이름
base_ckpt = "google/vit-base-patch16-224-in21k"  # 예: 나의 코드에 맞게

fresh_model = AutoModelForImageClassification.from_pretrained(
    base_ckpt,
    num_labels=2
)

fresh_trainer = Trainer(
    model=fresh_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

eval_result = fresh_trainer.evaluate()
print(eval_result)

# COMMAND ----------

# DBTITLE 1,Train (MLflow)
import mlflow
mlflow.end_run()
from transformers import Trainer, EarlyStoppingCallback
from mlflow.models.signature import infer_signature
import time
import torch

# Stat an MLflow run
run_name = f"vit-classification-wrapped-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
with mlflow.start_run(run_name=run_name):
    early_stop = EarlyStoppingCallback(early_stopping_patience=5)
    # Initialize and train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stop]    
    )

    # Train the model
    train_result = trainer.train()

    # Log training metrics
    mlflow.log_metrics(train_result.metrics)

    # Evaluate the log metrics
    eval_result = trainer.evaluate()
    mlflow.log_metrics(eval_result)

    # Get a sample input and prepare it for signature
    sample_input = next(iter(eval_dataset))
    input_tensor = sample_input["pixel_values"].unsqueeze(0)  # [-1, 3, 224, 224]
    
    # Get model prediction for signature
    with torch.no_grad():
       model.eval()
       sample_output = model(input_tensor)

    # Convert to numpy arrays for MLflow
    input_array = input_tensor.cpu().numpy()
    output_array = sample_output.logits.cpu().numpy()

    # Create signature
    signature = infer_signature(input_array, output_array)

    # Log requirement
    reqs = mlflow.transformers.get_default_pip_requirements(model)

    # Log the model with MLflow
    mlflow.pytorch.log_model(
        pytorch_model=wrapped_model,
        artifact_path="model",
        signature=signature,
        pip_requirements=reqs
    )

    # Log the input dataset for lineage tracking from table to model
    src_dataset = mlflow.data.load_delta(table_name = dataset_name)
    mlflow.log_input(src_dataset, context='Training-Input')

# COMMAND ----------

from sklearn.metrics import confusion_matrix
pred = trainer.predict(eval_dataset)
y_true = pred.label_ids
y_pred = pred.predictions.argmax(axis=1)
print(confusion_matrix(y_true, y_pred))

# COMMAND ----------

for exp in mlflow.search_experiments():
    print(exp.name)

# COMMAND ----------

import mlflow

exp = mlflow.get_experiment_by_name(
    "/Users/suhwa.seong@ue-germany.de/03_hf_deep_learning"
)

print("Experiment ID:", exp.experiment_id)

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["start_time DESC"],
    max_results=5
)

display(runs[["run_id", "status", "start_time"]])

# COMMAND ----------

# DBTITLE 1,# Post-Training Probabilistic Separation Validation
import mlflow
import torch
import numpy as np
import io
from PIL import Image
from datasets import Dataset
from pyspark.sql.functions import col

# ----------------------------
# 1) CONFIG
# ----------------------------
dataset_name = "training_dataset_augmented"
model_url = "runs:/a85be662d53c4de0ba630f3215483a0e/model"   # <-- 반드시 실제 run_id로 교체

# ----------------------------
# 2) DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 3) LOAD MODEL
# ----------------------------
model = mlflow.pytorch.load_model(model_url)
model = model.to(device)
model.eval()

print("Model loaded successfully.")

# ----------------------------
# 4) PREPROCESS
# ----------------------------
def preprocess(example):
    image = Image.open(io.BytesIO(example["image"])).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = image_array.transpose(2, 0, 1)
    example["pixel_values"] = torch.tensor(image_array, dtype=torch.float32)
    return example

# ----------------------------
# 5) SCORE FUNCTION
# ----------------------------
def get_batch_scores(label_value, n=50):
    df = (
        spark.read.table(dataset_name)
        .filter(col("label") == label_value)
        .select("image", "label")
        .limit(n)
    )

    ds = Dataset.from_spark(df).map(preprocess)

    x = torch.stack([
        ex["pixel_values"] if isinstance(ex["pixel_values"], torch.Tensor)
        else torch.tensor(ex["pixel_values"], dtype=torch.float32)
        for ex in ds
    ]).to(device)

    with torch.no_grad():
        out = model(x)

    logits = out.logits if hasattr(out, "logits") else out
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

    return probs[:, 0]   # anomaly_score = P(class 0)

# ----------------------------
# 6) RUN STATS
# ----------------------------
normal_scores = get_batch_scores(1, 50)
abnormal_scores = get_batch_scores(0, 50)

print("NORMAL anomaly_score stats")
print("mean :", float(np.mean(normal_scores)))
print("std  :", float(np.std(normal_scores)))
print("min  :", float(np.min(normal_scores)))
print("max  :", float(np.max(normal_scores)))

print("\nABNORMAL anomaly_score stats")
print("mean :", float(np.mean(abnormal_scores)))
print("std  :", float(np.std(abnormal_scores)))
print("min  :", float(np.min(abnormal_scores)))
print("max  :", float(np.max(abnormal_scores)))