# Databricks notebook source
import mlflow
import torch
import io
import numpy as np
from datasets import Dataset
from pyspark.sql.functions import col
from PIL import Image

def preprocess(example):
    """Convert binary image to a tensor of shape (3, 224, 224)"""
    image = Image.open(io.BytesIO(example['image'])).convert('RGB')

    ### AutoImageProcessor ###
    # 1. Resize 
    image = image.resize((224, 224))  # [224,224,3]

    # 2. Convert to NumPy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0

    # 3. Transpose [224,224,3] > [3,224,224]
    image_array = image_array.transpose(2, 0, 1)

    # 4. Store as a PyTorch tensor instead of a list
    example['pixel_values'] = torch.tensor(image_array, dtype=torch.float32)

    return example

dataset_name = 'training_dataset_augmented'
normal_samples = (
    spark.read.table(dataset_name)
    .filter(col('label') == 1)
    .select('image', 'label')
    .limit(5)
)

# Spark DataFrame >> Hugging Face Dataset
normal_samples = Dataset.from_spark(normal_samples)
data = normal_samples.map(preprocess)

# Extract tensors and stack them
pixel_values_list = [torch.tensor(example['pixel_values']) for example in data]
input_tensor = torch.stack(pixel_values_list)

print('First pixel values:', pixel_values_list[0].shape)
print(f'Final input tensor shape: {input_tensor.shape}')  # (5, 3, 224, 224)

# Load model and move it to appropriate device (cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_url = 'runs:/a85be662d53c4de0ba630f3215483a0e/model'
model = mlflow.pytorch.load_model(model_url)
model = model.to(device)
model.eval()

# Move input tensor to device and make prediction
input_tensor = input_tensor.to(device)
with torch.no_grad():
    predictions = model(input_tensor)
    print(predictions)

# COMMAND ----------

import torch

logits = predictions  # (B,2)
probs = torch.softmax(logits, dim=1)

p_class0 = probs[:, 0].detach().cpu()
p_class1 = probs[:, 1].detach().cpu()

pred_class = torch.argmax(probs, dim=1).detach().cpu()

print("P(class0):", p_class0.tolist())
print("P(class1):", p_class1.tolist())
print("pred_class:", pred_class.tolist())

anom_df = (
    spark.read.table(dataset_name)
    .filter(col("label") == 0)
    .select("image", "label")
    .limit(5)
)

anom_ds = Dataset.from_spark(anom_df).map(preprocess)
# list -> Tensor 변환 후 stack
anom_x = torch.stack(
    [torch.tensor(ex["pixel_values"], dtype=torch.float32) for ex in anom_ds]
).to(device)

with torch.no_grad():
    out = model(anom_x)
# out이 (B,2) Tensor일 수도 있고, HuggingFace output일 수도 있음
logits = out.logits if hasattr(out, "logits") else out
probs = torch.softmax(logits, dim=1).detach().cpu()

print("anomaly batch P(class0):", probs[:, 0].tolist())
print("anomaly batch P(class1):", probs[:, 1].tolist())
print("anomaly batch pred_class:", torch.argmax(probs, dim=1).tolist())

# COMMAND ----------

# Print output shape and predictions
print(f"Output predictions: {predictions}")
print("Predictions shape:", predictions.shape)

# COMMAND ----------

import torch.nn.functional as F

# Class 0 -> "abnormal"
# Class 1 -> "normal"

# Apply softmax to get probabilities
probabilities = F.softmax(predictions, dim=1)

print('Probabilities:\n', probabilities)