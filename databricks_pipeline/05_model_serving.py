# Databricks notebook source
# MAGIC %md
# MAGIC # Create a Model Serving Endpoint

# COMMAND ----------

# DBTITLE 1,Preprocess and configuration
import os
import requests
import numpy as np
import pandas as pd
import json
from PIL import Image

# Databricks API & Model Endpoint
databricks_instance = 'adb-7405615468372730.10.azuredatabricks.net'
model_name = 'swan-anomaly-inference-wrapped'
model_version = '1'
url = f"https://{databricks_instance}/serving-endpoints/{model_name}/invocations"

databricks_token = os.getenv("DATABRICKS_TOKEN")
if not databricks_token:
    raise ValueError("DATABRICKS_TOKEN is not set")

headers = {
  'Authorization': f'Bearer {databricks_token}',
  'Content-Type': 'application/json'
}

def preprocess_image(image_path):
  image = Image.open(image_path).convert('RGB')

  # (224, 224, 3)
  image = image.resize((224, 224))
  image_array = np.array(image, dtype=np.float32) / 255.0
  # (224, 224, 3) -> (3, 224, 224)
  image_array = image_array.transpose(2, 0, 1)
  # (3, 224, 224) -> (1, 3, 224, 224)
  image_array = np.expand_dims(image_array, axis=0)
  return image_array.astype(np.float32)

def create_tensor_payload(x: np.ndarray):
    return {
        "inputs": {
            "tensor": {
                "shape": list(x.shape),
                "values": x.flatten().tolist()
            }
        }
    }

def score_model(input_tensor: np.ndarray):
    url = f"https://{databricks_instance}/serving-endpoints/{model_name}/invocations"

    # (1,3,224,224) -> nested list
    payload = {"instances": input_tensor.tolist()}

    print("DEBUG POST url:", url)
    resp = requests.post(url, headers=headers, json=payload)

    print("DEBUG status:", resp.status_code)
    print("DEBUG body:", resp.text[:1000])

    if resp.status_code != 200:
        raise Exception(f"Request failed: {resp.status_code}, {resp.text[:1000]}")
    return resp.json()

# Load an image and make predictions
mount_path = "/mnt/cv-anomaly-e2e/"
image_dir = f"{mount_path}images"
files = [f.path.replace("dbfs:/", "/dbfs/") for f in dbutils.fs.ls(image_dir)]
image_path_sample = files[0]

print("sample image:", image_path_sample)

input_data = preprocess_image(image_path_sample)
print("input data shape:", input_data.shape)

response = score_model(input_data)
print("raw response:", response)

# COMMAND ----------

import numpy as np

logits = np.array(response["predictions"][0], dtype=np.float32)

# softmax
exp = np.exp(logits - np.max(logits))
probs = exp / exp.sum()

anomaly_score = float(probs[0])  # class 0이 anomaly라고 가정
print("anomaly_score:", anomaly_score)
print("logits:", logits)
print("probs :", probs)          # [p(class0), p(class1)]
print("pred_class:", int(np.argmax(probs)))
print("confidence:", float(np.max(probs)))

# COMMAND ----------

# DBTITLE 1,Make predictions
response = score_model(input_data)
response

# COMMAND ----------

type(response['predictions'])

# COMMAND ----------

# Class 0 -> abnormal
# Class 1 -> normal

# Softmax
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def post_process(response):
    logits = response['predictions'][0]
    return softmax(np.array(logits))

probabilities = post_process(response)
print('Class logits: ', response['predictions'][0])
print('Class probabilities: ', probabilities)
print("anomaly_score (P(abnormal)): ", float(probabilities[0]))