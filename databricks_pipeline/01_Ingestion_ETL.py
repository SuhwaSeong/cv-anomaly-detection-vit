# Databricks notebook source
mount_path = f"/mnt/cv-anomaly-e2e"
dbutils.fs.ls(mount_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # Process Image

# COMMAND ----------

# DBTITLE 1,Load Image
# Convert the list of FileInfo objects to a spark DataFrame
def create_file_info_df(source_dir:str):
  files = dbutils.fs.ls(f"{mount_path}/{source_dir}")

  file_info_list = [
          {
            "path": file.path,
            "name": file.name,
          }
          for file in files
  ]

  return spark.createDataFrame(file_info_list)

file_info_df = create_file_info_df("images")
display(file_info_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Label Join Cell (Do Not Modify)
# MAGIC
# MAGIC This cell is the **final reference cell** responsible for loading the label CSV
# MAGIC and joining it with the image list in this notebook.
# MAGIC
# MAGIC ### Purpose
# MAGIC - Prevent CSV parsing issues (e.g., the file being incorrectly read as a single column)
# MAGIC - Reliably obtain the `image_name`, `label`, `confidence`, and `notes` columns
# MAGIC - Perform a left join using image files (`name`, `path`) as the reference
# MAGIC - Ensure reproducible and consistent results across executions
# MAGIC
# MAGIC ### Core Principles
# MAGIC - The label CSV must always be **reloaded in this cell**
# MAGIC - Do not modify the `sep`, `quote`, or `escape` options
# MAGIC - If the number of columns is not exactly 4, execution must stop immediately (assert)
# MAGIC - Do not use `inner join` for debugging in this cell
# MAGIC - Always use `left join` in the actual pipeline
# MAGIC
# MAGIC ### Output
# MAGIC - Final result DataFrame: `final_df`
# MAGIC - Columns:
# MAGIC   - `name`
# MAGIC   - `path`
# MAGIC   - `label`
# MAGIC   - `confidence`
# MAGIC   - `notes`
# MAGIC
# MAGIC This cell is **not for debugging**, but a fixed, production-oriented step in the pipeline.
# MAGIC Any changes must be validated in separate experimental cells before being applied here.

# COMMAND ----------

# DBTITLE 1,Join Labels
from pyspark.sql import functions as F

# 1) Find labels.csv reliably
label_files = dbutils.fs.ls(f"{mount_path}/labels")
label_csv_paths = [
    f.path for f in label_files
    if f.path.lower().endswith("labels.csv")
]
if not label_csv_paths:
    raise FileNotFoundError(f"No labels.csv found under {mount_path}/labels")

label_path = label_csv_paths[0]

# 2) Read labels.csv and align join key to file_info_df.name
# labels.csv: image_name = frame_001
label_df = (
    spark.read.csv(label_path, header=True, inferSchema=True)
    .withColumn(
        "name",
        F.concat(F.col("image_name"), F.lit(".jpg"))
    )
    .select("name", "label")
)

# 3) Ensure left DF has no stale label columns
file_info_df2 = file_info_df.drop("label")

# 4) Join directly on 'name'
joined_df = file_info_df2.join(
    label_df,
    on="name",
    how="left"
)

display(joined_df)

# COMMAND ----------

# 중요한 점: Azure Data Lake에서 파일 경로 구조를 받아올 때, '콜론(:)'이 있으면 에러가 발생한다.
#           이것을 해결하기 위해, 'dbfs:/'를 '/dbfs/'로 변경한다.

import os
import matplotlib.pyplot as plt
from PIL import Image

def display_image(df, num_images: int = 5):
    images = df.take(num_images)
    plt.figure(figsize=(10, 10))
    for i, row in enumerate(images):
        img_path.replace('dbfs:/', '/dbfs/')
        img = Image.open(img_path)
        plt.subplot(num_images, 1, i+1)
        plt.imshow(img)
        plt.title(row.name)
        plt.axis('off')
    plt.show()

display_image(file_info_df)

# COMMAND ----------

for row in file_info_df.collect():
    img_path = row.path.replace('dbfs:/', '/dbfs/')
    img = Image.open(img_path)

    width, height = img.size
    print(f"{row.name}: {width} x {height}")

    break

# COMMAND ----------

for row in file_info_df.collect():
    img_path = row.path.replace('dbfs:/', '/dbfs/')
    img = Image.open(img_path)
    width, height = img.size
    new_size = min(width, height)

    # Crop: LL, UL, LR, UR
    img = img.crop(((width-new_size)/2, (height-new_size)/2, (width+new_size)/2, (height+new_size)/2))
    print(f"{img.size = }")

    # Resize: 256 x 256
    img = img.resize((256, 256), Image.NEAREST)
    plt.imshow(img)
    plt.show()
    break

# COMMAND ----------

# DBTITLE 1,Test: Crop and Resize
from PIL import Image
import matplotlib.pyplot as plt
import math

rows = file_info_df.select("path").collect()
n = len(rows)

cols = 3
rows_fig = math.ceil(n / cols)

plt.figure(figsize=(cols * 4, rows_fig * 4))

# 0.0 = fully left, 0.5 = center, 1.0 = fully right
x_bias = 0.40  # move crop window to the left (try 0.2 ~ 0.35)

for i, row in enumerate(rows):
    img_path = row.path.replace("dbfs:/", "/dbfs/")
    img = Image.open(img_path).convert("RGB")

    w, h = img.size
    side = min(w, h)

    # Vertical crop stays centered (fair)
    top = (h - side) // 2
    bottom = top + side

    # Horizontal crop is biased to the left (to keep the beak)
    max_left = w - side
    left = int(max_left * x_bias)
    right = left + side

    img = img.crop((left, top, right, bottom))
    img = img.resize((256, 256), Image.NEAREST)

    plt.subplot(rows_fig, cols, i + 1)
    plt.imshow(img)
    plt.title(f"Image {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# COMMAND ----------

# DBTITLE 1,pandas UDF
import io
import pandas as pd
from PIL import Image
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import BinaryType

@pandas_udf(BinaryType())
def make_image_udf(s: pd.Series) -> pd.Series:
    """
    Input:  dbfs:/... path (string)
    Output: 256x256 JPEG bytes (Databricks previewable)
    Crop:   square crop with a slight left bias (to keep the beak)
            x_bias: 0.0=fully left, 0.5=center, 1.0=fully right
    """
    x_bias = 0.40  # adjust if needed (e.g., 0.35 ~ 0.45)

    def one(path: str) -> bytes:
        # Convert dbfs:/ path to local filesystem path
        img_path = path.replace("dbfs:/", "/dbfs/")
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        side = min(w, h)

        # Vertical crop stays centered (fair)
        top = (h - side) // 2
        bottom = top + side

        # Horizontal crop is biased to the left
        max_left = w - side
        left = int(max_left * x_bias)
        right = left + side

        # Crop + resize
        img = img.crop((left, top, right, bottom))
        img = img.resize((256, 256), Image.NEAREST)

        # Serialize to JPEG bytes
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    return s.apply(one)

# Databricks image preview metadata
image_meta = {"spark.contentAnnotation": '{"mimeType":"image/jpeg"}'}

df_with_image = (
    joined_df
    .withColumn("image", make_image_udf(col("path")))
    .withColumn("image", col("image").alias("image", metadata=image_meta))
)

display(df_with_image)

# COMMAND ----------

import io
import pandas as pd
from PIL import Image
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import BinaryType

@pandas_udf(BinaryType())
def make_image_udf(s: pd.Series) -> pd.Series:
    # same crop logic as your visual check (left bias)
    x_bias = 0.40

    def one(path: str) -> bytes:
        img_path = path.replace("dbfs:/", "/dbfs/")
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        side = min(w, h)

        # vertical center
        top = (h - side) // 2
        bottom = top + side

        # horizontal left-biased crop
        max_left = w - side
        left = int(max_left * x_bias)
        right = left + side

        img = img.crop((left, top, right, bottom))
        img = img.resize((256, 256), resample=Image.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    return s.apply(one)

image_meta = {"spark.contentAnnotation": '{"mimeType":"image/jpeg"}'}

df_with_image = (
    joined_df
    .withColumn("image", make_image_udf(col("path")))
    .withColumn("image", col("image").alias("image", metadata=image_meta))
)

display(df_with_image.select("name", "path", "label", "image"))

# COMMAND ----------

out_path = f"{mount_path}/images_resized"

(df_with_image
 .write.mode("overwrite")
 .format("parquet")
 .save(out_path)
)

# COMMAND ----------

new_df = spark.read.parquet(out_path)
print("count =", new_df.count())
display(new_df.select("name", "path", "label", "image"))

# COMMAND ----------

import io
import pandas as pd
from PIL import Image
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType
from pyspark.sql.functions import pandas_udf

# Databricks image preview metadata (important)
image_meta = {"spark.contentAnnotation": '{"mimeType":"image/jpeg"}'}

@pandas_udf(BinaryType())
def flip_image_horizontally_udf(s: pd.Series) -> pd.Series:
    def flip_one(binary_content: bytes) -> bytes:
        img = Image.open(io.BytesIO(binary_content)).convert("RGB")
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        out = io.BytesIO()
        img.save(out, format="JPEG")
        return out.getvalue()

    return s.apply(flip_one)

df_flipped = (
    df_with_image
    # flip the already-generated JPEG bytes
    .withColumn("image", flip_image_horizontally_udf(F.col("image")))
    # re-attach metadata so Databricks renders it as an image
    .withColumn("image", F.col("image").alias("image", metadata=image_meta))
    # replace ONLY the trailing ".jpg" -> "_flipped.jpg"
    .withColumn("name", F.regexp_replace("name", r"\.jpg$", "_flipped.jpg"))
    .withColumn("path", F.lit("n/a"))
)

display(df_flipped)


# COMMAND ----------

df_flipped.write.mode("append").format("parquet").save(f"{mount_path}/images_resized")

# COMMAND ----------

new_df = spark.read.format("parquet").load(f"{mount_path}/images_resized")
display(new_df)

# COMMAND ----------

from pyspark.sql import functions as F

# Databricks image preview metadata
image_meta = {"spark.contentAnnotation": '{"mimeType":"image/jpeg"}'}

noisy_df = create_file_info_df("noisy_images").withColumn("label", F.lit("noisy"))

resized_noisy_df = (
    noisy_df
    .withColumn("image", make_image_udf(F.col("path")))
    .withColumn("image", F.col("image").alias("image", metadata=image_meta))
)

display(resized_noisy_df.select("name", "path", "label", "image"))

# COMMAND ----------

import io
import pandas as pd
from PIL import Image
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BinaryType

@pandas_udf(BinaryType())
def flip_image_horizontally_udf(s: pd.Series) -> pd.Series:
    def flip_one(binary_content: bytes) -> bytes:
        img = Image.open(io.BytesIO(binary_content)).convert("RGB")
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        out = io.BytesIO()
        img.save(out, format="JPEG")
        return out.getvalue()
    return s.apply(flip_one)

# 1) Make flipped only from non-flipped rows (prevents flip-of-flip when re-running)
flipped_resized_noisy_df = (
    resized_noisy_df
    .filter(~F.col("name").rlike(r"_flipped\.jpg$"))
    .withColumn("image", flip_image_horizontally_udf(F.col("image")))
    .withColumn("image", F.col("image").alias("image", metadata=image_meta))
    .withColumn("name", F.regexp_replace(F.col("name"), r"\.jpg$", "_flipped.jpg"))
)

# 2) Union and drop duplicates by name (idempotent: safe to run multiple times)
final_noisy_df = (
    resized_noisy_df
    .unionByName(flipped_resized_noisy_df)
    .dropDuplicates(["name"])
)

display(final_noisy_df.select("name", "path", "label", "image"))

# COMMAND ----------

from functools import reduce
from pyspark.sql.functions import DataFrame

final_df = reduce(DataFrame.unionAll, [df_with_image, df_flipped, resized_noisy_df, flipped_resized_noisy_df])
display(final_df)

# COMMAND ----------

final_df.write.mode("overwrite").format("parquet").save(f"{mount_path}/images_final")

# COMMAND ----------

(
    final_df.groupBy('label').count()
).display()