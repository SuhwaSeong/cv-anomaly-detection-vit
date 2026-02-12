# Databricks notebook source
# MAGIC %run /Workspace/Users/suhwa.seong@ue-germany.de/00_utils

# COMMAND ----------

# %run 다음 셀에서
[name for name in globals().keys() if "display" in name]

# COMMAND ----------

# MAGIC %md
# MAGIC # ***Image Augmentation***
# MAGIC * **Open the dataset (`images_final`)**
# MAGIC * **Test and apply anomaly image augmentation techniques**
# MAGIC * **flip, rotation**
# MAGIC * **Salt-and-pepper patches**

# COMMAND ----------

mount_path = "/mnt/cv_anomaly_e2e"
dbutils.fs.ls(mount_path)

# COMMAND ----------

df = spark.read.format('parquet').load(f"{mount_path}/images_final")
df.show()
display(df )

# COMMAND ----------

display(df.groupBy('label').count())

# COMMAND ----------

# DBTITLE 1,TEST: Augment Anomalies
import io
import random
from PIL import Image
from pyspark.sql.functions import pandas_udf, col, regexp_replace, lit
from pyspark.sql.types import BinaryType

@pandas_udf(BinaryType())
def transpose_image_udf(df_series):
    def transpose_image(content):
        """Transpose image and serialize back as jpeg"""
        image = Image.open(io.BytesIO(content)).convert("RGB")

        transpose_types = ['horizontal', 'vertical', 'rotate_90', 'rotate_180', 'rotate_270', 'squash&skew']

        # Randomly select a subset of transpose types to apply
        selected_transpose_types = random.sample(transpose_types, random.randint(1, len(transpose_types)))
        # selected_transpose_types = transpose_types[-1:]  # squash & skew only

        # Get image size directly from the image (FIX)
        width, height = image.size

        # squash & skew matrix
        ss_matrix = (
            1, 0.3, -width * 0.15,
            0.3, 1, -height * 0.15
        )

        # Transpose
        for transpose_type in selected_transpose_types:
            match transpose_type:
                case 'horizontal':
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                case 'vertical':
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                case 'rotate_90':
                    image = image.transpose(Image.ROTATE_90)
                case 'rotate_180':
                    image = image.transpose(Image.ROTATE_180)
                case 'rotate_270':
                    image = image.transpose(Image.ROTATE_270)
                case 'squash&skew':
                    image = image.transform(
                        (width, height),
                        Image.AFFINE,
                        ss_matrix,
                        resample=Image.BILINEAR,
                        fillcolor=(0, 0, 0)
                    )

        # Save back as jpeg
        output = io.BytesIO()
        image.save(output, format='JPEG')
        return output.getvalue()

    return df_series.apply(transpose_image)

# Define the image metadata
image_meta = {
    "spark.contentAnnotation": '{"mimeType":"image/jpeg"}'
}

# Apply the UDF to transpose images
noisy_df_transposed = (
    df.filter(col('label') == 'noisy')
      .withColumn("image", transpose_image_udf(col("image")).alias("image", metadata=image_meta))
      .withColumn("name", regexp_replace(col("name"), r"\.jpg$", "_tr.jpg"))
      .withColumn("path", lit("n/a"))
)

display(df.filter(col('label') == 'noisy'))
display(noisy_df_transposed)

# COMMAND ----------

# DBTITLE 1,Augment Anomalies
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_replace, lit

# Generate multiple df_transposed DataFrames
num_transposed_dfs = 5
transposed_dfs = []
for i in range(num_transposed_dfs):
    df_transposed = (
        df.filter(col('label') == 'noisy')
          .withColumn("image", transpose_image_udf(col("image")).alias("image", metadata=image_meta))
          # IMPORTANT: Make filename unique per iteration
          .withColumn("name", regexp_replace(col("name"), r"\.jpg", f"_tr{i}.jpg"))
          .withColumn("path", lit("n/a"))
        )
    transposed_dfs.append(df_transposed)

# Union all transposed DataFrames
noisy_df_transposed = reduce(DataFrame.union, transposed_dfs)

# Display the resulting DataFrame
display(noisy_df_transposed)

# COMMAND ----------

# DBTITLE 1,salt_and_pepper patches
import io
import numpy as np
from PIL import Image, ImageDraw
from pyspark.sql.functions import pandas_udf, col, regexp_replace, lit
from pyspark.sql.types import BinaryType

@pandas_udf(BinaryType())
def add_salt_and_pepper_patches_udf(df_series):
    def add_salt_and_pepper_patches(content):
        """Adds irregular, polygonal noise patches to the image and serializes back as JPEG"""

        # Base config
        patch_pixels = 500

        image = Image.open(io.BytesIO(content)).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # radius r (area ≈ πr²)
        r = int(np.sqrt(patch_pixels / np.pi))
        r = max(r, 5)

        def make_polygon(center_x, center_y):
            # Generate an irregular polygon
            num_points = np.random.randint(5, 10)
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            angles += np.random.uniform(0, 2 * np.pi / num_points, size=num_points)
            radii = np.random.uniform(0.5 * r, 1.5 * r, size=num_points)

            points = [
                (int(center_x + radius * np.cos(angle)),
                 int(center_y + radius * np.sin(angle)))
                for angle, radius in zip(angles, radii)
            ]
            return points

        def clamp(val, lo, hi):
            return max(lo, min(val, hi))

        # --------------------------------------------------
        # 1) Force ONE patch near the center (must appear)
        # --------------------------------------------------
        cx_min = clamp(int(width * 0.35), r, width - r)
        cx_max = clamp(int(width * 0.55), r + 1, width - r)
        cy_min = clamp(int(height * 0.35), r, height - r)
        cy_max = clamp(int(height * 0.55), r + 1, height - r)

        # If the image is very small, fall back safely
        if cx_max <= cx_min:
            cx_min, cx_max = r, max(r + 1, width - r)
        if cy_max <= cy_min:
            cy_min, cy_max = r, max(r + 1, height - r)

        center_x = np.random.randint(cx_min, cx_max)
        center_y = np.random.randint(cy_min, cy_max)

        noise_value = int(np.random.choice([0, 255]))  # pepper or salt
        draw.polygon(make_polygon(center_x, center_y), fill=(noise_value, noise_value, noise_value))

        # --------------------------------------------------
        # 2) Add RANDOM patches on the edges (optional extras)
        #    Sample centers outside the central region
        # --------------------------------------------------
        extra_patches = np.random.randint(0, 7)  # 0~3 extra edge patches

        for _ in range(extra_patches):
            # Choose whether we place the patch on left/right/top/bottom edge bands
            edge_type = np.random.choice(["left", "right", "top", "bottom"])

            if edge_type == "left":
                ex_min, ex_max = r, max(r + 1, int(width * 0.25))
                ey_min, ey_max = r, height - r
            elif edge_type == "right":
                ex_min, ex_max = min(width - r - 1, int(width * 0.75)), width - r
                ey_min, ey_max = r, height - r
            elif edge_type == "top":
                ex_min, ex_max = r, width - r
                ey_min, ey_max = r, max(r + 1, int(height * 0.25))
            else:  # bottom
                ex_min, ex_max = r, width - r
                ey_min, ey_max = min(height - r - 1, int(height * 0.75)), height - r

            # Safety for tiny images
            if ex_max <= ex_min:
                ex_min, ex_max = r, max(r + 1, width - r)
            if ey_max <= ey_min:
                ey_min, ey_max = r, max(r + 1, height - r)

            edge_x = np.random.randint(ex_min, ex_max)
            edge_y = np.random.randint(ey_min, ey_max)

            noise_value = int(np.random.choice([0, 255]))
            draw.polygon(make_polygon(edge_x, edge_y), fill=(noise_value, noise_value, noise_value))

        # Serialize back to JPEG
        output = io.BytesIO()
        image.save(output, format="JPEG")
        return output.getvalue()

    return df_series.apply(add_salt_and_pepper_patches)

noisy_df_damaged = (
    df.filter(col("label") == "swan")
      .withColumn("image", add_salt_and_pepper_patches_udf(col("image")).alias("image", metadata=image_meta))
      .withColumn("name", regexp_replace(col("name"), r"\.jpg$", "_damaged.jpg"))
      .withColumn("path", lit("n/a"))
)

display(noisy_df_damaged)


# COMMAND ----------

display_image_content(noisy_df_damaged)

# COMMAND ----------

noisy_df_final = noisy_df_transposed.union(noisy_df_damaged)
noisy_df_final.write.mode("overwrite").format("parquet").save(f"{mount_path}/images_noisy_final")
# noisy_df_final.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Nomal data VS Abnormal data = 800:200

# COMMAND ----------

from tqdm import tqdm
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, regexp_replace

# Define available image transpose operations (used inside the UDF)
transpose_types = ["horizontal", "vertical", "rotate_90", "rotate_180", "rotate_270"]

# Count the number of normal (swan) samples
num_swan = df.filter(col("label") == "swan").count()

# Target number of synthetic samples to generate
num_normal = 800

# Number of repetitions needed to approximately reach the target size
num_rep = int(num_normal / num_swan)
print("Number of repetitions:", num_rep)

# Temporary list to store intermediate DataFrames
dfs = []

# Counter for chunk-based writing
cnt_chunk = 0

# Repeat augmentation to scale the dataset
for i in tqdm(range(num_rep), total=num_rep, desc="Processing..."):

    # Apply random transpose augmentation to swan images
    df_swan = (
        df.filter(col("label") == "swan")
          .withColumn("image", transpose_image_udf(col("image")).alias("image", metadata=image_meta))
          .withColumn("name", regexp_replace(col("name"), ".jpg", f'_{i}.jpg'))
    )
    # Accumulate DataFrames in memory
    dfs.append(df_swan)

    # Write to disk every 10 iterations to reduce memory pressure
    if (i % 10 == 0):
        if dfs:
            cnt_chunk += 1
            df_swan_chunk = reduce(DataFrame.unionAll, dfs)
            mode = "overwrite" if cnt_chunk == 1 else "append"
            df_swan_chunk.write.mode(mode).format("parquet").save(f"{mount_path}/images_swan_final")
            dfs = []

# COMMAND ----------

# DBTITLE 1,Normal data를 다룸
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pyspark.sql.functions import (col, lit, regexp_replace)

def transpose_images(i):
    return (
        df.filter(col("label") == "swan")
        .withColumns(
            {
                "image": transpose_image_udf(col("image")).alias("image", metadata=image_meta),
                "name": regexp_replace(col("name"), ".jpg", f"_tr{i}.jpg"),
                "label": lit(1)
            }
        )

    )

dfs = []
count_chunk = 0
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(transpose_images, i) for i in range(num_rep)]

    for i, future in enumerate(tqdm(as_completed(futures), total=num_rep, desc="Processing...")):
        dfs.append(future.result())
        if (i % 10 == 0):
            if dfs:
                count_chunk += 1
                mode = "overwrite" if count_chunk == 1 else "append"
                df_swan_chunk = reduce(DataFrame.unionAll, dfs)
                df_swan_chunk.write.mode(mode).format("parquet").save(f"{mount_path}/gold_dataset_normal")
                dfs = []
    else:
        if dfs:
            df_swan_chunk = reduce(DataFrame.unionAll, dfs)
            df_swan_chunk.write.mode(mode).format("parquet").save(f"{mount_path}/gold_dataset_normal")
            dfs = []

# COMMAND ----------

# DBTITLE 1,Abnormal data를 다룸
num_abnormal = 200
num_noisy = noisy_df_final.count()
num_iter = int(num_abnormal / num_noisy)

print('num of repetition:', num_iter)

def transpose_images_noise(i):
    return (
        noisy_df_final.withColumns(
            {
                "image": transpose_image_udf(col("image")).alias("image", metadata=image_meta),
                "name": regexp_replace(col("name"), ".jpg", f"_tr{i}.jpg"),
                "label": lit(0)
            }
        )

    )

dfs = []
count_chunk = 0
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(transpose_images_noise, i) for i in range(num_iter)]

    for i, future in enumerate(tqdm(as_completed(futures), total=num_iter, desc="Processing...")):
        dfs.append(future.result())
        if (i % 4 == 0):
            if dfs:
                count_chunk += 1
                mode = "overwrite" if count_chunk == 1 else "append"
                df_swan_chunk = reduce(DataFrame.unionAll, dfs)
                df_swan_chunk.write.mode(mode).format("parquet").save(f"{mount_path}/gold_dataset_abnormal")
                dfs = []
    else:
        df_swan_chunk = reduce(DataFrame.unionAll, dfs)
        df_swan_chunk.write.mode("append").format("parquet").save(f"{mount_path}/gold_dataset_abnormal")
        dfs = []

# COMMAND ----------

df_normal = spark.read.parquet(f"{mount_path}/gold_dataset_normal")
df_abnormal = spark.read.parquet(f"{mount_path}/gold_dataset_abnormal")

print("normal count:", df_normal.count())
print("abnormal count:", df_abnormal.count())