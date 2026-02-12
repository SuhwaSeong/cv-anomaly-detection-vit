# Databricks notebook source
import io
from PIL import Image
import matplotlib.pyplot as plt

# 중요한 점: Azure Data Lake에서 파일 경로 구조를 받아올 때, '콜론(:)'이 있으면 에러가 발생한다.
#           이것을 해결하기 위해, 'dbfs:/'를 '/dbfs/'로 변경한다.

def display_image(df, num_images: int = 5):
    images = df.take(num_images)
    plt.figure(figsize=(10, 10))

    for i, row in enumerate(images):
        img_path = row.path.replace("dbfs:/", "/dbfs/")
        img = Image.open(img_path)
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(img)
        plt.title(row.name)
        plt.axis("off")

    plt.show()

def display_image_content(df, num_images: int = 5):
    images = df.take(num_images)
    plt.figure(figsize=(10, 10))

    for i, row in enumerate(images):
        img = Image.open(io.BytesIO(row.image))
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(img)
        plt.title(row.name)
        plt.axis("off")

    plt.show()