# -*- coding: utf-8 -*-
"""task2(i).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1h3CoreXhfpOKcr1aN3Ffw2vHw_kmO8LU
"""

import os
import torch
from PIL import Image, ImageDraw
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')


path_with_overlay = '/content/drive/MyDrive/overlay'
path_without_overlay = '/content/drive/MyDrive/no_overlay'


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def detect_objects(image_path, model):
    img = Image.open(image_path).convert('RGB')
    results = model(img, size=640)
    return results.pandas().xyxy[0]


# draw bounding boxes on image
def draw_boxes(image, bboxes):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle([(bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax'])], outline="red", width=3)
    return image

# process images from a directory, draw bounding boxes, and save results in a DataFrame
def process_images(directory_path, model, output_dir):
    image_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:10]
    detection_data = []

    for image_name in image_files:
        image_path = os.path.join(directory_path, image_name)
        detections = detect_objects(image_path, model)
        image = Image.open(image_path).convert('RGB')

        bboxes = [{
            'xmin': row['xmin'],
            'ymin': row['ymin'],
            'xmax': row['xmax'],
            'ymax': row['ymax'],
            'object_detected': row['name']
        } for index, row in detections.iterrows()]

        image_with_boxes = draw_boxes(image, bboxes)
        image_with_boxes.save(os.path.join(output_dir, image_name))

        # Store aggregated results for each image
        detection_data.append({
            'image_number': os.path.splitext(image_name)[0],
            'image_name': image_name,
            'detections': bboxes
        })

    return pd.DataFrame(detection_data)


path_with_overlay = '/content/drive/MyDrive/overlay'
path_without_overlay = '/content/drive/MyDrive/no_overlay'
output_dir_with = '/content/drive/MyDrive/overlay'
output_dir_without = '/content/drive/MyDrive/output_no_overlay'


os.makedirs(output_dir_with, exist_ok=True)
os.makedirs(output_dir_without, exist_ok=True)


df_with_overlay = process_images(path_with_overlay, model, output_dir_with)
df_without_overlay = process_images(path_without_overlay, model, output_dir_without)


df_with_overlay.to_csv('/content/drive/MyDrive/overlay_detections_aggregated.csv', index=False)
df_without_overlay.to_csv('/content/drive/MyDrive/no_overlay_detections_aggregated.csv', index=False)