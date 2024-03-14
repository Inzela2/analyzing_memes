# -*- coding: utf-8 -*-
"""task_1(i).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U5J0chnfiYIZ4wa1RKZ_9N_NpiD87Iwf
"""

# Commented out IPython magic to ensure Python compatibility.
!pip install torch torchvision torchaudio
!pip install -Uq yolov5

!git clone https://github.com/ultralytics/yolov5
# %cd yolov5
!pip install -qr requirements.txt

!sudo apt update
!sudo apt install tesseract-ocr

!pip install pytesseract

!pip install opencv-python-headless==4.8.0.74

import torch
from PIL import Image
import pytesseract
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import numpy as np
import cv2
import os
import csv
import random

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def process_image(image_path, model):
    try:
        img = Image.open(image_path)
        results = model(img, size=640)
        pred = results.pred[0]
        detected_names = set()
        for i in range(len(pred)):
            class_id = int(pred[i, -1])
            if class_id < len(results.names):
                detected_names.add(results.names[class_id])
        text_detected = "text" in pytesseract.image_to_string(img)
        if text_detected:
            detected_names.add('text')
        return os.path.basename(image_path), detected_names
    except Exception as e:
        print(f"Error processing image {os.path.basename(image_path)}: {e}")
        return os.path.basename(image_path), {"Error"}


image_dir = '/content/drive/MyDrive/missed'
all_image_paths = [os.path.join(image_dir, file_name) for file_name in os.listdir(image_dir) if file_name.lower().endswith(('png', 'jpg', 'jpeg'))]


image_paths = random.sample(all_image_paths, 3000)

detection_results = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_image = {executor.submit(process_image, image_path, model): image_path for image_path in image_paths}
    for future in concurrent.futures.as_completed(future_to_image):
        image_name, objects_detected = future.result()
        detection_results[image_name] = objects_detected


output_csv_file = 'detection_results.csv'


with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Number', 'Objects Detected'])
    for image_name, objects in detection_results.items():
        objects_detected = ', '.join(objects) if objects else "No detections"
        writer.writerow([image_name, objects_detected])

print(f"Detection results saved to {output_csv_file}")

import pandas as pd
from collections import Counter

csv_file_path = '/content/yolov5/detection_results.csv'

df = pd.read_csv(csv_file_path)

df_filtered = df[~df['Objects Detected'].isin(['Error', 'No detections'])]

all_objects = [item for sublist in df_filtered['Objects Detected'].str.split(', ') for item in sublist]

object_frequency = Counter(all_objects)

df_frequencies = pd.DataFrame.from_records(list(object_frequency.items()), columns=['Object Detected', 'Frequency'])

output_csv_path = '/content/yolov5/object_frequencies.csv'
df_frequencies.to_csv(output_csv_path, index=False)

print(f"Frequencies saved to {output_csv_path}")