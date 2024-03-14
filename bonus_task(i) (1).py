# -*- coding: utf-8 -*-
"""bonus_task(i).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xQ_FcgSCyYFuhh5fuVqiwQ0jsy0mImUW
"""

!pip install transformers

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sentence = "I beat children"
candidate_labels = ["toxic", "non-toxic"]

result = classifier(sentence, candidate_labels)

print(f"Label: {result['labels'][0]}, Score: {result['scores'][0]}")