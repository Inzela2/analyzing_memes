# analyzing_memes
Project Overview
This codebase is designed to perform various tasks related to image processing and machine learning, including object detection, text detection and removal, and classification. The project utilizes several advanced libraries such as PyTorch, OpenCV, and Transformers, and is structured to work with Google Colab notebooks for easy access and execution.

Directory Structure
The project is organized into the following directories:

- Colab Notebooks: Contains the Google Colab notebooks for each task.
- hate_meme: A directory containing memes for processing.
- missed: Directory where ~25% of the dataset images (3000 images).
- no_overlay: Contains images without text overlay.
- not_meme: Directory for images that are not memes.
- output_no_overlay: Output directory for images without overlay post-processing.
- overlay: Contains overlay images.
- test_meme: Test directory for meme images.
- test_not_meme: Test directory for images that are not memes.

Setup Instructions

To set up the project, follow these steps:

1. Clone the project to your local machine or Google Colab environment.
2. Install the required dependencies using the provided requirements.txt file in the yolov5 directory or by running the installation commands in each Colab notebook.
3. Mount your Google Drive.

Dependency Libraries

The project makes use of several libraries, including:

- torch, torchvision, torchaudio: For machine learning models.
- yolov5: For object detection tasks.
- opencv-python-headless: For image processing tasks.
- pytesseract: For OCR to detect text in images.
- pandas: For data manipulation and analysis.
- transformers: For zero-shot classification.
- mediapipe: For hand gesture recognition.

Make sure to install the exact versions as specified in the codes to avoid compatibility issues.

How to Run

To run the project, navigate to the specific task directory and execute the Colab notebook. Each task has its own set of instructions provided within the notebook. For example, to run object detection:

1. Navigate to the yolov5 directory.
2. Execute the Colab notebook associated with the YOLOv5 model.
3. The script will download pre-trained models, perform object detection, and save the results.

Approach
The project follows these general approaches for the tasks:

- Object Detection (Task 1): Uses the YOLOv5 model pre-trained on the COCO dataset to detect objects in images.
- Text Detection and Removal (Task 2): Utilizes the EAST text detector to find text in images and OpenCV's inpainting to remove it.
- Image Classification (Task 3): Employs a ResNet model, fine-tuned on the provided dataset, to classify images as memes or not memes.
- Zero-Shot Classification (Bonus Task I): Implements a Transformer model for classifying text without explicit examples of the classes.
- Hand Gesture Recognition (Bonus Task II): Applies MediaPipe to detect hand gestures in images(if middle finger is raised).

Note
The project assumes that the data for processing is located in Google Drive.
