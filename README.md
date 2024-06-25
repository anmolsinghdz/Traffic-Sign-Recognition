# Traffic-Sign-Recognition
Welcome to the Traffic Sign Recognition project! This repository contains code and resources for detecting and recognizing traffic signs using the YOLO v4 model, implemented with the Ultralytics YOLOv5 framework on Google Colab.

Table of Contents
Introduction
Features
Google Colab
Dataset
Training
Inference
Results
Contributing
License

Introduction
Traffic sign recognition is a crucial component of autonomous driving systems. This project leverages the power of YOLO v4, a state-of-the-art object detection model, to accurately detect and classify various traffic signs in real-time.

Features
Accurate Detection: Utilizes YOLO v4 for high precision and recall.
Real-time Processing: Optimized for real-time traffic sign detection.
Easy to Use: Simple and intuitive implementation using Ultralytics YOLOv5 framework.
Google Colab
You can run the entire workflow on Google Colab without needing to download dependencies locally. Follow these steps:

Open the Colab Notebook: Traffic Sign Recognition on Colab

Clone the repository:
!git clone https://github.com/yourusername/traffic-sign-recognition.git
%cd traffic-sign-recognition

Install dependencies:
!pip install -r requirements.txt

Download YOLOv4 weights:
!wget -P weights/ https://pjreddie.com/media/files/yolov4.weights

Download and prepare the dataset:
!mkdir -p data/gtsrb
!wget -P data/gtsrb/ http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
!unzip data/gtsrb/GTSRB_Final_Training_Images.zip -d data/gtsrb/

Dataset
For training and evaluation, you can use the German Traffic Sign Recognition Benchmark (GTSRB) dataset. Download the dataset and extract it into the data/ directory.

Training
To train the YOLO v4 model on the traffic sign dataset, run the following command in the Colab notebook:
!python train.py --data data/gtsrb.yaml --cfg cfg/yolov4.cfg --weights weights/yolov4.weights --name yolov4-traffic-sign
This will start the training process and save the trained model weights in the runs/ directory.

Inference
To perform inference using the trained model, use the following command in the Colab notebook:
!python detect.py --weights runs/exp0/weights/best.pt --source data/test_images
This will detect traffic signs in the images located in the data/test_images directory and save the results in runs/detect.

Results
After training and inference, you can evaluate the model's performance using various metrics such as precision, recall, and mAP (mean Average Precision). The results will be logged and can be visualized using tools like TensorBoard.
