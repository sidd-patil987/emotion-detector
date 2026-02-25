# 😊 Emotion Detection System | Python & Machine Learning

A real-time **Emotion Detection System** built using Python, Machine Learning, and Computer Vision.

This project detects human facial emotions such as Happy, Sad, Angry, Surprise, Neutral, etc., using a trained deep learning model and webcam input.

---

## 📌 Project Overview

The objective of this project is to:

- Detect human faces using computer vision
- Classify facial expressions into emotions
- Perform real-time emotion prediction
- Build a practical AI-based application

This project demonstrates the implementation of deep learning for real-world emotion recognition.

---

## 🛠 Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Haar Cascade Classifier
- Convolutional Neural Network (CNN)

---

## 📂 Project Structure

Emotion-Detection/
│
├── model.h5                  # Trained CNN Model
├── emotion_model.json        # Model Architecture
├── haarcascade_frontalface.xml
├── emotion_detector.py       # Main Detection Script
├── dataset/                  # Training Dataset
└── README.md                 # Documentation

---

## 🔄 Project Workflow

### 1️⃣ Data Collection
Use labeled facial emotion dataset.

### 2️⃣ Data Preprocessing
- Resize images
- Convert to grayscale
- Normalize pixel values
- Split into train & test sets

### 3️⃣ Model Training
- Build CNN model
- Train using training dataset
- Validate using test data
- Save trained model (.h5)

### 4️⃣ Face Detection
Use Haar Cascade Classifier to detect faces from webcam.

### 5️⃣ Emotion Prediction
- Extract face region
- Preprocess image
- Pass to trained model
- Display predicted emotion on screen

---

## 📊 Model Architecture (CNN)

- Convolution Layers
- ReLU Activation
- MaxPooling Layers
- Flatten Layer
- Dense Layers
- Softmax Output Layer

---

## 🎯 Emotions Detected

- Happy
- Sad
- Angry
- Surprise
- Fear
- Disgust
- Neutral

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

pip install opencv-python tensorflow keras numpy pandas matplotlib

### 2️⃣ Run the Script

python emotion_detector.py

### 3️⃣ Webcam Opens
The system will detect face and display emotion in real time.

---

## 📈 Features

- Real-time webcam detection
- Deep learning-based classification
- Face detection using Haar Cascade
- CNN-based emotion recognition
- Lightweight and fast processing

---

## 🧠 Skills Demonstrated

- Computer Vision
- Deep Learning
- CNN Model Building
- Real-Time Detection
- Image Processing
- Model Deployment

---

## 🔮 Future Enhancements

- Add GUI Interface
- Improve model accuracy with larger dataset
- Deploy as Web App (Flask)
- Mobile App Integration
- Add Emotion-Based Music Recommendation

---

## 🎯 Resume Description

Emotion Detection System | Python, OpenCV, TensorFlow

Developed a real-time emotion detection system using Convolutional Neural Networks and OpenCV. Implemented facial recognition and emotion classification using deep learning techniques.

---

## 👨‍💻 Author

Siddhesh Patil  
Machine Learning Enthusiast | Python | Data Analytics  

---

⭐ Star this repository if you found it useful!
