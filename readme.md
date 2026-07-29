# ⚡ PokéVision - Pokémon Generation I Image Classifier

A deep learning web application that classifies **Generation I Pokémon (151 Classes)** from uploaded images using a **Custom Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** and deployed with **Streamlit**.

---

## 📌 Project Overview

PokéVision is an image classification system capable of recognizing all **151 Generation I Pokémon** from user-uploaded images.

The project demonstrates the complete deep learning workflow:

- Data preprocessing
- CNN model development
- Model training and evaluation
- Performance visualization
- Model deployment using Streamlit

---

## 🚀 Features

- 🔍 Classifies all **151 Generation I Pokémon**
- 📷 Upload JPG, JPEG or PNG images
- 🎯 Displays predicted Pokémon name
- 📈 Shows prediction confidence
- 🏆 Displays Top-5 predicted classes
- ⚡ Fast inference using a trained CNN model
- 💻 Clean and interactive Streamlit interface

---

## 🧠 Model Architecture

Custom CNN built using TensorFlow/Keras.

Architecture includes:

- Image Rescaling
- Multiple Conv2D Layers
- MaxPooling2D Layers
- Flatten Layer
- Dense Hidden Layer
- Softmax Output Layer (151 Classes)

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras

### Data Processing

- NumPy
- Pillow

### Visualization

- Matplotlib

### Deployment

- Streamlit

---

## 📂 Dataset

**Pokémon Generation One – 20,100 Images**

Dataset Characteristics:

- 151 Pokémon Classes
- Over 20,000 Images
- Images grouped by Pokémon
- Square images
- Center cropped
- High-quality labels
- JPG and PNG formats

---

## 📊 Model Performance

The model achieved approximately:

- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~93%

These results demonstrate strong learning while maintaining good generalization on unseen images.

---

## 📸 Application Preview

The web application allows users to:

1. Upload a Pokémon image
2. Predict the Pokémon species
3. View confidence score
4. Explore the Top-5 predictions

---

## 📁 Project Structure

```
Gen1-Pokemon-Classifier/
│
├── app.py
├── pokemon_classifier.keras
├── class_names.json
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── Pokemon_CNN.ipynb
│
├── dataset/
│
└── images/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Arfa1h/Gen1-Pokemon-Classifier.git
```

Move into the project

```bash
cd Gen1-Pokemon-Classifier
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🎯 Future Improvements

- Support Pokémon from all generations
- Transfer Learning using EfficientNet or ResNet
- Grad-CAM visualization for model interpretability
- Mobile-friendly UI
- Webcam-based real-time prediction
- Pokémon information panel (Type, Stats, Evolution)

---

## 📚 Learning Outcomes

This project helped me gain practical experience in:

- Image Classification
- Convolutional Neural Networks (CNN)
- TensorFlow/Keras
- Data Preprocessing
- Model Evaluation
- Deep Learning Workflows
- Streamlit Deployment
- Git & GitHub Version Control

---

## 👨‍💻 Author

**Mohammed Arfath**

Artificial Intelligence & Machine Learning Student

GitHub: https://github.com/Arfa1h
