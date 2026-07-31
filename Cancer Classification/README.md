# Brain Cancer Classification using Convolutional Neural Networks (CNN)

## Student Information

**Name:** Palak Narang  
**Registration Number:** 23BCE11819  
**Application Number:** IN26011657  
**Batch:** 1A  
**Department:** Computer Science and Engineering  
**Institution:** VIT Bhopal University

---

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify brain MRI images for cancer detection. The model learns important visual features from MRI scans and predicts the presence of brain tumors, assisting in automated medical image analysis.

---

## Dataset

**Dataset:** Brain MRI Images for Brain Tumor Classification

Source: Kaggle

https://www.kaggle.com/

---

## Technologies and Libraries Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

---

## Methodology

### Data Preparation
- Loaded MRI image dataset.
- Resized all images to a fixed resolution.
- Normalized pixel values.
- Split the dataset into training and testing sets.

### Model Development

The CNN architecture includes:
- Convolution Layers
- Max Pooling Layers
- Flatten Layer
- Dense Hidden Layers
- Output Layer

### Model Training

- Optimizer: Adam
- Loss Function: Binary Crossentropy / Categorical Crossentropy
- Evaluation Metric: Accuracy

### Model Evaluation

- Test Accuracy
- Confusion Matrix
- Classification Report
- Accuracy vs Epoch Graph
- Loss vs Epoch Graph

---

## Results

The CNN successfully learned discriminative MRI features and achieved high classification accuracy. The generated confusion matrix and classification report demonstrate the model's effectiveness in identifying brain tumor images.

---

## Conclusion

This project demonstrates the application of Convolutional Neural Networks for medical image classification. CNNs automatically learn complex visual features from MRI scans, reducing the need for manual feature extraction. Although deep learning models require larger datasets and greater computational resources, they provide reliable performance for medical image analysis and disease detection.

---

## Repository Structure

```text
Brain-Cancer-CNN/
│
├── Palak_Narang_cancer.ipynb
├── README.md
└── .gitignore
```

---

### Developed By

**Palak Narang**
