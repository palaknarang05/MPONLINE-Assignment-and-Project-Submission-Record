# Face Recognition using Convolutional Neural Networks (CNN)

## Student Information

**Name:** Palak Narang  
**Registration Number:** 23BCE11819  
**Application Number:** IN26011657  
**Batch:** 1A  
**Department:** Computer Science and Engineering  
**Institution:** VIT Bhopal University

---

## Project Overview

This project develops a Convolutional Neural Network (CNN) for face recognition using the Labeled Faces in the Wild (LFW) dataset. The objective is to classify facial images by learning distinctive facial features through deep learning.

---

## Dataset

**Dataset:** Labeled Faces in the Wild (LFW)

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html

---

## Technologies and Libraries Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

---

## Methodology

### Data Preparation
- Loaded the LFW dataset.
- Normalized image pixel values.
- Split the dataset into training and testing sets.

### Model Development

The CNN architecture consists of:
- Convolution Layers
- Max Pooling Layers
- Flatten Layer
- Dense Hidden Layers
- Softmax Output Layer

### Model Training
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Accuracy Metric

### Model Evaluation
- Accuracy
- Classification Report
- Confusion Matrix
- Training Accuracy Graph
- Validation Accuracy Graph

---

## Results

The CNN successfully extracted facial features and classified identities with good accuracy. Performance was assessed using accuracy, confusion matrix, and classification metrics.

---

## Conclusion

Deep learning techniques such as CNNs provide an effective solution for face recognition by automatically learning facial representations from images. While CNNs outperform traditional machine learning approaches, they require sufficient training data and computational resources for optimal performance.

---

## Repository Structure

```text
Face-Recognition-CNN/
│
├── Palak_Narang_LWF.ipynb
├── README.md
└── .gitignore
```

---

### Developed By

**Palak Narang**
