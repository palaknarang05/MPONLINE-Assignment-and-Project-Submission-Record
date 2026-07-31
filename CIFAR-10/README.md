# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

## Student Information

**Name:** Palak Narang  
**Registration Number:** 23BCE11819  
**Application Number:** IN26011657  
**Batch:** 1A  
**Department:** Computer Science and Engineering  
**Institution:** VIT Bhopal University

---

## Project Overview

This project develops a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into ten object categories. The workflow includes image preprocessing, CNN model design, training, evaluation, and visualization of model performance using TensorFlow/Keras.

---

## Dataset

**Dataset:** CIFAR-10 Dataset

TensorFlow/Keras Built-in Dataset

https://www.cs.toronto.edu/~kriz/cifar.html

---

## Technologies and Libraries Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Methodology

### Data Preparation
- Loaded the CIFAR-10 dataset.
- Normalized pixel values to the range 0–1.
- Split the dataset into training and testing sets.

### Model Development
The CNN architecture includes:
- Convolution Layers
- Max Pooling Layers
- Flatten Layer
- Fully Connected Dense Layers
- Softmax Output Layer

### Model Training
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy

### Model Evaluation
- Test Accuracy
- Confusion Matrix
- Classification Report
- Accuracy vs Epoch Graph
- Loss vs Epoch Graph

---

## Results

The CNN successfully learned meaningful image features and achieved good classification accuracy on the CIFAR-10 dataset. Performance was evaluated using multiple classification metrics and visualization plots.

---

## Conclusion

This project demonstrates the effectiveness of CNNs for multiclass image classification. Convolution and pooling operations enable automatic feature extraction, reducing the need for manual feature engineering. Although CNNs require considerable computational resources and training time, they provide excellent performance for image recognition tasks.

---

## Repository Structure

```text
CIFAR10-CNN/
│
├── Palak_Narang_CIFAR10.ipynb
├── README.md
└── .gitignore
```

---

### Developed By

**Palak Narang**
