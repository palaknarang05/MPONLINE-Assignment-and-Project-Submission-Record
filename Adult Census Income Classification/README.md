# Adult Census Income Classification

Machine Learning project for predicting whether an individual's annual income exceeds **$50K** using the Adult Census Income Dataset. This project covers the complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and comparison of multiple classification algorithms.

## 📌 Objective

Build and compare classification models to predict income category (`>50K` or `<=50K`) based on demographic and employment-related features.

## 📂 Dataset

- **Dataset:** Adult Census Income Dataset
- **Source:** UCI Machine Learning Repository / Kaggle
- **Target Variable:** `income`

The dataset contains attributes such as:

- Age
- Workclass
- Education
- Occupation
- Marital Status
- Relationship
- Race
- Sex
- Hours-per-week
- Native Country
- Capital Gain/Loss
- Income Class

---

## 🛠️ Project Workflow

### 1. Dataset Understanding
- Load dataset
- Explore dimensions and data types
- Analyze target class distribution

### 2. Data Cleaning
- Remove leading/trailing whitespaces
- Replace hidden missing values (`?`)
- Impute missing values using mode
- Remove duplicate records

### 3. Feature Engineering
- Encode target variable
- Apply One-Hot Encoding to categorical features
- Standardize numerical features using StandardScaler

### 4. Model Building

The following classification algorithms are trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### 5. Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The best-performing model is selected based on overall classification performance.

---

## 📚 Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🚀 How to Run

1. Clone the repository.

```bash
git clone <repository-url>
```

2. Install dependencies.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open the Jupyter Notebook.

```bash
jupyter notebook
```

4. Run all cells sequentially.

---

## 📈 Learning Outcomes

- Data preprocessing techniques
- Handling missing values
- Feature encoding
- Feature scaling
- Classification model training
- Model comparison
- Performance evaluation using multiple metrics

---

## 👤 Author

**Palak Narang**

B.Tech Computer Science Engineering  
VIT Bhopal University
