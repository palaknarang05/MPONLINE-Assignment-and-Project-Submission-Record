# Car Price Prediction System

A machine learning-powered web application that predicts the selling price of a used car based on user inputs such as brand, manufacturing year, kilometers driven, fuel type, seller type, transmission, and ownership history. The application is built using **Flask** and **Scikit-learn** and is deployed on **Render**.

## Live Demo

**Application:**
https://car-price-prediction-model-eqj5.onrender.com

## GitHub Repository

https://github.com/palaknarang05/Car-Price-Prediction-Model

---

## Features

* Predicts used car selling prices using Machine Learning
* Clean and responsive web interface
* Handles both categorical and numerical inputs
* Uses One-Hot Encoding for preprocessing
* Random Forest Regression model
* Flask backend for prediction
* Deployed on Render for public access

---

## Tech Stack

### Frontend

* HTML5
* CSS3

### Backend

* Flask
* Python

### Machine Learning

* Scikit-learn
* Pandas
* NumPy

### Deployment

* Render

---

## Project Structure

```
Car-Price-Prediction-Model/
│
├── app.py
├── train_model.py
├── car_data.csv
├── requirements.txt
├── render.yaml
├── .gitignore
├── README.md
│
└── templates/
    └── index.html
```

---

## Machine Learning Workflow

1. Load the dataset.
2. Clean and preprocess the data.
3. Extract the car brand from the car name.
4. Apply One-Hot Encoding to categorical features.
5. Split the dataset into training and testing sets.
6. Train a Random Forest Regression model.
7. Evaluate the model using the R² Score.
8. Save the trained model using Pickle.
9. Use the saved model for real-time predictions through the Flask application.

---

## Input Features

The application accepts the following inputs:

* Brand
* Manufacturing Year
* Kilometers Driven
* Fuel Type
* Seller Type
* Transmission
* Owner Type

---

## Output

The application predicts the estimated selling price of the selected used car.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/palaknarang05/Car-Price-Prediction-Model.git
cd Car-Price-Prediction-Model
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

This generates:

* model.pkl
* columns.pkl

Run the application:

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## Deployment

The application is deployed on Render.

Live URL:

https://car-price-prediction-model-eqj5.onrender.com

---

## Future Improvements

* Support more car brands and features
* Improve prediction accuracy using advanced regression models
* Add model comparison and evaluation dashboard
* Integrate a database to store prediction history
* Enhance the UI with charts and analytics
* Add user authentication

---

## Author

**Palak Narang**

GitHub: https://github.com/palaknarang05

LinkedIn: https://www.linkedin.com/in/palaknarang05
