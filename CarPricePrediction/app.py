from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and feature columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    year = int(request.form["year"])
    km_driven = int(request.form["km_driven"])
    fuel = request.form["fuel"]
    seller_type = request.form["seller_type"]
    transmission = request.form["transmission"]
    owner = request.form["owner"]
    brand = request.form["brand"]

    # Create DataFrame
    data = pd.DataFrame({
        "year": [year],
        "km_driven": [km_driven],
        "brand": [brand],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner]
    })

    # One-Hot Encoding
    data = pd.get_dummies(
        data,
        columns=[
            "brand",
            "fuel",
            "seller_type",
            "transmission",
            "owner"
        ],
        drop_first=True
    )

    # Add missing columns
    for col in columns:
        if col not in data.columns:
            data[col] = 0

    # Arrange columns in correct order
    data = data[columns]

    # Prediction
    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Car Price: ₹ {prediction:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)