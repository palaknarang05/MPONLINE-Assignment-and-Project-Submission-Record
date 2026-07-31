import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("car_data.csv")

print(df.head())
print(df.info())
print(df.shape)
print(df.columns)

# -----------------------------
# Check Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Extract Brand from Car Name
# -----------------------------
df["brand"] = df["name"].apply(lambda x: x.split()[0])

# Remove name column
df.drop("name", axis=1, inplace=True)

# -----------------------------
# One-Hot Encoding
# -----------------------------
df = pd.get_dummies(
    df,
    columns=[
        "brand",
        "fuel",
        "seller_type",
        "transmission",
        "owner"
    ],
    drop_first=True
)

print("\nPreprocessed Dataset:")
print(df.head())

print("\nDataset Info After Preprocessing:")
print(df.info())

# -----------------------------
# Split Features and Target
# -----------------------------
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Save feature names
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Random Forest Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)

print("\nModel Training Completed Successfully!")
print(f"R² Score: {score:.4f}")

# -----------------------------
# Save Model
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel saved successfully as model.pkl")
print("Feature columns saved as columns.pkl")