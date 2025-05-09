"""
Model Training

This script trains a RandomForestRegressor model on the processed dataset.
It performs the following steps:
1. Loads the processed dataset.
2. Selects relevant features and target variable.
3. Splits data into training and testing sets (80% train, 20% test).
4. Trains a Random Forest model.
5. Evaluates model performance using MAE and R² Score.
6. Saves the trained model for later use.

Usage:
    Run this script after feature engineering to train and evaluate the model.

Author: Umamaheswari
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the processed dataset
df = pd.read_parquet("C:/Users/umadv/PycharmProjects/basf_data_case_study/data/processed_dataset.gzip", engine="pyarrow")

# Select features and target variable
features = ["voltage", "discharge_capacity", "current", "internal_resistance", "temperature"]
target = "cycle_index"

# Drop rows with NaN values in the selected columns
df = df.dropna(subset=features + [target])

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# ✅ Check if training and testing sets have the same columns
if set(X_train.columns) != set(X_test.columns):
    missing_in_train = set(X_test.columns) - set(X_train.columns)
    missing_in_test = set(X_train.columns) - set(X_test.columns)
    raise ValueError(f"Feature mismatch detected!\nMissing in train: {missing_in_train}\nMissing in test: {missing_in_test}")

# Initialize a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Training Complete!")
print(f"📊 Mean Absolute Error (MAE): {mae:.4f}")
print(f"📈 R² Score: {r2:.4f}")

# Save the trained model
joblib.dump(model, "C:/Users/umadv/PycharmProjects/basf_data_case_study/models/trained_model.pkl")
print("✅ Model saved successfully!")
