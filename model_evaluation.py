"""
Model Evaluation

This script evaluates a trained RandomForestRegressor model on the processed dataset.
It performs the following steps:
1. Loads the processed dataset and trained model.
2. Ensures feature consistency between training and test data.
3. Handles missing values in the test set if required.
4. Makes predictions and calculates evaluation metrics.
5. Visualizes results using scatter and residual plots.
6. Saves evaluation plots for further analysis.

Usage:
    Run this script after model training to assess performance.

Author: Umamaheswari
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load the processed dataset
df = pd.read_parquet("C:/Users/umadv/PycharmProjects/basf_data_case_study/data/processed_dataset.gzip", engine="pyarrow")

# Define feature columns (must match model training!)
features = ["voltage", "discharge_capacity", "current", "internal_resistance", "temperature"]
target = "cycle_index"

# Drop rows with NaN values in these columns
df = df.dropna(subset=features + [target])

# Load the trained model
model = joblib.load("C:/Users/umadv/PycharmProjects/basf_data_case_study/models/trained_model.pkl")

# Split dataset as before
_, X_test, _, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# ✅ Ensure X_test has the same features as seen in training
model_features = model.feature_names_in_  # Features used during training

# Check for missing or extra columns
missing_in_test = set(model_features) - set(X_test.columns)
extra_in_test = set(X_test.columns) - set(model_features)

if missing_in_test or extra_in_test:
    print(f"⚠ Feature mismatch detected!")
    print(f"Missing in test set: {missing_in_test}")
    print(f"Extra in test set: {extra_in_test}")

    # Handle missing features by setting them to zero or using the mean
    for col in missing_in_test:
        X_test[col] = 0  # Alternatively, use X_train[col].mean()

    X_test = X_test[model_features]  # Ensure correct feature order

# Ensure no missing values before making predictions
if X_test.isnull().sum().sum() > 0:
    print("⚠️ Warning: Missing values detected in X_test! Handling them now...")
    X_test = X_test.fillna(X_test.mean())  # Fill missing values with mean
    print("✅ Missing values handled.")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Evaluation Complete!")
print(f"📊 Mean Absolute Error (MAE): {mae:.4f}")
print(f"📈 R² Score: {r2:.4f}")

# ✅ Ensure 'plots' directory exists
plots_dir = "C:/Users/umadv/PycharmProjects/basf_data_case_study/plots"
os.makedirs(plots_dir, exist_ok=True)

# ✅ 1️⃣ Actual vs. Predicted Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Cycle Index")
plt.ylabel("Predicted Cycle Index")
plt.title("Actual vs. Predicted Values")
plt.axline((0, 0), slope=1, color="red", linestyle="--")  # Perfect prediction line
scatter_plot_path = os.path.join(plots_dir, "actual_vs_predicted.png")
plt.savefig(scatter_plot_path)  # Save plot
plt.close()

# ✅ 2️⃣ Residual Distribution Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel("Residuals (Error)")
plt.ylabel("Frequency")
plt.title("Residuals Distribution")
residual_plot_path = os.path.join(plots_dir, "residual_distribution.png")
plt.savefig(residual_plot_path)  # Save plot
plt.close()

print(f"✅ Model Evaluation with Visuals Completed! Plots saved in '{plots_dir}' directory.")


# Create DataFrame with predictions and actuals
results_df = pd.DataFrame({
    "actual_cycle_index": y_test,
    "predicted_cycle_index": y_pred
})
results_df["residuals"] = results_df["actual_cycle_index"] - results_df["predicted_cycle_index"]

# Export to CSV for Power BI
results_df.to_csv("C:/Users/umadv/PycharmProjects/basf_data_case_study/dashboard_exports/model_results.csv", index=False)
print("✅ Predictions CSV exported!")
