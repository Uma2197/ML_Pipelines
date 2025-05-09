"""
Model Interpretation

This script interprets the trained RandomForestRegressor model by analyzing feature importance.
It performs the following steps:
1. Loads the trained model.
2. Extracts feature importance scores.
3. Visualizes feature importance using a bar chart.
4. Saves the feature importance plot for further analysis.

Usage:
    Run this script after training the model to understand which features contribute most to predictions.

Author: Umamaheswari
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the trained model
model = joblib.load("C:/Users/umadv/PycharmProjects/basf_data_case_study/models/trained_model.pkl")

# Define feature names (same as those used in training)
features = ["voltage", "discharge_capacity", "current", "internal_resistance", "temperature"]

# Extract feature importance
feature_importance = model.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# ✅ Ensure 'plots' directory exists
plots_dir = "C:/Users/umadv/PycharmProjects/basf_data_case_study/plots"
os.makedirs(plots_dir, exist_ok=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest Model")
plt.gca().invert_yaxis()  # Highest importance at the top

# Save and show the plot
feature_importance_path = os.path.join(plots_dir, "feature_importance.png")
plt.savefig(feature_importance_path)  # Save plot
plt.show()

print(f"✅ Feature Importance Analysis Completed! Plot saved at {feature_importance_path}.")
