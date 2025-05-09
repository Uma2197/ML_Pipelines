import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

"""
Feature Importance Analysis Script

This script loads a trained Random Forest model, extracts feature importances, 
visualizes them using a bar plot, and saves the results for further analysis.
"""

# Load the trained model
model_path = "C:/Users/umadv/PycharmProjects/basf_data_case_study/models/tuned_model.pkl"
model = joblib.load(model_path)

# Define feature names
feature_names = ["voltage", "discharge_capacity", "current", "internal_resistance", "temperature"]

# Extract feature importance scores
importances = model.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance - Random Forest Model")
plt.grid(True)
plt.show()

# Save feature importance data
importance_df.to_csv("C:/Users/umadv/PycharmProjects/basf_data_case_study/models/feature_importance.csv", index=False)
print("✅ Feature importance analysis complete! Results saved as 'feature_importance.csv'")
