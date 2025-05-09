"""
Feature Engineering

This script enhances the dataset by:
- Removing rows where 'internal_resistance' is zero.
- Creating new features to improve model performance.
- Saving the processed dataset for further analysis.

Feature Modifications:
1. **Charge Efficiency**: Derived from 'discharge_capacity' and 'current'.
2. **Temperature Category**: Binned temperature values into categorical labels.

Usage:
    Run this script after EDA to generate engineered features.

Author: Umamaheswari
"""

import pandas as pd

# Load cleaned dataset (after EDA)
df = pd.read_parquet('C:/Users/umadv/PycharmProjects/basf_data_case_study/data/case_study_sample_dataset.gzip', engine='pyarrow')

# Remove rows where internal resistance is zero (as identified in EDA)
df = df[df["internal_resistance"] != 0]

# ---------- Feature Engineering Begins Here ---------- #

# Create a new feature - Charge Efficiency
df["charge_efficiency"] = df["discharge_capacity"] / df["current"]

# Binning Temperature into Categories
df["temperature_category"] = pd.cut(df["temperature"],
                                    bins=[24, 28, 32, 36, 42],
                                    labels=["Low", "Moderate", "High", "Very High"])

# Save transformed dataset
df.to_parquet("C:/Users/umadv/PycharmProjects/basf_data_case_study/data/processed_dataset.gzip", compression="gzip", engine="pyarrow")

print("✅ Feature Engineering Completed! Processed dataset saved.")
