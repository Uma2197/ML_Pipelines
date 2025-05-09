"""
Exploratory Data Analysis (EDA)

This script performs an exploratory analysis of the dataset, including:
- Displaying dataset information, missing values, and statistical summaries.
- Checking unique values for the 'cycle_index' column.
- Visualizing distributions of key features using histograms.
- Filtering out rows where 'internal_resistance' is zero.

Usage:
    Run this script to understand the structure and distribution of data.

Author: Umamaheswari
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_parquet('C:/Users/umadv/PycharmProjects/basf_data_case_study/data/case_study_sample_dataset.gzip', engine='pyarrow')

# Display dataset info
print("Dataset Info:")
print(df.info())
print("\n")

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print("\n")

# Get statistical summary
print("Statistical Summary:")
print(df.describe())
print("\n")

# Check unique cycle indices (helps in plotting later)
print("Unique cycle indices (first 10):")
print(df['cycle_index'].unique()[:10])

# Define columns to visualize
columns_to_plot = ['test_time', 'cycle_index', 'voltage', 'discharge_capacity',
                   'current', 'internal_resistance', 'temperature']

# Set up subplots
plt.figure(figsize=(14, 10))
for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# Filter out rows where internal_resistance is zero
df = df[df["internal_resistance"] != 0]
zero_internal_resistance = df[df["internal_resistance"] == 0]

# Display removed rows (if any)
print("Rows with zero internal resistance:")
print(zero_internal_resistance)

# Uncomment to see new dataset size after filtering
# print(f"Dataset size after removing zero internal resistance rows: {len(df)}")
