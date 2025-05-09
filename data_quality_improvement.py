import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def load_dataset(filepath):
    """Load the dataset from a parquet file."""
    return pd.read_parquet(filepath, engine="pyarrow")

def identify_features(df):
    """Identify numerical and categorical columns."""
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["category", "object"]).columns.tolist()
    return numerical, categorical

def optimize_numerical(df, numerical_features):
    """Convert float64 to float32 for optimization."""
    df[numerical_features] = df[numerical_features].astype("float32")
    return df

def remove_outliers(df, numerical_features):
    """Remove outliers using Z-score method."""
    z_scores = np.abs(zscore(df[numerical_features], nan_policy="omit"))
    return df[(z_scores < 3).all(axis=1)].copy()

def handle_missing_values(df, numerical_features, categorical_features):
    """Handle missing values in numerical and categorical columns."""
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].mean())
    df[categorical_features] = df[categorical_features].fillna(method="ffill")
    return df

def plot_feature_distribution(df_before, df_after, numerical_features):
    """Plot feature distribution before and after cleaning."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    for i, feature in enumerate(numerical_features[:6]):
        sns.histplot(df_before[feature], bins=30, color="red", alpha=0.5, label="Before Cleaning", kde=True, ax=axes[i])
        sns.histplot(df_after[feature], bins=30, color="green", alpha=0.5, label="After Cleaning", kde=True, ax=axes[i])
        axes[i].set_title(f"{feature} Distribution")
        axes[i].legend()
    plt.tight_layout()
    plt.show()

def save_dataset(df, filepath):
    """Save the cleaned dataset to a parquet file."""
    df.to_parquet(filepath, engine="pyarrow", compression="gzip")

def main():
    """Main function to execute the data cleaning pipeline."""
    df = load_dataset("C:/Users/umadv/PycharmProjects/basf_data_case_study/data/processed_dataset.gzip")
    numerical_features, categorical_features = identify_features(df)
    df = optimize_numerical(df, numerical_features)
    df_cleaned = remove_outliers(df, numerical_features)
    df_cleaned = handle_missing_values(df_cleaned, numerical_features, categorical_features)
    plot_feature_distribution(df, df_cleaned, numerical_features)
    save_dataset(df_cleaned, "C:/Users/umadv/PycharmProjects/basf_data_case_study/data/cleaned_dataset.gzip")
    print(f"✅ Data cleaning complete! Cleaned dataset saved with {len(df_cleaned)} rows.")

if __name__ == "__main__":
    main()
