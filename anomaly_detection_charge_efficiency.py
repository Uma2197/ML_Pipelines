import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def filter_anomalies(input_path: str, output_path: str) -> None:
    """Loads a dataset, detects anomalies in charge efficiency, filters them out, and saves the cleaned dataset.

    Args:
        input_path (str): Path to the input cleaned dataset.
        output_path (str): Path to save the filtered dataset.
    """
    # Load the cleaned dataset
    df = pd.read_parquet(input_path, engine="pyarrow")

    # Compute mean and standard deviation
    mean_ce = df["charge_efficiency"].mean()
    std_ce = df["charge_efficiency"].std()
    print(f"Mean: {mean_ce:.4f}, Standard Deviation: {std_ce:.4f}")

    # Compute Z-scores and identify outliers
    df["charge_efficiency_z"] = np.abs((df["charge_efficiency"] - mean_ce) / std_ce)
    outliers = df[df["charge_efficiency_z"] > 3]
    anomaly_percentage = (len(outliers) / len(df)) * 100
    print(f"Number of anomalies: {len(outliers)} ({anomaly_percentage:.2f}%)")

    # Remove anomalies and save the filtered dataset
    df_filtered = df[df["charge_efficiency_z"] <= 3].drop(columns=["charge_efficiency_z"])
    df_filtered.to_parquet(output_path, engine="pyarrow", compression="gzip")
    print(f"✅ Filtered dataset saved with {len(df_filtered)} rows.")


def visualize_anomaly_handling(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Plots the charge efficiency distribution before and after anomaly handling.

    Args:
        original_df (pd.DataFrame): Original dataset before filtering anomalies.
        filtered_df (pd.DataFrame): Dataset after removing anomalies.
    """
    plt.figure(figsize=(10, 5))

    sns.histplot(filtered_df["charge_efficiency"], bins=100, color="blue", alpha=0.5, label="After Handling Anomalies",
                 kde=True)
    sns.histplot(original_df["charge_efficiency"], bins=100, color="red", alpha=0.3, label="Before Handling Anomalies",
                 kde=True)

    plt.axvline(-3.9858, color='black', linestyle='dashed', label='Mean')
    plt.axvline(-3.9858 + 3 * 7.9260, color='green', linestyle='dashed', label='Upper 3σ')
    plt.axvline(-3.9858 - 3 * 7.9260, color='purple', linestyle='dashed', label='Lower 3σ')

    plt.title("Charge Efficiency Distribution Before and After Anomaly Handling")
    plt.xlabel("Charge Efficiency")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# Example usage
input_file = "C:/Users/umadv/PycharmProjects/basf_data_case_study/data/cleaned_dataset.gzip"
output_file = "C:/Users/umadv/PycharmProjects/basf_data_case_study/data/filtered_dataset.gzip"
filter_anomalies(input_file, output_file)

# Load filtered dataset for visualization
df_original = pd.read_parquet(input_file, engine="pyarrow")
df_filtered = pd.read_parquet(output_file, engine="pyarrow")
visualize_anomaly_handling(df_original, df_filtered)
