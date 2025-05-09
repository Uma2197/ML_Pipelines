"""
Data Loader Script

This script loads the case study dataset from a Parquet file and displays a preview.

Functionality:
- Reads the dataset from a compressed Parquet file.
- Uses the PyArrow engine for efficient data loading.
- Handles exceptions if the file is missing or corrupted.
- Displays the first few rows of the dataset.

Usage:
    Run this script to verify that the dataset loads correctly.

Author: Umamaheswari
"""

import pandas as pd

# Path to your dataset
file_path = 'C:/Users/umadv/PycharmProjects/basf_data_case_study/data/case_study_sample_dataset.gzip'

# Load the parquet file
try:
    df = pd.read_parquet(file_path, engine='pyarrow')
    print("Dataset loaded successfully ✅")
    print(df.head())  # Preview the dataset
except Exception as e:
    print(f"Error loading dataset: {e}")
