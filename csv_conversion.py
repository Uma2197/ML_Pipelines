import pandas as pd

'''# Load the data before anomaly filtering
df_original = pd.read_parquet("C:/Users/umadv/PycharmProjects/basf_data_case_study/data/cleaned_dataset.gzip", engine="pyarrow")

# Export to CSV
df_original.to_csv("C:/Users/umadv/PycharmProjects/basf_data_case_study/dashboard_exports/cleaned_before_filtering.csv", index=False)
df_filtered = pd.read_parquet("C:/Users/umadv/PycharmProjects/basf_data_case_study/data/filtered_dataset.gzip", engine="pyarrow")
df_filtered.to_csv("C:/Users/umadv/PycharmProjects/basf_data_case_study/dashboard_exports/cleaned_after_filtering.csv", index=False)
correlation_matrix = df_filtered.corr(numeric_only=True)
correlation_matrix.to_csv("C:/Users/umadv/PycharmProjects/basf_data_case_study/dashboard_exports/correlation_matrix.csv")'''

# Load the data before anomaly filtering
df_original = pd.read_parquet("C:/Users/umadv/PycharmProjects/basf_data_case_study/data/case_study_sample_dataset.gzip", engine="pyarrow")

# Export to CSV
df_original.to_csv("C:/Users/umadv/PycharmProjects/basf_data_case_study/dashboard_exports/case_study_sample_dataset.csv", index=False)

