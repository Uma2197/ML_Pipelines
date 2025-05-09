import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_parquet("C:/Users/umadv/PycharmProjects/basf_data_case_study/data/cleaned_dataset.gzip", engine="pyarrow")

# Generate Data Quality Report
report = {}
report['Total Rows'] = len(df)
report['Total Columns'] = len(df.columns)
report['Missing Values'] = df.isnull().sum().to_dict()
report['Duplicate Rows'] = df.duplicated().sum()
report['Data Types'] = df.dtypes.to_dict()
report['Descriptive Statistics'] = df.describe().to_dict()

# Logical consistency checks
negative_values = {col: (df[col] < 0).sum() for col in df.select_dtypes(include=[np.number]).columns}
report['Negative Values'] = negative_values

# Save the report
df_report = pd.DataFrame.from_dict(report, orient='index')
df_report.to_csv("C:/Users/umadv/PycharmProjects/basf_data_case_study/reports/data_quality_report.csv")

print("✅ Data quality report generated and saved at: ../reports/data_quality_report.csv")
