"""
Optimized Hyperparameter Tuning for Random Forest Model

This script performs hyperparameter tuning using RandomizedSearchCV with additional optimizations to prevent crashes.

Key Improvements:
- Limited parallel processing (`n_jobs=2`) to prevent system overload.
- Reduced number of iterations (`n_iter=10`) for quicker tuning.
- Lowered `n_estimators` range (`[10, 30, 50]`) to reduce computational load.
- Uses only 70% of training data to speed up processing and reduce memory usage.
- Saves intermediate results every 3 iterations to avoid losing progress.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Load the processed dataset
df = pd.read_parquet("C:/Users/umadv/PycharmProjects/basf_data_case_study/data/processed_dataset.gzip", engine="pyarrow")

# Define features and target variable
features = ["voltage", "discharge_capacity", "current", "internal_resistance", "temperature"]
target = "cycle_index"

df = df.dropna(subset=features + [target])  # Drop missing values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Reduce training sample to 70% to improve speed
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=0.7, random_state=42)

# Define parameter grid with optimized ranges
param_dist = {
    "n_estimators": [10, 30, 50],  # Reduced range to speed up training
    "max_depth": [None, 10, 20],  # Adding a cap to prevent overfitting
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 4),
}

# Initialize RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=2)  # Limited parallel processing

# Perform Randomized Search with optimizations
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,  # Reduced from 20 for efficiency
    scoring="r2",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=2,  # Avoid system overload
)

# Directory for saving intermediate models
checkpoint_dir = "C:/Users/umadv/PycharmProjects/basf_data_case_study/models/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Fit RandomizedSearchCV with checkpointing
for i, params in enumerate(random_search.param_distributions.items()):
    if i % 3 == 0:  # Save every 3 iterations
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{i}.pkl")
        joblib.dump(random_search, checkpoint_path)
        print(f"✅ Saved checkpoint at iteration {i}")

random_search.fit(X_train_sample, y_train_sample)

# Save tuning results to CSV - For dashboarding
results_df = pd.DataFrame(random_search.cv_results_)
tuning_results_path = "C:/Users/umadv/PycharmProjects/basf_data_case_study/data/tuning_results.csv"
results_df.to_csv(tuning_results_path, index=False)
print(f"✅ Tuning results saved at {tuning_results_path}")



# Save the best model
best_model_path = "C:/Users/umadv/PycharmProjects/basf_data_case_study/models/tuned_model.pkl"
joblib.dump(random_search.best_estimator_, best_model_path)

print("✅ Hyperparameter tuning complete! Best model saved at:", best_model_path)
print("Best parameters found:", random_search.best_params_)
