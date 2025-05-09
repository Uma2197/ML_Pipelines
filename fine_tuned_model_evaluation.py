import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_model(model_path):
    """Load the fine-tuned model from the specified path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        object: Loaded model if successful, otherwise exits the script.
    """
    try:
        model = joblib.load(model_path)
        print("✅ Fine-tuned model loaded successfully!")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found at {model_path}. Ensure hyperparameter tuning was completed successfully.")
        exit()


def load_data(file_path):
    """Load the processed dataset from the specified file.

    Args:
        file_path (str): Path to the processed dataset.

    Returns:
        DataFrame: Loaded dataset.
    """
    return pd.read_parquet(file_path, engine="pyarrow")


def preprocess_data(df, features, target):
    """Preprocess the dataset by handling missing values and splitting into test data.

    Args:
        df (DataFrame): The dataset to preprocess.
        features (list): List of feature column names.
        target (str): Target column name.

    Returns:
        tuple: Features (X_test) and target (y_test) test data.
    """
    df = df.dropna(subset=features + [target])
    _, X_test, _, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model using test data and compute performance metrics.

    Args:
        model (object): Trained model for prediction.
        X_test (DataFrame): Feature test data.
        y_test (Series): Target test data.

    Returns:
        dict: Evaluation metrics including MAE, RMSE, and R² Score.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    }

    print("📊 Model Evaluation Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def save_results(results, results_path):
    """Save evaluation results to a text file.

    Args:
        results (dict): Evaluation metrics.
        results_path (str): File path to save results.
    """
    with open(results_path, "w") as f:
        f.write("Fine-Tuned Model Evaluation Results\n")
        f.write("-----------------------------------\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"✅ Evaluation results saved at: {results_path}")


if __name__ == "__main__":
    model_path = "C:/Users/umadv/PycharmProjects/basf_data_case_study/models/tuned_model.pkl"
    data_path = "C:/Users/umadv/PycharmProjects/basf_data_case_study/data/processed_dataset.gzip"
    results_path = "C:/Users/umadv/PycharmProjects/basf_data_case_study/models/evaluation_results.txt"

    model = load_model(model_path)
    df = load_data(data_path)
    features = ["voltage", "discharge_capacity", "current", "internal_resistance", "temperature"]
    target = "cycle_index"
    X_test, y_test = preprocess_data(df, features, target)
    results = evaluate_model(model, X_test, y_test)
    save_results(results, results_path)
