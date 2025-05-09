import pytest
import joblib
import numpy as np
import os

# Correct path to the best trained model
MODEL_PATH = "C:/Users/umadv/PycharmProjects/basf_data_case_study/models/final_trained_model.pkl"

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_model_prediction():
    """Test whether the trained model produces valid predictions."""
    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Convert categorical features to numerical (assuming label encoding)
    encoded_cell_index = 0  # If "Cell_A" was encoded as 0 in training

    # Define a sample test input with 10 numerical features
    sample_input = np.array([[834355.0, 1801.83, 25, encoded_cell_index, 3.7, 0.1, 0.002, 28, 0.98, 2]])

    # Ensure the model makes a prediction without error
    prediction = model.predict(sample_input)

    # Check if the prediction is a valid number
    assert isinstance(prediction[0], (float, np.float32, np.float64)), "Prediction is not a number"

    # Optional: Check if prediction falls within an expected range
    assert -1 <= prediction[0] <= 1, "Prediction is outside expected range"
