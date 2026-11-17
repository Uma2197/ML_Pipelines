# ML Pipelines

Machine learning workflow in Python, split into clear, reusable steps, plus notebooks on prediction methods, NLP and exploratory data analysis.

## Pipeline scripts

| Step | Script |
|---|---|
| Load & convert data | `data_loader.py`, `csv_conversion.py` |
| Data quality | `data_quality_report.py`, `data_quality_improvement.py` |
| Exploration | `eda.py` |
| Anomaly detection | `anomaly_detection_charge_efficiency.py` |
| Features | `feature_engineering.py`, `feature_importance_analysis.py` |
| Modeling | `model_training.py` (Random Forest regression), `hyperparameter_tuning.py` (randomized search) |
| Evaluation | `model_evaluation.py`, `fine_tuned_model_evaluation.py`, `model_interpretation.py` |
| Tests | `test_model.py` (pytest) |

## Notebooks

- `ML_Pipeline (1).ipynb` – the pipeline end to end
- `Methods_of_prediction.ipynb` – comparison of prediction methods
- `NLP_Pipeline.ipynb` – text processing pipeline
- `Python_EDA.ipynb` – exploratory data analysis

## Setup

```bash
pip install pandas pyarrow numpy scikit-learn matplotlib seaborn scipy pytest joblib
python main.py   # checks that all packages are installed
```

## Tech

Python, Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, Seaborn, pytest, Jupyter
