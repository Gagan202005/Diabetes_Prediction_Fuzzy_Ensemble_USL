# Diabetes Prediction using Fuzzy Ensemble Learning

This project implements a machine learning pipeline to predict diabetes risk using an Advanced Fuzzy Ensemble Learning approach. It combines traditional base learners (Random Forest, XGBoost, Logistic Regression) with a Fuzzy Inference System (FIS) to effectively handle uncertainty and model disagreement near decision boundaries.

## 🛠 Software Environment & Prerequisites

- **Python Version**: Python 3.8 to 3.11+ is recommended.
- **Operating System**: macOS, Linux, or Windows. 
  - *Note for macOS users*: The `xgboost` library requires `libomp`. You can install it via Homebrew before installing the python packages: 
    ```bash
    brew install libomp
    ```

### Required Libraries
To ensure compatibility, make sure you install the following library versions (or more recent compatible releases):
- `pandas` (>= 1.3.0)
- `numpy` (>= 1.21.0)
- `scikit-learn` (>= 1.0.2)
- `imbalanced-learn` (>= 0.9.0)
- `xgboost` (>= 1.5.0)
- `scikit-fuzzy` (>= 0.4.2)
- `matplotlib` (>= 3.4.0)
- `seaborn` (>= 0.11.0)
- `joblib` (>= 1.1.0)

You can install all dependencies via pip:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost scikit-fuzzy matplotlib seaborn joblib
```

## 📂 Project Structure

The project is modularized into three sequential Jupyter Notebooks to mimic a professional machine learning pipeline:

1. **`01_Data_Preprocessing.ipynb`**:
   - Loads the raw data (`diabetes_prediction_dataset.csv`).
   - Handles missing values, encodes categorical variables, and scales numerical features using `StandardScaler`.
   - Solves class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
   - Exports the cleaned, separated data to `processed_data.pkl`.

2. **`02_Base_Model_Training.ipynb`**:
   - Loads the preprocessed data.
   - Trains three base classifiers: Logistic Regression, Random Forest, and XGBoost.
   - Extracts their predicted probabilities.
   - Exports the predictions and the test labels to `model_predictions.pkl`.

3. **`03_Fuzzy_Ensemble_Evaluation.ipynb`**:
   - Implements the Fuzzy Inference System using `skfuzzy`.
   - Fuzzifies the outputs of the underlying models into 'low', 'medium', and 'high' certainty bounds.
   - Applies carefully crafted logical rules (e.g., relying heavily on XGBoost when highly confident, but falling back to Random Forest consensus when uncertain).
   - Generates and plots the final comparative evaluations, including ROC AUC curves and Confusion Matrices.

## 🚀 Execution Order & How to Use

To successfully reproduce the project, you **must run the notebooks in the following exact order**. Each step relies on the exported `.pkl` files from the prior notebook.

1. **Step 1**: Open `01_Data_Preprocessing.ipynb` and select **"Run All"**. Wait for it to finish and save `processed_data.pkl`.
2. **Step 2**: Open `02_Base_Model_Training.ipynb` and select **"Run All"**. Wait for it to finish and save `model_predictions.pkl`.
3. **Step 3**: Open `03_Fuzzy_Ensemble_Evaluation.ipynb` and select **"Run All"**. This will generate all the performance metrics and graphs comparing your base models against the Advanced Fuzzy Ensemble.

*Ensure that your Jupyter environment in VS Code (or your browser) is using the Python kernel/Virtual Environment (`.venv`) where all the dependencies above are installed.*