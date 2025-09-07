# 🌦️ Weather Prediction (MLOps Second Assignment)

This project demonstrates **Git Collaboration** and a **data preprocessing pipeline** for weather prediction using the **Australian Weather Dataset**.
The pipeline includes **data cleaning, train/validation/test splitting, feature engineering, scaling, and one-hot encoding**, making the dataset ready for ML model training.

---

## 📂 Project Structure
```
mlops-second-assignment/
│── __pycache__/                   # Python cache files (gitignored)
│── weather-dataset/               # Raw dataset (CSV)
│── analysis.ipynb                 # Updated exploratory/data analysis notebook
│── preprocessing.py               # Complete preprocessing pipeline script
│── train_model.py                 # Complete training pipeline script
│── main.py                        # Entry point for running the full pipeline
│── requirements.txt               # Project dependencies
│── README.md                       # Updated project documentation
│── .gitignore                     # Git ignored files (e.g., __pycache__)
```

---

## 🚀 Features

1. **Data Cleaning**

   * Removes rows with missing target values (`RainToday`, `RainTomorrow`).

2. **Data Splitting**

   * Train set: Data before 2015
   * Validation set: Data from 2015
   * Test set: Data after 2015

3. **Preprocessing**

   * **Numeric Columns**: Missing values imputed with mean, scaled between 0-1 using MinMaxScaler.
   * **Categorical Columns**: One-hot encoded with consistent categories across splits.

4. **Reusable Pipeline**

   * Modular functions for each step (`load_and_clean_data`, `split_by_year`, `separate_inputs_targets`, etc.).
   * Final `preprocess_pipeline()` to execute the full process.

5. **Model Training**
   * Logistic Regression with **scikit-learn**.
   * Validation and test evaluation with accuracy, precision, recall, F1-score, and confusion matrix

---


## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone <https://github.com/faryasir07/MLOps-Assignment-2>
pip install -r requirements.txt
```

---

## 📊 Usage

Run ONLY preprocessing pipeline:

```bash
python preprocessing.py
```

Run the complete pipeline (preprocessing + training):

```bash
python main.py
```


Expected output (shapes may vary depending on dataset updates):

```
Train shape: (73814, 75) (73814,)
Validation shape: (8197, 75) (8197,)
Test shape: (1244, 75) (1244,)
```
Sample Model Output:
```
Validation Accuracy : 0.84
Test Accuracy : 0.83
```

## 📦 Requirements

Dependencies are listed in `requirements.txt`:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🔮 Next Steps

* Add hyperparameter tuning (GridSearchCV) for Logistic Regression.
* Train additional ML models (Random Forest, XGBoost, LightGBM).
* Save & load models with joblib/pickle.
* Extend pipeline for model persistence and deployment (MLOps best practices).

---


