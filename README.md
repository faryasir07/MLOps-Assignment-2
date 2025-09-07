# 🌦️ Weather Prediction (MLOps Second Assignment)

This project demonstrates **Git Collaboration** and a **data preprocessing pipeline** for weather prediction using the **Australian Weather Dataset**.
The pipeline includes **data cleaning, train/validation/test splitting, feature engineering, scaling, and one-hot encoding**, making the dataset ready for ML model training.

---

## 📂 Project Structure

```
mlops-second-assignment/
│── weather-dataset/                # Raw dataset (CSV)
│── preprocessing.py                # Preprocessing pipeline script
│── requirements.txt                # Project dependencies
│── README.md                       # Project documentation
│── .gitignore                      # Git ignored files
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

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
pip install -r requirements.txt
```

---

## 📊 Usage

Run the preprocessing pipeline:

```bash
python preprocessing.py
```

Expected output (shapes may vary depending on dataset updates):

```
Train shape: (73814, 75) (73814,)
Validation shape: (8197, 75) (8197,)
Test shape: (1244, 75) (1244,)
```

---

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

* Train ML models (Logistic Regression, Random Forest, XGBoost).
* Evaluate using accuracy, precision, recall, and F1-score.
* Extend pipeline for model persistence and deployment (MLOps best practices).

---
