import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# 1. Load & clean data
def load_and_clean_data(path):
    """
    Load dataset and remove rows with missing target values.
    """
    df = pd.read_csv(path)
    # Drop rows where RainToday or RainTomorrow is missing
    df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
    return df


# 2. Split dataset by year into train, validation, and test
def split_by_year(df):
    """
    Split the dataset into Train (<2015), Validation (=2015), and Test (>2015).
    """
    year = pd.to_datetime(df.Date).dt.year
    train_df = df[year < 2015]
    val_df = df[year == 2015]
    test_df = df[year > 2015]
    return train_df, val_df, test_df


# 3. Separate inputs (features) & targets (labels)
def separate_inputs_targets(df, target_col='RainTomorrow'):
    """
    Separate features and target column from dataset.
    Ignores first column (Date) and last column (target).
    """
    input_cols = list(df.columns)[1:-1]  # exclude Date and target
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets


# 4. Preprocess numeric columns (impute + normalize)
def preprocess_numeric(train_inputs, val_inputs, test_inputs, df):
    """
    Impute missing numeric values with mean and scale features between 0-1.
    """
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()

    # Fit imputer & scaler on full dataset numeric columns
    imputer = SimpleImputer(strategy='mean').fit(df[numeric_cols])
    scaler = MinMaxScaler().fit(df[numeric_cols])

    # Apply imputation + scaling on each split
    for dataset in [train_inputs, val_inputs, test_inputs]:
        dataset[numeric_cols] = imputer.transform(dataset[numeric_cols])
        dataset[numeric_cols] = scaler.transform(dataset[numeric_cols])
    
    return numeric_cols


# 5. Preprocess categorical columns (One-Hot Encoding)
def preprocess_categorical(train_inputs, val_inputs, test_inputs, df):
    """
    One-Hot Encode categorical features, ensuring consistent encoding across splits.
    """
    categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

    # Fit encoder on full dataset categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    def encode_and_concat(inputs):
        """
        Apply encoding and concatenate with original inputs.
        """
        transformed = pd.DataFrame(
            encoder.transform(inputs[categorical_cols]),
            columns=encoded_cols,
            index=inputs.index
        )
        # Drop original categorical cols and add encoded ones
        return pd.concat([inputs.drop(categorical_cols, axis=1), transformed], axis=1)

    # Encode train, validation, and test splits
    train_inputs = encode_and_concat(train_inputs)
    val_inputs = encode_and_concat(val_inputs)
    test_inputs = encode_and_concat(test_inputs)

    return train_inputs, val_inputs, test_inputs, encoded_cols


# 6. Full preprocessing pipeline
def preprocess_pipeline(path):
    """
    Execute full preprocessing pipeline:
    - Load & clean data
    - Split by year
    - Separate features & target
    - Process numeric & categorical features
    """
    raw_df = load_and_clean_data(path)
    train_df, val_df, test_df = split_by_year(raw_df)

    train_inputs, train_targets = separate_inputs_targets(train_df)
    val_inputs, val_targets = separate_inputs_targets(val_df)
    test_inputs, test_targets = separate_inputs_targets(test_df)

    preprocess_numeric(train_inputs, val_inputs, test_inputs, raw_df)
    train_inputs, val_inputs, test_inputs, encoded_cols = preprocess_categorical(
        train_inputs, val_inputs, test_inputs, raw_df
    )

    return (train_inputs, train_targets,
            val_inputs, val_targets,
            test_inputs, test_targets)


# Run pipeline & check output
if __name__ == "__main__":
    train_X, train_y, val_X, val_y, test_X, test_y = preprocess_pipeline(
        "weather-dataset/weatherAUS.csv"
    )

    print("Train shape:", train_X.shape, train_y.shape)
    print("Validation shape:", val_X.shape, val_y.shape)
    print("Test shape:", test_X.shape, test_y.shape)
