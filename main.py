from preprocessing import preprocess_pipeline
from train_model import train_logistic_regression

if __name__ == "__main__":
    train_X, train_y, val_X, val_y, test_X, test_y = preprocess_pipeline(
        "D:/MLOPs Assignemnt 2/MLOps-Assignment-2/weather-dataset/weatherAUS.csv"
    )

    model = train_logistic_regression(train_X, train_y, val_X, val_y, test_X, test_y)
