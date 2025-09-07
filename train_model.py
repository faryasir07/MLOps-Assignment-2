from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_logistic_regression(train_X, train_y, val_X, val_y, test_X, test_y):
    """
    Train Logistic Regression model and evaluate on validation and test sets.
    """
    # Initialize Logistic Regression
    model = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)

    # Train model
    model.fit(train_X, train_y)

    # Validation performance
    val_preds = model.predict(val_X)
    print("Validation Accuracy:", accuracy_score(val_y, val_preds))
    print("\nValidation Report:\n", classification_report(val_y, val_preds))
    print("Validation Confusion Matrix:\n", confusion_matrix(val_y, val_preds))

    # Final Test performance
    test_preds = model.predict(test_X)
    print("\nTest Accuracy:", accuracy_score(test_y, test_preds))
    print("\nTest Report:\n", classification_report(test_y, test_preds))
    print("Test Confusion Matrix:\n", confusion_matrix(test_y, test_preds))

    return model
