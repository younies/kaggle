from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from data_reader import prepare_training_data


def train_logistic_regression(
    csv_path: Optional[Path | str] = None, test_size: float = 0.2, random_state: int = 42
) -> Tuple[LogisticRegression, float, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Train a logistic regression model on the Titanic data.
    
    Returns (model, accuracy, X_test, y_test, X_train, y_train).
    """
    # Load and preprocess data
    X, y, _, _ = prepare_training_data(csv_path)
    
    # Split into train/validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train logistic regression
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, X_train, y_train


def _print_model_results(model: LogisticRegression, accuracy: float, X_test: np.ndarray, y_test: pd.Series) -> None:
    print(f"\nLogistic Regression Results:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Model coefficients shape: {model.coef_.shape}")
    
    # Predictions on test set
    y_pred = model.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))


def _print_brief_overview(
    X: np.ndarray, y: pd.Series, feature_names: List[str]
) -> None:
    print("Train matrix shape:", X.shape)
    print("Target shape:", y.shape)
    preview_cols = min(10, X.shape[1])
    print("First 5 rows (first", preview_cols, "features):")
    with np.printoptions(precision=3, suppress=True):
        print(X[:5, :preview_cols])
    print("Feature names (first", preview_cols, "):")
    print(feature_names[:preview_cols])


def main() -> LogisticRegression:
    # Show data overview
    X, y, _, feature_names = prepare_training_data()
    _print_brief_overview(X, y, feature_names)
    
    # Train and evaluate model
    model, accuracy, X_test, y_test, _, _ = train_logistic_regression()
    _print_model_results(model, accuracy, X_test, y_test)
    
    print(f"\nTrained model is ready for predictions!")
    return model


if __name__ == "__main__":
    trained_model = main()
