from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC

from data_reader import prepare_training_data


def compare_models(csv_path: Optional[Path | str] = None, test_size: float = 0.2, random_state: int = 42):
    """Compare multiple machine learning models on the Titanic data."""
    # Load and preprocess data
    X, y, _, _ = prepare_training_data(csv_path)
    
    # Split into train/validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'SVM': SVC(random_state=random_state, probability=True)
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    print("Model Comparison Results:")
    print("=" * 50)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Test set predictions
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'model': model,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        print(f"\n{name}:")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        # Track best model
        if cv_mean > best_score:
            best_score = cv_mean
            best_model = model
            best_model_name = name
    
    print(f"\nBest Model: {best_model_name} (CV Score: {best_score:.4f})")
    
    return best_model, results, X_test, y_test


def train_best_model_with_tuning(csv_path: Optional[Path | str] = None, test_size: float = 0.2, random_state: int = 42):
    """Train the best model with hyperparameter tuning."""
    # Load and preprocess data
    X, y, _, _ = prepare_training_data(csv_path)
    
    # Split into train/validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Random Forest with hyperparameter tuning
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Hyperparameter tuning for Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        rf_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    rf_grid.fit(X_train, y_train)
    
    # Get best model
    best_model = rf_grid.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nBest Random Forest Parameters: {rf_grid.best_params_}")
    print(f"Best CV Score: {rf_grid.best_score_:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return best_model, test_accuracy, X_test, y_test


def train_logistic_regression(
    csv_path: Optional[Path | str] = None, test_size: float = 0.2, random_state: int = 42
) -> Tuple[LogisticRegression, float, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Train a logistic regression model on the Titanic data (legacy function)."""
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


def main():
    # Show data overview
    X, y, _, feature_names = prepare_training_data()
    _print_brief_overview(X, y, feature_names)
    
    print("\n" + "="*60)
    print("PHASE 1: Model Comparison")
    print("="*60)
    
    # Compare multiple models
    best_model, results, X_test, y_test = compare_models()
    
    print("\n" + "="*60)
    print("PHASE 2: Hyperparameter Tuning")
    print("="*60)
    
    # Train with hyperparameter tuning
    tuned_model, tuned_accuracy, _, _ = train_best_model_with_tuning()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    # Final evaluation
    y_pred = tuned_model.predict(X_test)
    print(f"\nFinal Model Test Accuracy: {tuned_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nOptimized model is ready for predictions!")
    return tuned_model


if __name__ == "__main__":
    trained_model = main()
