#!/usr/bin/env python3
"""
Generate Kaggle submission file using the trained Titanic model.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from data_reader import extract_title, get_project_root
from training import train_best_model_with_tuning


def load_test_data(csv_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load the Titanic test dataset."""
    if csv_path is None:
        csv_path = get_project_root() / "test.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find test.csv at: {csv_path}")
    return pd.read_csv(csv_path)


def preprocess_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering to test data as used in training."""
    # Create enhanced feature set (same as in training)
    X = df.copy()
    
    # Family size engineering
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
    X['SmallFamily'] = ((X['FamilySize'] >= 2) & (X['FamilySize'] <= 4)).astype(int)
    X['LargeFamily'] = (X['FamilySize'] > 4).astype(int)
    
    # Title extraction
    X['Title'] = X['Name'].apply(extract_title)
    
    # Age groups (fill missing ages with median first for grouping)
    age_median = X['Age'].median()
    X['Age_filled'] = X['Age'].fillna(age_median)
    X['AgeGroup'] = pd.cut(X['Age_filled'], bins=[0, 12, 18, 35, 60, 100], 
                          labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])
    X['IsChild'] = (X['Age_filled'] <= 12).astype(int)
    
    # Fare groups (handle missing fares)
    fare_median = X['Fare'].median()
    X['FareGroup'] = pd.qcut(X['Fare'].fillna(fare_median), 
                            q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # Select the same features used in training
    feature_columns = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'FamilySize', 'IsAlone', 'SmallFamily', 'LargeFamily', 
        'Title', 'AgeGroup', 'IsChild', 'FareGroup'
    ]
    
    X_final = X[feature_columns].copy()
    return X_final


def generate_submission(model_type: str = "tuned", output_file: str = "submission.csv") -> None:
    """Generate Kaggle submission file.
    
    Args:
        model_type: "tuned" for hyperparameter-tuned model, "simple" for basic logistic regression
        output_file: Name of the output CSV file
    """
    print("Loading and preprocessing test data...")
    
    # Load test data
    test_df = load_test_data()
    passenger_ids = test_df['PassengerId'].copy()
    
    # Preprocess test data with same feature engineering
    X_test = preprocess_test_data(test_df)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of passengers: {len(passenger_ids)}")
    
    # Train the model (or load if you have a saved one)
    print(f"\nTraining {model_type} model...")
    
    if model_type == "tuned":
        # Get the best tuned model
        model, train_accuracy, _, _ = train_best_model_with_tuning()
        print(f"Model trained with accuracy: {train_accuracy:.4f}")
    else:
        # Use simple logistic regression
        from training import train_logistic_regression
        model, train_accuracy, _, _, _, _ = train_logistic_regression()
        print(f"Model trained with accuracy: {train_accuracy:.4f}")
    
    # Get the same preprocessor used in training
    print("\nPreprocessing test data...")
    
    # We need to fit the preprocessor on training data first to ensure consistency
    from data_reader import prepare_training_data
    X_train, y_train, preprocessor, _ = prepare_training_data()
    
    # Transform test data using the fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed test data shape: {X_test_processed.shape}")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test_processed)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    # Save submission file
    output_path = get_project_root() / output_file
    submission.to_csv(output_path, index=False)
    
    print(f"\n✅ Submission file created: {output_path}")
    print(f"📊 Predictions summary:")
    print(f"   - Total passengers: {len(predictions)}")
    print(f"   - Predicted survivors: {predictions.sum()} ({predictions.mean():.1%})")
    print(f"   - Predicted non-survivors: {len(predictions) - predictions.sum()} ({1 - predictions.mean():.1%})")
    
    # Show first few predictions
    print(f"\n🔍 First 10 predictions:")
    print(submission.head(10).to_string(index=False))
    
    return submission


def main():
    """Main function to generate submission."""
    print("🚢 Titanic Survival Prediction - Kaggle Submission Generator")
    print("=" * 60)
    
    # Generate submission with tuned model
    submission = generate_submission(model_type="tuned", output_file="submission.csv")
    
    print("\n" + "=" * 60)
    print("🎯 Ready for Kaggle submission!")
    print("Upload the 'submission.csv' file to Kaggle competition.")
    
    return submission


if __name__ == "__main__":
    submission = main()
