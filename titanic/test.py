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


def preprocess_test_data(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering to test data using TRAINING data statistics."""
    # Create enhanced feature set (same as in training)
    X = df.copy()
    
    # Family size engineering
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
    X['SmallFamily'] = ((X['FamilySize'] >= 2) & (X['FamilySize'] <= 4)).astype(int)
    X['LargeFamily'] = (X['FamilySize'] > 4).astype(int)
    
    # Title extraction
    X['Title'] = X['Name'].apply(extract_title)
    
    # Age groups - USE TRAINING DATA MEDIAN!
    train_age_median = train_df['Age'].median()
    X['Age_filled'] = X['Age'].fillna(train_age_median)
    X['AgeGroup'] = pd.cut(X['Age_filled'], bins=[0, 12, 18, 35, 60, 100], 
                          labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])
    X['IsChild'] = (X['Age_filled'] <= 12).astype(int)
    
    # Deck extraction from Cabin (first letter), unknown as 'U'
    X['Deck'] = X['Cabin'].fillna('U').astype(str).str[0]

    # Fare per person (avoid divide by zero)
    X['FarePerPerson'] = X['Fare'] / (X['SibSp'] + X['Parch'] + 1)

    # Fare groups - USE TRAINING DATA STATISTICS!
    train_fare_median = train_df['Fare'].median()
    train_fare_filled = train_df['Fare'].fillna(train_fare_median)
    
    # Get quartile boundaries from training data
    _, fare_bins = pd.qcut(train_fare_filled, q=4, retbins=True, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # Apply same boundaries to test data
    X['FareGroup'] = pd.cut(X['Fare'].fillna(train_fare_median), 
                           bins=fare_bins, labels=['Low', 'Medium', 'High', 'VeryHigh'], include_lowest=True)
    
    # Select the same features used in training
    feature_columns = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'FamilySize', 'IsAlone', 'SmallFamily', 'LargeFamily', 
        'Title', 'AgeGroup', 'IsChild', 'FareGroup', 'Deck', 'FarePerPerson'
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
    
    # Load training data to get consistent preprocessing statistics
    from data_reader import load_train_dataframe
    train_df = load_train_dataframe()
    
    # Preprocess test data using training data statistics
    X_test = preprocess_test_data(test_df, train_df)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of passengers: {len(passenger_ids)}")
    
    # Train the model (or load if you have a saved one)
    print(f"\nTraining {model_type} model...")
    
    if model_type == "simple":
        # Use simple logistic regression to avoid overfitting
        from training import train_logistic_regression
        model, train_accuracy, _, _, _, _ = train_logistic_regression()
        print(f"Model trained with accuracy: {train_accuracy:.4f}")
    else:
        # Use a simpler Random Forest to avoid overfitting
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from data_reader import prepare_training_data
        
        X_train_full, y_train_full, _, _ = prepare_training_data()
        
        # Use simpler Random Forest parameters
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=6,  # Reduced from 10
            min_samples_split=5,  # Increased from 2
            min_samples_leaf=3,   # Increased from 2
            random_state=42
        )
        model.fit(X_train_full, y_train_full)
        
        # Check CV score
        cv_scores = cross_val_score(model, X_train_full, y_train_full, cv=5)
        train_accuracy = cv_scores.mean()
        print(f"Simplified RF CV accuracy: {train_accuracy:.4f}")
    
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
