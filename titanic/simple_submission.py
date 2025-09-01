#!/usr/bin/env python3
"""
Simple, robust Titanic submission with minimal feature engineering.
Focus on avoiding overfitting and ensuring train/test consistency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

def get_project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent

def simple_feature_engineering(df, is_train=True, train_stats=None):
    """Simple, robust feature engineering that avoids overfitting."""
    X = df.copy()
    
    # Basic family size
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
    
    # Simple title extraction
    X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 5, 'Col': 5, 'Major': 5, 'Mlle': 2, 'Countess': 3, 'Ms': 2, 'Lady': 3, 'Jonkheer': 5, 'Don': 5, 'Dona': 3, 'Mme': 3, 'Capt': 5, 'Sir': 5}
    X['Title'] = X['Title'].map(title_mapping).fillna(0)
    
    # Handle missing values consistently
    if is_train:
        # Compute statistics from training data
        age_median = X['Age'].median()
        fare_median = X['Fare'].median()
        embarked_mode = X['Embarked'].mode()[0] if not X['Embarked'].mode().empty else 'S'
        stats = {'age_median': age_median, 'fare_median': fare_median, 'embarked_mode': embarked_mode}
    else:
        # Use training statistics for test data
        stats = train_stats
        age_median = stats['age_median']
        fare_median = stats['fare_median']
        embarked_mode = stats['embarked_mode']
    
    # Fill missing values
    X['Age'] = X['Age'].fillna(age_median)
    X['Fare'] = X['Fare'].fillna(fare_median)
    X['Embarked'] = X['Embarked'].fillna(embarked_mode)
    
    # Simple age groups
    X['IsChild'] = (X['Age'] < 16).astype(int)
    X['IsElderly'] = (X['Age'] > 60).astype(int)
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    if is_train:
        X['Sex_encoded'] = le_sex.fit_transform(X['Sex'])
        X['Embarked_encoded'] = le_embarked.fit_transform(X['Embarked'])
        encoders = {'sex': le_sex, 'embarked': le_embarked}
        stats['encoders'] = encoders
    else:
        # Use training encoders
        le_sex = train_stats['encoders']['sex']
        le_embarked = train_stats['encoders']['embarked']
        
        # Handle unseen categories
        sex_classes = le_sex.classes_
        embarked_classes = le_embarked.classes_
        
        X['Sex_encoded'] = X['Sex'].apply(lambda x: le_sex.transform([x])[0] if x in sex_classes else 0)
        X['Embarked_encoded'] = X['Embarked'].apply(lambda x: le_embarked.transform([x])[0] if x in embarked_classes else 0)
    
    # Select final features
    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 
                'Embarked_encoded', 'FamilySize', 'IsAlone', 'Title', 'IsChild', 'IsElderly']
    
    X_final = X[features].copy()
    
    if is_train:
        return X_final, stats
    else:
        return X_final

def create_simple_submission():
    """Create submission with simple, robust approach."""
    print("🚢 Simple Titanic Submission Generator")
    print("=" * 50)
    
    # Load data
    train_df = pd.read_csv(get_project_root() / "train.csv")
    test_df = pd.read_csv(get_project_root() / "test.csv")
    
    print(f"Train data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Prepare training data
    X_train, train_stats = simple_feature_engineering(train_df, is_train=True)
    y_train = train_df['Survived']
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # Prepare test data using training statistics
    X_test = simple_feature_engineering(test_df, is_train=False, train_stats=train_stats)
    
    print(f"Test features shape: {X_test.shape}")
    
    # Compare different simple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Simple RF': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    print("\nModel Comparison:")
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = cv_scores.mean()
        print(f"{name}: {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with CV score: {best_score:.4f}")
    
    # Train best model on full training data
    best_model.fit(X_train, y_train)
    
    # Make predictions
    predictions = best_model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    # Save submission
    output_path = get_project_root() / "simple_submission.csv"
    submission.to_csv(output_path, index=False)
    
    print(f"\n✅ Simple submission created: {output_path}")
    print(f"📊 Predictions: {predictions.sum()} survivors out of {len(predictions)} ({predictions.mean():.1%})")
    
    # Show feature importance if Random Forest
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Top 5 Feature Importances:")
        print(importance_df.head().to_string(index=False))
    
    return submission

if __name__ == "__main__":
    submission = create_simple_submission()
