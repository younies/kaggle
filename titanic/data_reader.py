from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def get_project_root() -> Path:
    """Return the repository root (two levels up from this file)."""
    return Path(__file__).resolve().parent


def load_train_dataframe(csv_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load the Titanic training dataset as a pandas DataFrame.

    If csv_path is not provided, it defaults to train.csv next to this script.
    """
    if csv_path is None:
        csv_path = get_project_root() / "train.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find train.csv at: {csv_path}")
    return pd.read_csv(csv_path)


def extract_title(name: str) -> str:
    """Extract title from passenger name."""
    if pd.isna(name):
        return "Unknown"
    
    title = name.split(',')[1].split('.')[0].strip()
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    return title_mapping.get(title, 'Rare')


def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the raw dataframe into features X and target y with advanced feature engineering.

    - Target: Survived
    - Features: Enhanced with family size, title, age groups, fare groups, etc.
    """
    required_columns: List[str] = [
        "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", 
        "Fare", "Embarked", "Name"
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError("train.csv is missing required columns: " + ", ".join(missing))

    # Create enhanced feature set
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
    
    # Fare groups
    X['FareGroup'] = pd.qcut(X['Fare'].fillna(X['Fare'].median()), 
                            q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # Select final features
    feature_columns = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'FamilySize', 'IsAlone', 'SmallFamily', 'LargeFamily', 
        'Title', 'AgeGroup', 'IsChild', 'FareGroup'
    ]
    
    y = df["Survived"].astype(int)
    X_final = X[feature_columns].copy()
    
    return X_final, y


def build_preprocessor() -> ColumnTransformer:
    """Create a ColumnTransformer that imputes and encodes features for ML models."""
    numeric_features: List[str] = [
        "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", 
        "IsAlone", "SmallFamily", "LargeFamily", "IsChild"
    ]
    categorical_features: List[str] = ["Sex", "Embarked", "Title", "AgeGroup", "FareGroup"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Use dense output for convenience when inspecting arrays locally.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def prepare_training_data(
    csv_path: Optional[Path | str] = None,
) -> Tuple[np.ndarray, pd.Series, ColumnTransformer, List[str]]:
    """Load, split, and preprocess the Titanic training data.

    Returns (X_prepared, y, preprocessor, feature_names).
    - X_prepared: numpy array ready for ML models
    - y: target Series (Survived)
    - preprocessor: fitted ColumnTransformer to reuse on validation/test data
    - feature_names: names of transformed columns (for inspection)
    """
    df = load_train_dataframe(csv_path)
    X_df, y = split_features_and_target(df)
    preprocessor = build_preprocessor()
    X_prepared = preprocessor.fit_transform(X_df)

    # Get feature names in the transformed matrix for interpretability.
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if older sklearn; names are not critical for training.
        feature_names = [f"f{i}" for i in range(X_prepared.shape[1])]

    # Ensure dense array (OneHotEncoder sparse=False already requests dense)
    if not isinstance(X_prepared, np.ndarray):
        X_prepared = X_prepared.toarray()

    return X_prepared, y, preprocessor, feature_names


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


def main() -> None:
    # Show data overview
    X, y, _, feature_names = prepare_training_data()
    _print_brief_overview(X, y, feature_names)


if __name__ == "__main__":
    trained_model = main()
