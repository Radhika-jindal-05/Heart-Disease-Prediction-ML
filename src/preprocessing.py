"""
Preprocessing utilities and ColumnTransformer definition for Heart Disease Prediction ML.
Developed during PYML Internship at Anveshan Foundation, IGDTUW.
"""

import pandas as pd
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Define feature sets
NUMERICAL_COLS: List[str] = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'MaxHR',
    'Oldpeak'
]

CATEGORICAL_COLS: List[str] = [
    'Sex',
    'ChestPainType',
    'FastingBS',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]

TARGET_COL: str = 'HeartDisease'

# Valid categories for validation and UI dropdowns
FEATURE_SCHEMA = {
    'Age': {'type': 'int', 'min': 18, 'max': 100, 'default': 54},
    'Sex': {'type': 'category', 'options': ['M', 'F'], 'default': 'M'},
    'ChestPainType': {'type': 'category', 'options': ['ASY', 'NAP', 'ATA', 'TA'], 'default': 'ASY'},
    'RestingBP': {'type': 'int', 'min': 50, 'max': 250, 'default': 130},
    'Cholesterol': {'type': 'int', 'min': 0, 'max': 650, 'default': 220},
    'FastingBS': {'type': 'category', 'options': [0, 1], 'default': 0},
    'RestingECG': {'type': 'category', 'options': ['Normal', 'LVH', 'ST'], 'default': 'Normal'},
    'MaxHR': {'type': 'int', 'min': 50, 'max': 230, 'default': 140},
    'ExerciseAngina': {'type': 'category', 'options': ['N', 'Y'], 'default': 'N'},
    'Oldpeak': {'type': 'float', 'min': -3.0, 'max': 7.0, 'default': 0.0},
    'ST_Slope': {'type': 'category', 'options': ['Flat', 'Up', 'Down'], 'default': 'Flat'}
}


def get_preprocessor() -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer for numerical standardization
    and categorical one-hot encoding.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CATEGORICAL_COLS)
        ],
        remainder='drop'
    )


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw CSV data and separates into feature matrix X and target vector y.
    """
    data = pd.read_csv(file_path)
    
    # Ensure all expected columns exist
    expected_cols = set(NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET_COL])
    if not expected_cols.issubset(set(data.columns)):
        missing = expected_cols - set(data.columns)
        raise ValueError(f"Missing required columns in dataset: {missing}")
        
    X = data[NUMERICAL_COLS + CATEGORICAL_COLS].copy()
    y = data[TARGET_COL].astype(int).copy()
    return X, y


def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train/test split to preserve class ratio across splits.
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
