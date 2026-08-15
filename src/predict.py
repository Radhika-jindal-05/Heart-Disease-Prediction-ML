"""
Inference engine and input validation for Heart Disease Risk Prediction.
Developed during PYML Internship at Anveshan Foundation, IGDTUW.
"""

import os
import sys
import joblib
import pandas as pd
from typing import Dict, Any, Union

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import NUMERICAL_COLS, CATEGORICAL_COLS, FEATURE_SCHEMA


class HeartDiseasePredictor:
    """
    Production-style predictor that loads the unified imblearn Pipeline
    (ColumnTransformer + SMOTE + XGBoost) and executes inference.
    """
    
    def __init__(self, pipeline_path: str = "models/xgboost_pipeline.pkl"):
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(
                f"Model pipeline artifact not found at '{pipeline_path}'. "
                "Please run 'python src/train.py' first to train and serialize the model."
            )
        self.pipeline = joblib.load(pipeline_path)
        self.expected_features = NUMERICAL_COLS + CATEGORICAL_COLS
        
    def validate_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates input dictionary against feature schema and returns a 1-row DataFrame.
        """
        missing_keys = [col for col in self.expected_features if col not in input_data]
        if missing_keys:
            raise ValueError(f"Missing required input features: {missing_keys}")
            
        validated_dict = {}
        for col in self.expected_features:
            val = input_data[col]
            rules = FEATURE_SCHEMA.get(col, {})
            
            # Numeric validation
            if rules.get('type') in ['int', 'float']:
                try:
                    num_val = float(val) if rules.get('type') == 'float' else int(val)
                except (ValueError, TypeError):
                    raise ValueError(f"Feature '{col}' must be numeric. Received: {val}")
                    
                if num_val < rules['min'] or num_val > rules['max']:
                    raise ValueError(
                        f"Feature '{col}' value ({num_val}) out of expected range "
                        f"[{rules['min']}, {rules['max']}]."
                    )
                validated_dict[col] = num_val
                
            # Categorical validation
            elif rules.get('type') == 'category':
                if val not in rules['options']:
                    # Try type conversion for integer categories like FastingBS (e.g. "0" -> 0)
                    try:
                        val_converted = int(val)
                        if val_converted in rules['options']:
                            val = val_converted
                        else:
                            raise ValueError()
                    except Exception:
                        raise ValueError(
                            f"Feature '{col}' invalid value: '{val}'. "
                            f"Must be one of {rules['options']}."
                        )
                validated_dict[col] = val
                
        return pd.DataFrame([validated_dict], columns=self.expected_features)
        
    def predict(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes prediction on validated patient input.
        Returns predicted binary class, human-readable label, and model-estimated probability score.
        """
        if isinstance(input_data, dict):
            df_input = self.validate_input(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df_input = input_data[self.expected_features].copy()
        else:
            raise TypeError("input_data must be a dictionary or a pandas DataFrame.")
            
        # The imblearn pipeline handles preprocessing and classification (bypassing SMOTE during predict)
        pred_class = int(self.pipeline.predict(df_input)[0])
        probabilities = self.pipeline.predict_proba(df_input)[0]
        prob_heart_disease = float(probabilities[1])
        
        label = (
            "Higher likelihood of heart disease" 
            if pred_class == 1 
            else "Lower likelihood of heart disease"
        )
        
        return {
            "prediction": pred_class,
            "prediction_label": label,
            "model_estimated_probability": prob_heart_disease
        }


if __name__ == "__main__":
    # Test predictor on a sample record
    sample_patient = {
        'Age': 54,
        'Sex': 'M',
        'ChestPainType': 'ASY',
        'RestingBP': 140,
        'Cholesterol': 289,
        'FastingBS': 0,
        'RestingECG': 'Normal',
        'MaxHR': 150,
        'ExerciseAngina': 'N',
        'Oldpeak': 1.5,
        'ST_Slope': 'Flat'
    }
    
    try:
        predictor = HeartDiseasePredictor()
        result = predictor.predict(sample_patient)
        print("Sample Prediction Result:")
        print(f"  Predicted Class: {result['prediction']}")
        print(f"  Label:           {result['prediction_label']}")
        print(f"  Estimated Prob:  {result['model_estimated_probability']*100:.1f}%")
    except Exception as e:
        print(f"Predictor test notice: {e}")
