"""
Automated verification tests for model artifacts, round-trip reproducibility,
and inference input schema validation using Python's built-in unittest framework.
"""

import os
import sys
import json
import joblib
import unittest
import pandas as pd
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data, split_data
from src.predict import HeartDiseasePredictor
from src.evaluation import calculate_metrics


class TestPipelineAndArtifacts(unittest.TestCase):

    def test_artifact_exists(self):
        """Verify that required model artifacts exist on disk."""
        self.assertTrue(os.path.exists("models/xgboost_pipeline.pkl"), "models/xgboost_pipeline.pkl missing!")
        self.assertTrue(os.path.exists("models/model_metrics.json"), "models/model_metrics.json missing!")
        self.assertTrue(os.path.exists("assets/model_comparison.png"), "assets/model_comparison.png missing!")
        self.assertTrue(os.path.exists("assets/confusion_matrix_xgboost.png"), "assets/confusion_matrix_xgboost.png missing!")
        self.assertTrue(os.path.exists("assets/feature_importance.png"), "assets/feature_importance.png missing!")

    def test_roundtrip_metric_parity(self):
        """
        Verify that loading the serialized pipeline in a fresh process and evaluating
        on the held-out test split produces exact parity with saved metrics.
        """
        X, y = load_data("data/heart_final.csv")
        _, X_test, _, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        pipeline = joblib.load("models/xgboost_pipeline.pkl")
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        fresh_metrics = calculate_metrics(y_test.values, y_pred, y_prob)
        
        with open("models/model_metrics.json", "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
            
        xgb_saved = saved_metrics["XGBoost"]
        
        self.assertAlmostEqual(fresh_metrics["accuracy"], xgb_saved["accuracy"], places=5)
        self.assertAlmostEqual(fresh_metrics["precision"], xgb_saved["precision"], places=5)
        self.assertAlmostEqual(fresh_metrics["recall"], xgb_saved["recall"], places=5)
        self.assertAlmostEqual(fresh_metrics["f1_score"], xgb_saved["f1_score"], places=5)
        self.assertAlmostEqual(fresh_metrics["roc_auc"], xgb_saved["roc_auc"], places=5)

    def test_predictor_valid_inference(self):
        """Verify inference engine on valid input records."""
        predictor = HeartDiseasePredictor()
        
        sample_normal = {
            'Age': 40,
            'Sex': 'M',
            'ChestPainType': 'ATA',
            'RestingBP': 140,
            'Cholesterol': 289,
            'FastingBS': 0,
            'RestingECG': 'Normal',
            'MaxHR': 172,
            'ExerciseAngina': 'N',
            'Oldpeak': 0.0,
            'ST_Slope': 'Up'
        }
        
        res = predictor.predict(sample_normal)
        self.assertIn(res['prediction'], [0, 1])
        self.assertTrue(0.0 <= res['model_estimated_probability'] <= 1.0)
        self.assertIn("likelihood of heart disease", res['prediction_label'])

    def test_predictor_validation_error(self):
        """Verify input validation catches invalid inputs."""
        predictor = HeartDiseasePredictor()
        
        # Missing required key
        invalid_sample = {
            'Age': 50,
            'Sex': 'M'
        }
        with self.assertRaises(ValueError):
            predictor.predict(invalid_sample)
            
        # Out of range numerical value
        out_of_range_sample = {
            'Age': 150,  # Invalid
            'Sex': 'M',
            'ChestPainType': 'ATA',
            'RestingBP': 140,
            'Cholesterol': 289,
            'FastingBS': 0,
            'RestingECG': 'Normal',
            'MaxHR': 172,
            'ExerciseAngina': 'N',
            'Oldpeak': 0.0,
            'ST_Slope': 'Up'
        }
        with self.assertRaises(ValueError):
            predictor.predict(out_of_range_sample)


if __name__ == "__main__":
    unittest.main()
