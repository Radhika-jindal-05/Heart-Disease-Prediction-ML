"""
Model training, hyperparameter optimization, evaluation, and artifact serialization.
Developed during PYML Internship at Anveshan Foundation, IGDTUW.
"""

import os
import sys
import json
import joblib
import pandas as pd
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocessing import get_preprocessor, load_data, split_data
from src.evaluation import (
    calculate_metrics,
    extract_xgboost_gain_importance,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_feature_importance
)


def get_model_definitions():
    """
    Returns base estimators and hyperparameter search spaces for all 6 models.
    All hyperparameter keys are prefixed with 'classifier__' for pipeline compatibility.
    """
    return {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['lbfgs', 'liblinear']
            },
            "n_iter": 5
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                'classifier__n_neighbors': list(range(3, 20)),
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan']
            },
            "n_iter": 10
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf', 'poly'],
                'classifier__gamma': ['scale', 'auto']
            },
            "n_iter": 6
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__max_depth': [None] + list(range(3, 20)),
                'classifier__min_samples_split': [2, 5, 10, 15],
                'classifier__min_samples_leaf': [1, 2, 4, 6]
            },
            "n_iter": 20
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'classifier__n_estimators': [100, 200, 300, 400, 500],
                'classifier__max_depth': [None, 5, 10, 15, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__bootstrap': [True, False]
            },
            "n_iter": 20
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', random_state=42),
            "params": {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 5, 7, 10],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__subsample': [0.6, 0.8, 1.0],
                'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                'classifier__gamma': [0, 1, 5]
            },
            "n_iter": 20
        }
    }


def train_and_evaluate(
    data_path: str = "data/heart_final.csv",
    output_dir: str = "models",
    assets_dir: str = "assets"
) -> Dict[str, Any]:
    """
    Executes the end-to-end training and evaluation pipeline:
    1. Loads raw dataset.
    2. Performs stratified train/test split.
    3. Builds imblearn.pipeline.Pipeline (ColumnTransformer -> SMOTE -> Classifier)
       to ensure zero data leakage within cross-validation folds.
    4. Evaluates all 6 models and serializes artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    print(f"Loading dataset from '{data_path}'...")
    X, y = load_data(data_path)
    print(f"Total samples: {len(X)} | Features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")
    
    model_defs = get_model_definitions()
    all_metrics = {}
    best_pipelines = {}
    
    print("\n" + "="*70)
    print("STARTING 5-FOLD CROSS-VALIDATION & HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    for name, config in model_defs.items():
        print(f"\n--- Training {name} ---")
        
        # Construct leakage-safe pipeline
        pipeline = ImbPipeline([
            ('preprocessor', get_preprocessor()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', config['model'])
        ])
        
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config['params'],
            n_iter=config['n_iter'],
            cv=5,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        best_pipelines[name] = best_pipeline
        
        # Evaluate on untouched test split
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1] if hasattr(best_pipeline, "predict_proba") else None
        
        m = calculate_metrics(y_test.values, y_pred, y_prob)
        clean_params = {k.replace('classifier__', ''): v for k, v in search.best_params_.items()}
        m['best_params'] = clean_params
        all_metrics[name] = m
        
        print(f"Test Accuracy:  {m['accuracy']*100:.2f}%")
        print(f"Precision:      {m['precision']*100:.2f}%")
        print(f"Recall:         {m['recall']*100:.2f}%")
        print(f"F1-Score:       {m['f1_score']*100:.2f}%")
        print(f"ROC-AUC:        {m['roc_auc']*100:.2f}%")
        print(f"Best Params:    {clean_params}")
        
    print("\n" + "="*70)
    print("FINAL 6-MODEL BENCHMARK COMPARISON TABLE")
    print("="*70)
    summary_df = pd.DataFrame([
        {
            'Model': m_name,
            'Accuracy': f"{m['accuracy']*100:.2f}%",
            'Precision': f"{m['precision']*100:.2f}%",
            'Recall': f"{m['recall']*100:.2f}%",
            'F1-Score': f"{m['f1_score']*100:.2f}%",
            'ROC-AUC': f"{m['roc_auc']*100:.2f}%"
        }
        for m_name, m in all_metrics.items()
    ])
    print(summary_df.to_string(index=False))
    
    # Serialize the best XGBoost pipeline artifact
    best_xgb_pipeline = best_pipelines['XGBoost']
    model_artifact_path = os.path.join(output_dir, "xgboost_pipeline.pkl")
    joblib.dump(best_xgb_pipeline, model_artifact_path)
    print(f"\n[Artifact Saved] Unified XGBoost Pipeline: '{model_artifact_path}'")
    
    # Serialize model metrics JSON
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[Artifact Saved] Benchmark Metrics: '{metrics_path}'")
    
    # Generate and save publication-quality plots
    print("\nGenerating asset plots...")
    comparison_chart_path = os.path.join(assets_dir, "model_comparison.png")
    plot_model_comparison(all_metrics, save_path=comparison_chart_path)
    print(f"[Plot Saved] '{comparison_chart_path}'")
    
    cm_chart_path = os.path.join(assets_dir, "confusion_matrix_xgboost.png")
    plot_confusion_matrix(all_metrics['XGBoost']['confusion_matrix'], model_name="XGBoost", save_path=cm_chart_path)
    print(f"[Plot Saved] '{cm_chart_path}'")
    
    df_imp = extract_xgboost_gain_importance(best_xgb_pipeline)
    imp_chart_path = os.path.join(assets_dir, "feature_importance.png")
    plot_feature_importance(df_imp, save_path=imp_chart_path)
    print(f"[Plot Saved] '{imp_chart_path}'")
    
    print("\nPipeline training and serialization completed successfully.")
    return all_metrics


if __name__ == "__main__":
    train_and_evaluate()
