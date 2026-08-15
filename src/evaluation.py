"""
Model evaluation, performance metric calculations, and visualization utilities.
Developed during PYML Internship at Anveshan Foundation, IGDTUW.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes all standard binary classification metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else 0.0
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report
    }


def extract_xgboost_gain_importance(pipeline) -> pd.DataFrame:
    """
    Extracts gain-based feature importance from a fitted imblearn Pipeline
    and accurately maps transformed feature names from ColumnTransformer.
    
    Handles both raw booster key formats ('f0', 'f1', etc.) and named features.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    feature_names = list(preprocessor.get_feature_names_out())
    booster = classifier.get_booster()
    score_dict = booster.get_score(importance_type='gain')
    
    importances = []
    for i, name in enumerate(feature_names):
        # XGBoost booster may index transformed columns as 'f0', 'f1' or direct feature strings
        gain_val = (
            score_dict.get(name) or
            score_dict.get(f'f{i}') or
            score_dict.get(str(i)) or
            0.0
        )
        importances.append(float(gain_val))
        
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Gain': importances
    })
    
    # Normalize gain to sum to 1.0 (or percentage) for clear interpretability
    total_gain = df_imp['Gain'].sum()
    if total_gain > 0:
        df_imp['RelativeGain'] = df_imp['Gain'] / total_gain
    else:
        df_imp['RelativeGain'] = 0.0
        
    # Clean up feature name prefixes for better presentation
    df_imp['CleanFeature'] = (
        df_imp['Feature']
        .str.replace('num__', '', regex=False)
        .str.replace('cat__', '', regex=False)
    )
    
    df_imp = df_imp.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    return df_imp


def plot_model_comparison(metrics_dict: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Generates a grouped bar chart comparing all 6 algorithms across standard metrics.
    """
    records = []
    for model_name, m in metrics_dict.items():
        records.append({
            'Model': model_name,
            'Accuracy': m['accuracy'] * 100,
            'Precision': m['precision'] * 100,
            'Recall': m['recall'] * 100,
            'F1-Score': m['f1_score'] * 100,
            'ROC-AUC': m['roc_auc'] * 100
        })
    df = pd.DataFrame(records)
    
    df_melt = df.melt(id_vars='Model', var_name='Metric', value_name='Score (%)')
    
    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("Set2", len(df['Model'].unique()))
    ax = sns.barplot(data=df_melt, x='Metric', y='Score (%)', hue='Model', palette=palette)
    plt.title('Performance Comparison of 6 Machine Learning Classifiers', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(65, 100)
    plt.ylabel('Score (%)', fontsize=12)
    plt.xlabel('Evaluation Metric', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return plt.gcf()


def plot_confusion_matrix(cm: list, model_name: str = "XGBoost", save_path: Optional[str] = None):
    """
    Plots a high-contrast heatmap of the confusion matrix.
    """
    cm_arr = np.array(cm)
    plt.figure(figsize=(6, 5))
    
    # Formatted annotations with count and category
    group_counts = [f"{val:d}" for val in cm_arr.flatten()]
    group_labels = ["True Neg (Normal)", "False Pos (Leak/Type I)", "False Neg (Miss/Type II)", "True Pos (Disease)"]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_labels, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    
    sns.heatmap(
        cm_arr, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        cbar=False,
        annot_kws={"size": 11, "fontweight": "medium"}
    )
    plt.title(f'{model_name} Confusion Matrix (Held-out Test Split)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label (0 = Normal, 1 = Heart Disease)', fontsize=10)
    plt.ylabel('Actual True Label (0 = Normal, 1 = Heart Disease)', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return plt.gcf()


def plot_feature_importance(df_imp: pd.DataFrame, top_n: int = 12, save_path: Optional[str] = None):
    """
    Plots horizontal bar chart of gain-based feature importances.
    """
    df_top = df_imp.head(top_n).sort_values(by='Gain', ascending=True)
    
    plt.figure(figsize=(9, 6))
    bars = plt.barh(df_top['CleanFeature'], df_top['Gain'], color='#2b5c8f', edgecolor='#183d61', alpha=0.85)
    plt.title('XGBoost Gain-Based Feature Importance', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Total Improvement in Accuracy/Loss Brought by Feature (Gain)', fontsize=11)
    plt.ylabel('Transformed Model Feature', fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add value annotations
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (0.01 * df_top['Gain'].max()), bar.get_y() + bar.get_height()/2, 
                 f"{width:.2f}", va='center', ha='left', fontsize=9, color='#333333')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return plt.gcf()
