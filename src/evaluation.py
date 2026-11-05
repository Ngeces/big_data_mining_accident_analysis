"""
evaluation.py
Fungsi evaluasi bersama. Scikit-Learn dan Spark MLlib sama-sama menghasilkan
probabilitas Class 1, lalu dievaluasi dengan fungsi yang sama agar definisi metrik identik.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from config import THRESHOLD_GRID


def tune_threshold(y_val, val_probs, grid=THRESHOLD_GRID):
    """Pilih ambang dengan F1 Class 1 tertinggi pada validation set saja."""
    best_thresh, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_val, (val_probs >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_thresh, best_f1 = float(t), float(f1)
    if best_thresh in (grid[0], grid[-1]):
        print(f"  [!] Threshold optimal {best_thresh} berada di tepi grid; pertimbangkan memperlebar grid.")
    return best_thresh, best_f1


def classification_metrics(y_true, probs, threshold):
    """Metrik Class 1 + metrik global pada ambang tertentu."""
    preds = (probs >= threshold).astype(int)
    return {
        "Accuracy": round(accuracy_score(y_true, preds), 4),
        "Precision_Class1": round(precision_score(y_true, preds, pos_label=1, zero_division=0), 4),
        "Recall_Class1": round(recall_score(y_true, preds, pos_label=1, zero_division=0), 4),
        "F1_Class1": round(f1_score(y_true, preds, pos_label=1, zero_division=0), 4),
        "F1_Macro": round(f1_score(y_true, preds, average="macro", zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, probs), 4),
        "PR_AUC": round(average_precision_score(y_true, probs), 4),
        "Confusion_Matrix": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
    }


def majority_baseline(y_train, y_test):
    """Pembanding sederhana: selalu memprediksi kelas mayoritas pada training set."""
    majority = int(np.mean(y_train) >= 0.5)
    probs = np.full(len(y_test), float(majority))
    preds = probs.astype(int)
    return {
        "Framework": "Baseline",
        "Model Name": f"Majority Class (selalu {majority})",
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision_Class1": round(precision_score(y_test, preds, pos_label=1, zero_division=0), 4),
        "Recall_Class1": round(recall_score(y_test, preds, pos_label=1, zero_division=0), 4),
        "F1_Class1": round(f1_score(y_test, preds, pos_label=1, zero_division=0), 4),
        "F1_Macro": round(f1_score(y_test, preds, average="macro", zero_division=0), 4),
        "ROC_AUC": 0.5,
        "PR_AUC": round(float(np.mean(y_test)), 4),  # PR-AUC acak = prevalensi Class 1
        "Confusion_Matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
    }
