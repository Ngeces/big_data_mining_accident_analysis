"""
models_baseline.py
Implementasi model Scikit-Learn (Logistic Regression & Random Forest)
menggunakan metodologi partisi temporal (Train 2018-2021, Val 2022, Test 2023),
threshold tuning pada validation set, dan evaluasi fokus pada Class 1 (Lost-Time/Severe).
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def run_sklearn_pipeline(
    data_dir="data/processed",
    artifacts_dir="artifacts"
):
    print("=" * 60)
    print("[*] MEMULAI TRAINING DAN EVALUASI SCIKIT-LEARN (BASELINE)")
    print("=" * 60)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load Data Partisi
    print("[*] Memuat data partisi...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    cat_cols = ["SUBUNIT", "COAL_METAL_IND", "ACCIDENT_TYPE", "MINING_EQUIP", "ACTIVITY", "SHIFT_PERIOD"]
    num_cols = ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER"]
    target_col = "IS_LOST_TIME_OR_SEVERE"

    X_train = train_df[cat_cols + num_cols]
    y_train = train_df[target_col].values

    X_val = val_df[cat_cols + num_cols]
    y_val = val_df[target_col].values

    X_test = test_df[cat_cols + num_cols]
    y_test = test_df[target_col].values

    print(f"  Train: {X_train.shape[0]:,} baris | Val: {X_val.shape[0]:,} baris | Test: {X_test.shape[0]:,} baris")
    print(f"  Class 1 ratio - Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")

    # 2. Definisikan Preprocessor (fit HANYA pada Train)
    # Gunakan drop='first' dan sparse_output=True untuk sinkronisasi dengan Spark dropLast=True
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True)
    scaler = StandardScaler(with_mean=False)

    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", scaler, num_cols)
        ]
    )

    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    print("\n[*] Menyesuaikan (fit) preprocessor pada Training Data...")
    preprocessor_scaled.fit(X_train)
    preprocessor_unscaled.fit(X_train)

    X_train_scaled = preprocessor_scaled.transform(X_train)
    X_val_scaled = preprocessor_scaled.transform(X_val)
    X_test_scaled = preprocessor_scaled.transform(X_test)

    X_train_unscaled = preprocessor_unscaled.transform(X_train)
    X_val_unscaled = preprocessor_unscaled.transform(X_val)
    X_test_unscaled = preprocessor_unscaled.transform(X_test)

    print(f"  Dimensi Vektor Fitur: {X_train_scaled.shape[1]} fitur setelah One-Hot Encoding")

    models_config = {
        "Scikit-Learn Logistic Regression": {
            "model": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "preprocessor": preprocessor_scaled
        },
        "Scikit-Learn Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=100, max_depth=15, class_weight="balanced",
                random_state=42, n_jobs=-1
            ),
            "X_train": X_train_unscaled,
            "X_val": X_val_unscaled,
            "X_test": X_test_unscaled,
            "preprocessor": preprocessor_unscaled
        }
    }

    results = {}
    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None
    best_preprocessor = None

    for name, cfg in models_config.items():
        print(f"\n--- [TRAINING: {name}] ---")
        clf = cfg["model"]
        X_tr = cfg["X_train"]
        X_v = cfg["X_val"]
        X_te = cfg["X_test"]

        # A. Ukur Training Time (Fit Time)
        t_start_train = time.time()
        clf.fit(X_tr, y_train)
        fit_time = time.time() - t_start_train
        print(f"  [+] Fit Time: {fit_time:.4f} detik")

        # B. Threshold Tuning pada Validation Set (2022)
        val_probs = clf.predict_proba(X_v)[:, 1]
        best_thresh = 0.50
        best_val_f1 = -1.0

        for thresh in np.linspace(0.30, 0.70, 41):
            val_preds_th = (val_probs >= thresh).astype(int)
            f1_th = f1_score(y_val, val_preds_th, pos_label=1, zero_division=0)
            if f1_th > best_val_f1:
                best_val_f1 = f1_th
                best_thresh = float(thresh)

        print(f"  [+] Optimal Validation Threshold: {best_thresh:.3f} (Val F1: {best_val_f1:.4f})")

        # C. Evaluasi Mutlak pada Test Set (2023)
        t_start_inf = time.time()
        test_probs = clf.predict_proba(X_te)[:, 1]
        test_preds = (test_probs >= best_thresh).astype(int)
        inference_time = time.time() - t_start_inf

        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds, pos_label=1, zero_division=0)
        rec = recall_score(y_test, test_preds, pos_label=1, zero_division=0)
        f1_c1 = f1_score(y_test, test_preds, pos_label=1, zero_division=0)
        f1_macro = f1_score(y_test, test_preds, average="macro", zero_division=0)
        roc_auc = roc_auc_score(y_test, test_probs)
        pr_auc = average_precision_score(y_test, test_probs)
        cm = confusion_matrix(y_test, test_preds).tolist()

        print("  [+] Hasil Evaluasi Test Set (2023):")
        print(f"      Accuracy:        {acc:.4f}")
        print(f"      Precision (C1):  {prec:.4f}")
        print(f"      Recall (C1):     {rec:.4f}")
        print(f"      F1-Score (C1):   {f1_c1:.4f}")
        print(f"      F1-Score (Macro):{f1_macro:.4f}")
        print(f"      ROC-AUC:         {roc_auc:.4f}")
        print(f"      PR-AUC:          {pr_auc:.4f}")
        print(f"      Confusion Matrix:[TN, FP], [FN, TP] = {cm}")
        print(f"      Inference Time:  {inference_time:.4f} detik")

        results[name] = {
            "Framework": "Scikit-Learn",
            "Model Name": name,
            "Accuracy": round(acc, 4),
            "Precision_Class1": round(prec, 4),
            "Recall_Class1": round(rec, 4),
            "F1_Class1": round(f1_c1, 4),
            "F1_Macro": round(f1_macro, 4),
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Fit_Time_Seconds": round(fit_time, 4),
            "Inference_Time_Seconds": round(inference_time, 4),
            "Optimal_Threshold": round(best_thresh, 3),
            "Confusion_Matrix": cm
        }

        if f1_c1 > best_f1:
            best_f1 = f1_c1
            best_model_name = name
            best_model_obj = clf
            best_preprocessor = cfg["preprocessor"]

    # Simpan Best Model untuk Streamlit
    print(f"\n[✓] Model Terbaik Terpilih: {best_model_name} (Test F1: {best_f1:.4f})")
    model_path = os.path.join(artifacts_dir, "best_model.joblib")
    preproc_path = os.path.join(artifacts_dir, "preprocessor.joblib")

    joblib.dump(best_model_obj, model_path)
    joblib.dump(best_preprocessor, preproc_path)
    print(f"  [+] Menyimpan model ke: {model_path}")
    print(f"  [+] Menyimpan preprocessor ke: {preproc_path}")

    # Simpan hasil sementara
    res_path = os.path.join(artifacts_dir, "sklearn_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [+] Menyimpan hasil evaluasi ke: {res_path}")

    return results


if __name__ == "__main__":
    run_sklearn_pipeline()
