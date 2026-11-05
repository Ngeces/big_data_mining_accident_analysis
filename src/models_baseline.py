"""
models_baseline.py
Implementasi model Scikit-Learn (Logistic Regression & Random Forest)
menggunakan partisi temporal (Train 2018-2021, Val 2022, Test 2023).
Threshold tuning DAN pemilihan model terbaik hanya memakai validation set;
test set dipakai sekali untuk evaluasi akhir.
"""

import os
import time
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import (
    PROCESSED_DIR, ARTIFACTS_DIR, CAT_COLS, NUM_COLS, TARGET_COL,
    RANDOM_SEED, LR_C, LR_MAX_ITER, RF_N_TREES, RF_MAX_DEPTH, VAL_YEAR,
)
from evaluation import tune_threshold, classification_metrics, majority_baseline


def load_partition(data_dir, name):
    """Kategori dibaca sebagai string tanpa konversi 'NA' -> NaN."""
    df = pd.read_csv(
        os.path.join(data_dir, f"{name}.csv"),
        dtype={c: str for c in CAT_COLS}, keep_default_na=False,
    )
    for col in NUM_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_preprocessor(scale_numeric):
    # drop=None + handle_unknown="ignore" setara dengan Spark StringIndexer(handleInvalid="keep")
    # + OneHotEncoder(dropLast=True): semua kategori nyata dipertahankan, kategori baru -> vektor nol.
    ohe = OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=True)
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(transformers=[("cat", ohe, CAT_COLS), ("num", num, NUM_COLS)])


def run_sklearn_pipeline(data_dir=PROCESSED_DIR, artifacts_dir=ARTIFACTS_DIR):
    print("=" * 60)
    print("[*] MEMULAI TRAINING DAN EVALUASI SCIKIT-LEARN (BASELINE)")
    print("=" * 60)
    os.makedirs(artifacts_dir, exist_ok=True)

    print("[*] Memuat data partisi...")
    train_df = load_partition(data_dir, "train")
    val_df = load_partition(data_dir, "val")
    test_df = load_partition(data_dir, "test")

    X_train, y_train = train_df[CAT_COLS + NUM_COLS], train_df[TARGET_COL].astype(int).values
    X_val, y_val = val_df[CAT_COLS + NUM_COLS], val_df[TARGET_COL].astype(int).values
    X_test, y_test = test_df[CAT_COLS + NUM_COLS], test_df[TARGET_COL].astype(int).values

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,} baris")
    print(f"  Class 1 ratio - Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")

    models_config = {
        "Scikit-Learn Logistic Regression": (
            LogisticRegression(C=LR_C, class_weight="balanced", max_iter=LR_MAX_ITER, random_state=RANDOM_SEED),
            make_preprocessor(scale_numeric=True),
        ),
        "Scikit-Learn Random Forest": (
            RandomForestClassifier(
                n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1,
            ),
            make_preprocessor(scale_numeric=False),  # pohon keputusan invarian terhadap skala
        ),
    }

    results = {}
    best = {"val_f1": -1.0}

    for name, (clf, preprocessor) in models_config.items():
        print(f"\n--- [TRAINING: {name}] ---")
        preprocessor.fit(X_train)  # fit HANYA pada train
        X_tr = preprocessor.transform(X_train)
        X_v = preprocessor.transform(X_val)
        X_te = preprocessor.transform(X_test)
        print(f"  Dimensi vektor fitur: {X_tr.shape[1]}")

        t0 = time.perf_counter()
        clf.fit(X_tr, y_train)
        fit_time = time.perf_counter() - t0
        print(f"  [+] Fit Time: {fit_time:.4f} detik")

        val_probs = clf.predict_proba(X_v)[:, 1]
        best_thresh, best_val_f1 = tune_threshold(y_val, val_probs)
        print(f"  [+] Threshold optimal (Val {VAL_YEAR}): {best_thresh:.2f} (Val F1: {best_val_f1:.4f})")

        t0 = time.perf_counter()
        test_probs = clf.predict_proba(X_te)[:, 1]
        inference_time = time.perf_counter() - t0

        metrics = classification_metrics(y_test, test_probs, best_thresh)
        print("  [+] Hasil Evaluasi Test Set:")
        for k, v in metrics.items():
            print(f"      {k:<18}: {v}")

        results[name] = {
            "Framework": "Scikit-Learn",
            "Model Name": name,
            **metrics,
            "Val_F1_Class1": round(best_val_f1, 4),
            "Fit_Time_Seconds": round(fit_time, 4),
            "Inference_Time_Seconds": round(inference_time, 4),
            "Optimal_Threshold": round(best_thresh, 2),
        }

        # Model terbaik dipilih berdasarkan VALIDATION F1, bukan test F1.
        if best_val_f1 > best["val_f1"]:
            best = {"val_f1": best_val_f1, "name": name, "model": clf,
                    "preprocessor": preprocessor, "threshold": best_thresh}

    results["Majority Class Baseline"] = majority_baseline(y_train, y_test)

    print(f"\n[✓] Model terpilih (berdasarkan Val F1): {best['name']} (Val F1: {best['val_f1']:.4f})")
    joblib.dump(best["model"], os.path.join(artifacts_dir, "best_model.joblib"))
    joblib.dump(best["preprocessor"], os.path.join(artifacts_dir, "preprocessor.joblib"))

    ohe = best["preprocessor"].named_transformers_["cat"]
    model_metadata = {
        "model_name": best["name"],
        "selection_metric": "F1_Class1 pada validation set",
        "val_f1_class1": round(best["val_f1"], 4),
        "threshold": round(best["threshold"], 2),
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "categories": {col: [str(c) for c in cats] for col, cats in zip(CAT_COLS, ohe.categories_)},
    }
    with open(os.path.join(artifacts_dir, "model_metadata.json"), "w") as f:
        json.dump(model_metadata, f, indent=2)

    with open(os.path.join(artifacts_dir, "sklearn_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [+] Artefak disimpan di {artifacts_dir}/ (best_model, preprocessor, model_metadata, sklearn_results)")

    return results


if __name__ == "__main__":
    run_sklearn_pipeline()
