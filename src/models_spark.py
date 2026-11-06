"""
models_spark.py
Implementasi model Apache Spark MLlib (Logistic Regression & Random Forest)
menggunakan partisi yang persis sama dengan Scikit-Learn (train.csv, val.csv, test.csv).

Aturan perbandingan adil dengan Scikit-Learn:
- Label, fitur, partisi, class weight ("balanced"), dan hyperparameter sama (lihat config.py).
- One-hot: StringIndexer(handleInvalid="keep") + OneHotEncoder(dropLast=True) membuang bucket
  "unknown" saja, setara dengan sklearn OneHotEncoder(drop=None, handle_unknown="ignore").
- Logistic Regression: hanya fitur numerik yang distandardisasi (seperti sklearn), standardization=False,
  regParam = 1 / (C * jumlah bobot train) agar setara dengan sklearn C.
- Threshold dipilih pada validation set, dan metrik dihitung dengan evaluation.py yang sama.
"""

import os
import time
import json

from config import (
    PROCESSED_DIR, ARTIFACTS_DIR, CAT_COLS, NUM_COLS, TARGET_COL,
    RANDOM_SEED, LR_C, LR_MAX_ITER, RF_N_TREES, RF_MAX_DEPTH, java_home_fallback,
)
from evaluation import tune_threshold, classification_metrics

java_home_fallback()

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler  # noqa: E402
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402


def read_partition(spark, data_dir, name):
    """Baca semua kolom sebagai string (tanpa inferSchema), lalu cast kolom numerik secara eksplisit."""
    sdf = spark.read.csv(os.path.join(data_dir, f"{name}.csv"), header=True, inferSchema=False)
    for col_name in NUM_COLS + [TARGET_COL]:
        sdf = sdf.withColumn(col_name, F.col(col_name).cast("double"))
    return sdf


def collect_probs(pred_sdf):
    """Ambil label dan probabilitas Class 1 ke driver (ukuran val/test ~6 ribu baris)."""
    pdf = pred_sdf.select(
        F.col(TARGET_COL).alias("y"),
        vector_to_array(F.col("probability"))[1].alias("p1"),
    ).toPandas()
    return pdf["y"].astype(int).values, pdf["p1"].values


def run_spark_pipeline(data_dir=PROCESSED_DIR, artifacts_dir=ARTIFACTS_DIR):
    print("=" * 60)
    print("[*] MEMULAI TRAINING DAN EVALUASI APACHE SPARK MLLIB")
    print("=" * 60)
    os.makedirs(artifacts_dir, exist_ok=True)

    print("[*] Menginisialisasi SparkSession (local[*])...")
    t0 = time.perf_counter()
    spark = (
        SparkSession.builder
        .appName("MSHA Mine Safety Analytics - Spark MLlib")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    spark_init_time = time.perf_counter() - t0
    print(f"[+] Spark Session aktif dalam {spark_init_time:.2f} detik.")

    train_sdf = read_partition(spark, data_dir, "train")
    val_sdf = read_partition(spark, data_dir, "val")
    test_sdf = read_partition(spark, data_dir, "test")

    # Class weight "balanced" (rumus yang sama dengan sklearn): n / (2 * n_kelas)
    total_train = train_sdf.count()
    pos_train = train_sdf.filter(F.col(TARGET_COL) == 1.0).count()
    neg_train = total_train - pos_train
    w0, w1 = total_train / (2.0 * neg_train), total_train / (2.0 * pos_train)
    print(f"  Train: {total_train:,} baris (Pos: {pos_train:,}, Neg: {neg_train:,}) | w0={w0:.3f}, w1={w1:.3f}")
    train_sdf = train_sdf.withColumn("classWeight", F.when(F.col(TARGET_COL) == 1.0, w1).otherwise(w0))

    # Preprocessing (fit hanya pada train)
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CAT_COLS]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in CAT_COLS],
        outputCols=[f"{c}_vec" for c in CAT_COLS],
        dropLast=True,
    )
    num_assembler = VectorAssembler(inputCols=NUM_COLS, outputCol="num_raw")
    num_scaler = StandardScaler(inputCol="num_raw", outputCol="num_scaled", withMean=True, withStd=True)
    cat_vecs = [f"{c}_vec" for c in CAT_COLS]
    assembler_lr = VectorAssembler(inputCols=cat_vecs + ["num_scaled"], outputCol="features_lr")
    assembler_rf = VectorAssembler(inputCols=cat_vecs + ["num_raw"], outputCol="features_rf")

    feature_pipeline = Pipeline(stages=indexers + [encoder, num_assembler, num_scaler, assembler_lr, assembler_rf])
    print("[*] Fit pipeline preprocessing Spark pada Training Data...")
    t0 = time.perf_counter()
    feature_model = feature_pipeline.fit(train_sdf)
    preprocess_fit_time = time.perf_counter() - t0

    train_prep = feature_model.transform(train_sdf).cache()
    val_prep = feature_model.transform(val_sdf).cache()
    test_prep = feature_model.transform(test_sdf).cache()
    for sdf in (train_prep, val_prep, test_prep):
        sdf.count()  # materialisasi cache

    sum_weights = w0 * neg_train + w1 * pos_train  # = total_train untuk bobot balanced
    spark_models = {
        "Spark MLlib Logistic Regression": LogisticRegression(
            featuresCol="features_lr", labelCol=TARGET_COL, weightCol="classWeight",
            maxIter=LR_MAX_ITER, regParam=1.0 / (LR_C * sum_weights),
            elasticNetParam=0.0, standardization=False,
        ),
        "Spark MLlib Random Forest": RandomForestClassifier(
            featuresCol="features_rf", labelCol=TARGET_COL, weightCol="classWeight",
            numTrees=RF_N_TREES, maxDepth=RF_MAX_DEPTH, featureSubsetStrategy="sqrt",
            seed=RANDOM_SEED,
        ),
    }

    spark_results = {}
    for name, algo in spark_models.items():
        print(f"\n--- [TRAINING: {name}] ---")
        t0 = time.perf_counter()
        model = algo.fit(train_prep)
        fit_time = time.perf_counter() - t0
        print(f"  [+] Fit Time: {fit_time:.4f} detik")

        y_val, val_probs = collect_probs(model.transform(val_prep))
        best_thresh, best_val_f1 = tune_threshold(y_val, val_probs)
        print(f"  [+] Threshold optimal (Val): {best_thresh:.2f} (Val F1: {best_val_f1:.4f})")

        t0 = time.perf_counter()
        test_pred = model.transform(test_prep).cache()
        test_pred.count()
        inference_time = time.perf_counter() - t0
        y_test, test_probs = collect_probs(test_pred)
        test_pred.unpersist()

        metrics = classification_metrics(y_test, test_probs, best_thresh)
        print("  [+] Hasil Evaluasi Test Set:")
        for k, v in metrics.items():
            print(f"      {k:<18}: {v}")

        spark_results[name] = {
            "Framework": "Apache Spark MLlib",
            "Model Name": name,
            **metrics,
            "Val_F1_Class1": round(best_val_f1, 4),
            "Fit_Time_Seconds": round(fit_time, 4),
            "Inference_Time_Seconds": round(inference_time, 4),
            "Optimal_Threshold": round(best_thresh, 2),
            "Spark_Init_Seconds": round(spark_init_time, 2),
            "Preprocess_Fit_Seconds": round(preprocess_fit_time, 2),
        }

    spark.stop()
    print("\n[✓] SparkSession dihentikan.")

    # Gabungkan dengan hasil Scikit-Learn
    sklearn_path = os.path.join(artifacts_dir, "sklearn_results.json")
    all_results = {}
    if os.path.exists(sklearn_path):
        with open(sklearn_path) as f:
            all_results = json.load(f)
    else:
        print("[!] sklearn_results.json belum ada; jalankan models_baseline.py terlebih dahulu.")
    all_results.update(spark_results)

    final_path = os.path.join(artifacts_dir, "benchmark_results.json")
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[✓] Hasil benchmark digabung ke: {final_path}")
    return all_results


if __name__ == "__main__":
    run_spark_pipeline()
