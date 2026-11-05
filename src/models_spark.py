"""
models_spark.py
Implementasi model Apache Spark MLlib (Logistic Regression & Random Forest)
menggunakan dataset partisi yang persis sama dengan Scikit-Learn (train.csv, val.csv, test.csv).
Menghitung metrik klasifikasi Class 1, ROC-AUC, PR-AUC, dan waktu komputasi.
"""

import os
import time
import json
import numpy as np
import pandas as pd

# Pastikan JAVA_HOME mengarah ke instalasi Java 11
if "JAVA_HOME" not in os.environ:
    os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-11.jdk/Contents/Home"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def run_spark_pipeline(
    data_dir="data/processed",
    artifacts_dir="artifacts"
):
    print("=" * 60)
    print("[*] MEMULAI TRAINING DAN EVALUASI APACHE SPARK MLLIB")
    print("=" * 60)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Inisialisasi Spark Session
    print("[*] Menginisialisasi SparkSession (local[*])...")
    t_spark_init = time.time()
    spark = SparkSession.builder \
        .appName("MSHA Mine Safety Analytics - Spark MLlib") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark_init_time = time.time() - t_spark_init
    print(f"[+] Spark Session aktif dalam {spark_init_time:.2f} detik.")

    # 1. Load Data Partisi
    print("[*] Membaca data partisi ke Spark DataFrame...")
    train_sdf = spark.read.csv(os.path.join(data_dir, "train.csv"), header=True, inferSchema=True)
    val_sdf = spark.read.csv(os.path.join(data_dir, "val.csv"), header=True, inferSchema=True)
    test_sdf = spark.read.csv(os.path.join(data_dir, "test.csv"), header=True, inferSchema=True)

    cat_cols = ["SUBUNIT", "COAL_METAL_IND", "ACCIDENT_TYPE", "MINING_EQUIP", "ACTIVITY", "SHIFT_PERIOD"]
    num_cols = ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER"]
    target_col = "IS_LOST_TIME_OR_SEVERE"

    # Pastikan tipe kolom numerik dan target Double
    for col_name in num_cols + [target_col]:
        train_sdf = train_sdf.withColumn(col_name, F.col(col_name).cast(DoubleType()))
        val_sdf = val_sdf.withColumn(col_name, F.col(col_name).cast(DoubleType()))
        test_sdf = test_sdf.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    total_train = train_sdf.count()
    pos_train = train_sdf.filter(F.col(target_col) == 1.0).count()
    neg_train = total_train - pos_train

    # Hitung bobot kelas (class weight)
    w0 = total_train / (2.0 * neg_train)
    w1 = total_train / (2.0 * pos_train)
    print(f"  Train: {total_train:,} baris (Pos: {pos_train:,}, Neg: {neg_train:,})")
    print(f"  Class Weights: w0={w0:.3f}, w1={w1:.3f}")

    def add_class_weight(df):
        return df.withColumn(
            "classWeight",
            F.when(F.col(target_col) == 1.0, w1).otherwise(w0)
        )

    train_sdf = add_class_weight(train_sdf)
    val_sdf = add_class_weight(val_sdf)
    test_sdf = add_class_weight(test_sdf)

    # 2. Pipeline Preprocessing Spark (StringIndexer + OneHotEncoder dropLast=True)
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat_cols
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat_cols],
        outputCols=[f"{c}_vec" for c in cat_cols],
        dropLast=True
    )
    assembler = VectorAssembler(
        inputCols=[f"{c}_vec" for c in cat_cols] + num_cols,
        outputCol="features_unscaled"
    )
    scaler = StandardScaler(
        inputCol="features_unscaled",
        outputCol="features_scaled",
        withStd=True,
        withMean=False
    )

    feature_pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler])
    print("\n[*] Menyesuaikan (fit) pipeline preprocessing Spark pada Training Data...")
    feature_model = feature_pipeline.fit(train_sdf)

    train_prep = feature_model.transform(train_sdf).cache()
    val_prep = feature_model.transform(val_sdf).cache()
    test_prep = feature_model.transform(test_sdf).cache()

    train_prep.count()  # trigger cache
    val_prep.count()
    test_prep.count()

    eval_roc = BinaryClassificationEvaluator(labelCol=target_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    eval_pr = BinaryClassificationEvaluator(labelCol=target_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    eval_acc = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
    eval_f1_c1 = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="fMeasureByLabel", metricLabel=1.0)
    eval_prec_c1 = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="precisionByLabel", metricLabel=1.0)
    eval_rec_c1 = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0)
    eval_f1_macro = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="f1")

    spark_models = {
        "Spark MLlib Logistic Regression": {
            "algo": LogisticRegression(
                featuresCol="features_scaled",
                labelCol=target_col,
                weightCol="classWeight",
                maxIter=100
            )
        },
        "Spark MLlib Random Forest": {
            "algo": RandomForestClassifier(
                featuresCol="features_unscaled",
                labelCol=target_col,
                weightCol="classWeight",
                numTrees=100,
                maxDepth=10,
                seed=42
            )
        }
    }

    spark_results = {}

    for name, cfg in spark_models.items():
        print(f"\n--- [TRAINING: {name}] ---")
        algo = cfg["algo"]

        # A. Ukur Training Time
        t0 = time.time()
        model = algo.fit(train_prep)
        fit_time = time.time() - t0
        print(f"  [+] Fit Time: {fit_time:.4f} detik")

        # B. Prediksi Test Set (2023)
        t_inf0 = time.time()
        test_preds = model.transform(test_prep).cache()
        test_preds.count()
        inf_time = time.time() - t_inf0

        roc_auc = eval_roc.evaluate(test_preds)
        pr_auc = eval_pr.evaluate(test_preds)
        acc = eval_acc.evaluate(test_preds)
        f1_c1 = eval_f1_c1.evaluate(test_preds)
        prec_c1 = eval_prec_c1.evaluate(test_preds)
        rec_c1 = eval_rec_c1.evaluate(test_preds)
        f1_macro = eval_f1_macro.evaluate(test_preds)

        # Confusion matrix
        cm_df = test_preds.groupBy(target_col, "prediction").count().toPandas()
        cm_dict = {(int(r[target_col]), int(r["prediction"])): int(r["count"]) for _, r in cm_df.iterrows()}
        tn = cm_dict.get((0, 0), 0)
        fp = cm_dict.get((0, 1), 0)
        fn = cm_dict.get((1, 0), 0)
        tp = cm_dict.get((1, 1), 0)

        print("  [+] Hasil Evaluasi Test Set (2023):")
        print(f"      Accuracy:        {acc:.4f}")
        print(f"      Precision (C1):  {prec_c1:.4f}")
        print(f"      Recall (C1):     {rec_c1:.4f}")
        print(f"      F1-Score (C1):   {f1_c1:.4f}")
        print(f"      F1-Score (Macro):{f1_macro:.4f}")
        print(f"      ROC-AUC:         {roc_auc:.4f}")
        print(f"      PR-AUC:          {pr_auc:.4f}")
        print(f"      Confusion Matrix:[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]")
        print(f"      Inference Time:  {inf_time:.4f} detik")

        spark_results[name] = {
            "Framework": "Apache Spark MLlib",
            "Model Name": name,
            "Accuracy": round(acc, 4),
            "Precision_Class1": round(prec_c1, 4),
            "Recall_Class1": round(rec_c1, 4),
            "F1_Class1": round(f1_c1, 4),
            "F1_Macro": round(f1_macro, 4),
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Fit_Time_Seconds": round(fit_time, 4),
            "Inference_Time_Seconds": round(inf_time, 4),
            "Optimal_Threshold": 0.50,
            "Confusion_Matrix": [[tn, fp], [fn, tp]]
        }

    # Hentikan Spark Session
    spark.stop()
    print("\n[✓] SparkSession dihentikan dengan rapi.")

    # Gabungkan dengan hasil Scikit-Learn
    sklearn_path = os.path.join(artifacts_dir, "sklearn_results.json")
    all_results = {}
    if os.path.exists(sklearn_path):
        with open(sklearn_path, "r") as f:
            all_results = json.load(f)

    all_results.update(spark_results)

    final_benchmark_path = os.path.join(artifacts_dir, "benchmark_results.json")
    with open(final_benchmark_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[✓] Berhasil menggabungkan hasil benchmark ke: {final_benchmark_path}")
    return all_results


if __name__ == "__main__":
    run_spark_pipeline()
