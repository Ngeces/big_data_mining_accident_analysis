"""
hadoop_storage.py
Implementasi Lapisan Penyimpanan Big Data Berbasis Arsitektur Hadoop HDFS:
1. Mengonversi data mentah MSHA ke format Hadoop Columnar Storage (Apache Parquet dengan Snappy Compression).
2. Menerapkan partisi temporal ala Hive/HDFS (partitionBy CAL_YR).
3. Mengelola metadata SQL Warehouse menggunakan Spark SQL.
4. Mengukur efisiensi kompresi dan kecepatan kueri analitik.

Catatan: pada mode lokal, fs.defaultFS = file:/// sehingga data ditulis ke filesystem lokal
dengan format dan tata letak partisi yang kompatibel dengan HDFS/Hive, bukan ke klaster HDFS.

"""

import os
import time
import json

from config import ACCIDENTS_FILE, DATA_DIR, ARTIFACTS_DIR, java_home_fallback

java_home_fallback()

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

NUMERIC_COLS = ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "DAYS_LOST", "DAYS_RESTRICT", "NO_INJURIES"]


def implement_hadoop_storage(
    raw_accidents_file=ACCIDENTS_FILE,
    warehouse_dir=os.path.join(DATA_DIR, "hdfs_warehouse"),
    artifacts_dir=ARTIFACTS_DIR,
):
    print("=" * 60)
    print("[*] MEMULAI IMPLEMENTASI HADOOP HDFS STORAGE LAYER")
    print("=" * 60)
    os.makedirs(warehouse_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Inisialisasi Spark Session dengan konfigurasi Hadoop Warehouse
    print("[*] Menginisialisasi SparkSession dengan Hadoop Client & Hive Support...")
    spark = SparkSession.builder \
        .appName("MSHA Hadoop Storage Layer") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.warehouse.dir", os.path.abspath(warehouse_dir)) \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Baca Data Mentah MSHA
    print(f"[*] Membaca data mentah dari {raw_accidents_file}...")
    t0_read = time.time()
    # inferSchema=False: MINE_ID / DEGREE_INJURY_CD tetap string (leading zero tidak hilang).
    raw_df = spark.read.csv(
        raw_accidents_file,
        sep="|",
        header=True,
        inferSchema=False,
        quote='"',
        encoding="ISO-8859-1",
    )
    raw_df = raw_df.withColumn("CAL_YR", F.col("CAL_YR").cast("int"))
    for c in NUMERIC_COLS:
        if c in raw_df.columns:
            raw_df = raw_df.withColumn(c, F.col(c).cast("double"))
    read_time = time.time() - t0_read
    raw_count = raw_df.count()
    print(f"[+] Total data terbaca: {raw_count:,} baris dalam {read_time:.2f} detik.")

    # Ukuran file mentah di disk
    raw_size_bytes = os.path.getsize(raw_accidents_file)
    raw_size_mb = raw_size_bytes / (1024 * 1024)
    print(f"  Ukuran File Mentah (CSV/TXT): {raw_size_mb:.2f} MB")

    # 3. Tulis ke Hadoop Parquet Storage Terpartisi (Hierarki HDFS)
    target_parquet_path = os.path.join(warehouse_dir, "msha_accidents")
    print(f"\n[*] Menulis data ke Hadoop Columnar Storage (Parquet + Snappy) terpartisi per CAL_YR...")
    print(f"  Target Direktori HDFS Warehouse: {target_parquet_path}")

    # Semua tahun ditulis agar perbandingan ukuran dengan file mentah adil (isi data sama).
    filtered_df = raw_df.filter(F.col("CAL_YR").isNotNull())
    dropped_null_year = raw_count - filtered_df.count()
    print(f"  Baris tanpa CAL_YR valid (tidak ditulis): {dropped_null_year:,}")

    t0_write = time.time()
    filtered_df.write \
        .partitionBy("CAL_YR") \
        .mode("overwrite") \
        .parquet(target_parquet_path)
    write_time = time.time() - t0_write
    print(f"[+] Berhasil menulis ke format Hadoop Parquet dalam {write_time:.2f} detik!")

    # 4. Hitung Ukuran Direktori Hadoop Storage
    parquet_total_size = 0
    partition_dirs = []
    for root, dirs, files in os.walk(target_parquet_path):
        for f in files:
            if f.endswith(".parquet"):
                fp = os.path.join(root, f)
                parquet_total_size += os.path.getsize(fp)
        for d in dirs:
            if d.startswith("CAL_YR="):
                partition_dirs.append(d)

    parquet_size_mb = parquet_total_size / (1024 * 1024)
    compression_ratio = (1 - (parquet_size_mb / raw_size_mb)) * 100

    print("\n--- [EFISIENSI PENYIMPANAN HADOOP VS RAW CSV] ---")
    print(f"  Ukuran Mentah (CSV/TXT):         {raw_size_mb:.2f} MB")
    print(f"  Ukuran Hadoop Parquet (Snappy):  {parquet_size_mb:.2f} MB")
    print(f"  Efisiensi Kompresi Storage:      {compression_ratio:.2f}% hemat disk!")
    print(f"  Partisi HDFS Terbentuk:          {len(partition_dirs)} partisi tahun ({sorted(partition_dirs)[:3]} ...)")

    # 5. Registrasi sebagai Tabel Spark SQL / Hive Metastore
    print("\n[*] Mendaftarkan direktori Hadoop Parquet sebagai Tabel SQL Warehouse...")
    parquet_df = spark.read.parquet(target_parquet_path)
    parquet_df.createOrReplaceTempView("msha_accidents")

    # Jalankan Contoh Kueri Analitik SQL (Agregasi Tren Insiden & Cedera Parah)
    print("[*] Mengeksekusi Kueri Analitik OLAP di atas Hadoop Storage...")
    t0_sql = time.time()
    sql_query = """
        SELECT 
            CAL_YR AS Tahun,
            COUNT(*) AS Total_Insiden,
            SUM(CASE WHEN DEGREE_INJURY_CD IN ('01', '02', '03', '04', '05') THEN 1 ELSE 0 END) AS Insiden_Lost_Time_Severe,
            ROUND(AVG(TOT_EXPER), 2) AS Rata_Rata_Pengalaman_Tahun
        FROM msha_accidents
        WHERE CAL_YR BETWEEN 2018 AND 2023
        GROUP BY CAL_YR
        ORDER BY CAL_YR DESC
    """
    res_df = spark.sql(sql_query)
    query_time = time.time() - t0_sql

    print(f"[+] Kueri SQL Berhasil Selesai dalam {query_time:.3f} detik!")
    pdf_res = res_df.toPandas()
    print("\nHasil Kueri Analitik SQL di atas Hadoop Warehouse:")
    print(pdf_res.to_string(index=False))

    # 6. Simpan Metadata Benchmark Hadoop Storage ke Artifacts
    summary_data = {
        "storage_technology": "Apache Parquet (format Hadoop) dengan partisi Hive-style, filesystem lokal (file:///)",
        "compression_codec": "Snappy Columnar",
        "raw_csv_size_mb": round(raw_size_mb, 2),
        "hadoop_parquet_size_mb": round(parquet_size_mb, 2),
        "storage_savings_pct": round(compression_ratio, 2),
        "write_time_seconds": round(write_time, 2),
        "olap_query_time_seconds": round(query_time, 3),
        "num_partitions": len(partition_dirs),
        "rows_written": int(raw_count - dropped_null_year),
        "rows_dropped_null_year": int(dropped_null_year),
        "partition_scheme": "CAL_YR=YYYY",
        "sample_query_results": pdf_res.to_dict(orient="records")
    }

    summary_file = os.path.join(artifacts_dir, "hadoop_storage_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[✓] Berhasil menyimpan ringkasan Hadoop Storage ke: {summary_file}")

    spark.stop()
    return summary_data


if __name__ == "__main__":
    implement_hadoop_storage()
