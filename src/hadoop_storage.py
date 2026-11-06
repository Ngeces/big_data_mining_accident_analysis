"""
hadoop_storage.py
Lapisan penyimpanan Big Data di HDFS (single-node / pseudo-distributed):

1. Mengunggah berkas mentah MSHA ke raw zone HDFS  (<base>/raw/).
2. Spark membaca Accidents.txt dari HDFS lalu menulis curated zone dalam Parquet + Snappy,
   terpartisi per tahun (<base>/warehouse/msha_accidents/CAL_YR=YYYY/).
3. Mendaftarkan Parquet sebagai tabel Spark SQL dan menjalankan kueri agregasi.
4. Mengukur ukuran penyimpanan, block size, dan replication factor melalui Hadoop FileSystem API.

Target HDFS diatur dengan variabel lingkungan MSHA_HDFS_URI (mis. hdfs://localhost:9000).
Jika tidak diatur, modul berjalan dengan filesystem lokal (file:///) memakai tata letak direktori yang sama,
dan ringkasan mencatat "filesystem_uri": "file:///" agar tidak tertukar dengan hasil HDFS.
"""

import os
import time
import json

from config import (
    ACCIDENTS_FILE, VIOLATIONS_FILE, MINES_FILE, DATA_DIR, ARTIFACTS_DIR,
    HDFS_URI, HDFS_BASE_DIR, HIGH_SEVERITY_CODES, java_home_fallback,
)

java_home_fallback()

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

NUMERIC_COLS = ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "DAYS_LOST", "DAYS_RESTRICT", "NO_INJURIES"]
RAW_FILES = [ACCIDENTS_FILE, VIOLATIONS_FILE, MINES_FILE]
MB = 1024 * 1024


def storage_roots():
    """(filesystem_uri, base_dir) untuk HDFS atau fallback lokal."""
    if HDFS_URI:
        return HDFS_URI, f"{HDFS_URI}{HDFS_BASE_DIR}"
    local_base = os.path.abspath(os.path.join(DATA_DIR, "hdfs_warehouse"))
    return "file:///", f"file://{local_base}"


class HadoopFS:
    """Pembungkus tipis Hadoop FileSystem API (lewat JVM Spark) untuk HDFS maupun file:///."""

    def __init__(self, spark, uri):
        jvm = spark.sparkContext._jvm
        self._Path = jvm.org.apache.hadoop.fs.Path
        conf = spark.sparkContext._jsc.hadoopConfiguration()
        self.fs = jvm.org.apache.hadoop.fs.FileSystem.get(jvm.java.net.URI(uri), conf)

    def path(self, p):
        return self._Path(p)

    def mkdirs(self, p):
        self.fs.mkdirs(self.path(p))

    def put(self, local_file, dest_dir):
        """Salin berkas lokal ke dest_dir (overwrite)."""
        self.fs.copyFromLocalFile(False, True, self.path(os.path.abspath(local_file)), self.path(dest_dir))
        return f"{dest_dir}/{os.path.basename(local_file)}"

    def size_bytes(self, p):
        return int(self.fs.getContentSummary(self.path(p)).getLength())

    def list_dirs(self, p, prefix=""):
        return sorted(
            s.getPath().getName() for s in self.fs.listStatus(self.path(p))
            if s.isDirectory() and s.getPath().getName().startswith(prefix)
        )

    def file_info(self, p):
        st = self.fs.getFileStatus(self.path(p))
        return {"block_size_mb": round(st.getBlockSize() / MB, 1), "replication": int(st.getReplication())}


def implement_hadoop_storage(artifacts_dir=ARTIFACTS_DIR):
    print("=" * 60)
    print("[*] MEMULAI LAPISAN PENYIMPANAN HDFS")
    print("=" * 60)
    os.makedirs(artifacts_dir, exist_ok=True)

    fs_uri, base = storage_roots()
    raw_dir, curated_path = f"{base}/raw", f"{base}/warehouse/msha_accidents"
    print(f"[*] Filesystem: {fs_uri}")
    print(f"    Raw zone    : {raw_dir}")
    print(f"    Curated zone: {curated_path}")
    if not HDFS_URI:
        print("[!] MSHA_HDFS_URI tidak diatur: memakai filesystem lokal (file:///), BUKAN HDFS.")

    builder = (
        SparkSession.builder
        .appName("MSHA HDFS Storage Layer")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.ui.enabled", "false")
    )
    if HDFS_URI:
        builder = builder.config("spark.hadoop.fs.defaultFS", HDFS_URI)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    hfs = HadoopFS(spark, fs_uri if HDFS_URI else "file:///")

    # 1. Raw zone: unggah berkas mentah apa adanya
    print("\n[*] Mengunggah berkas mentah ke raw zone...")
    hfs.mkdirs(raw_dir)
    raw_paths = {}
    t0 = time.perf_counter()
    for local in RAW_FILES:
        if not os.path.isfile(local):
            print(f"  [!] Lewati {local}: berkas tidak ditemukan")
            continue
        raw_paths[os.path.basename(local)] = hfs.put(local, raw_dir)
        print(f"  [+] {os.path.basename(local)} -> {raw_paths[os.path.basename(local)]}")
    upload_time = time.perf_counter() - t0
    raw_accidents = raw_paths[os.path.basename(ACCIDENTS_FILE)]
    raw_info = hfs.file_info(raw_accidents)
    raw_size_mb = hfs.size_bytes(raw_accidents) / MB
    raw_zone_mb = hfs.size_bytes(raw_dir) / MB
    print(f"  Block size: {raw_info['block_size_mb']} MB · replication: {raw_info['replication']}")

    # 2. Baca dari raw zone (semua kolom string agar MINE_ID / kode tidak kehilangan leading zero)
    print(f"\n[*] Spark membaca {raw_accidents} ...")
    t0 = time.perf_counter()
    raw_df = spark.read.csv(raw_accidents, sep="|", header=True, inferSchema=False, quote='"',
                            encoding="ISO-8859-1")
    raw_df = raw_df.withColumn("CAL_YR", F.col("CAL_YR").cast("int"))
    for c in NUMERIC_COLS:
        if c in raw_df.columns:
            raw_df = raw_df.withColumn(c, F.col(c).cast("double"))
    raw_count = raw_df.count()
    read_time = time.perf_counter() - t0
    print(f"[+] {raw_count:,} baris dibaca dalam {read_time:.2f} detik")

    # 3. Curated zone: Parquet + Snappy, partisi per tahun (semua tahun agar perbandingan ukuran adil)
    valid_df = raw_df.filter(F.col("CAL_YR").isNotNull())
    rows_written = valid_df.count()
    dropped_null_year = raw_count - rows_written
    print(f"\n[*] Menulis Parquet terpartisi CAL_YR ke {curated_path} (tanpa CAL_YR valid: {dropped_null_year:,})")
    t0 = time.perf_counter()
    valid_df.write.partitionBy("CAL_YR").mode("overwrite").parquet(curated_path)
    write_time = time.perf_counter() - t0

    partitions = hfs.list_dirs(curated_path, "CAL_YR=")
    parquet_size_mb = hfs.size_bytes(curated_path) / MB
    savings = (1 - parquet_size_mb / raw_size_mb) * 100
    print(f"  Raw Accidents.txt : {raw_size_mb:.2f} MB")
    print(f"  Parquet (Snappy)  : {parquet_size_mb:.2f} MB  ({savings:.2f}% lebih kecil)")
    print(f"  Partisi           : {len(partitions)} ({partitions[:2]} ... {partitions[-1:]})")

    # 4. Spark SQL di atas curated zone
    spark.read.parquet(curated_path).createOrReplaceTempView("msha_accidents")
    codes = ", ".join(f"'{c}'" for c in HIGH_SEVERITY_CODES)
    sql_query = f"""
        SELECT CAL_YR AS Tahun,
               COUNT(*) AS Total_Insiden,
               SUM(CASE WHEN DEGREE_INJURY_CD IN ({codes}) THEN 1 ELSE 0 END) AS Insiden_Lost_Time_Severe,
               ROUND(AVG(TOT_EXPER), 2) AS Rata_Rata_Pengalaman_Tahun
        FROM msha_accidents
        WHERE CAL_YR BETWEEN 2018 AND 2023
        GROUP BY CAL_YR
        ORDER BY CAL_YR DESC
    """
    t0 = time.perf_counter()
    pdf_res = spark.sql(sql_query).toPandas()
    query_time = time.perf_counter() - t0
    print(f"\n[+] Kueri Spark SQL selesai dalam {query_time:.3f} detik")
    print(pdf_res.to_string(index=False))

    summary = {
        "storage_technology": "HDFS" if HDFS_URI else "Filesystem lokal (file:///), tata letak sama dengan HDFS",
        "filesystem_uri": fs_uri,
        "raw_zone_path": raw_dir,
        "curated_zone_path": curated_path,
        "raw_files": raw_paths,
        "block_size_mb": raw_info["block_size_mb"],
        "replication_factor": raw_info["replication"],
        "compression_codec": "Snappy (Parquet)",
        "raw_zone_total_mb": round(raw_zone_mb, 2),
        "raw_csv_size_mb": round(raw_size_mb, 2),
        "hadoop_parquet_size_mb": round(parquet_size_mb, 2),
        "storage_savings_pct": round(savings, 2),
        "upload_time_seconds": round(upload_time, 2),
        "read_time_seconds": round(read_time, 2),
        "write_time_seconds": round(write_time, 2),
        "olap_query_time_seconds": round(query_time, 3),
        "num_partitions": len(partitions),
        "partition_scheme": "CAL_YR=YYYY",
        "rows_written": int(rows_written),
        "rows_dropped_null_year": int(dropped_null_year),
        "sample_query_results": pdf_res.to_dict(orient="records"),
    }
    path = os.path.join(artifacts_dir, "hadoop_storage_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Ringkasan disimpan ke: {path}")

    spark.stop()
    return summary


if __name__ == "__main__":
    implement_hadoop_storage()
