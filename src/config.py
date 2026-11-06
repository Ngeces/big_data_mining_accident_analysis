"""
config.py
Konfigurasi bersama untuk seluruh pipeline (data, Scikit-Learn, Spark MLlib, graph, dashboard).
Satu sumber kebenaran agar kedua framework memakai label, fitur, partisi, dan hyperparameter yang sama.
"""

import os

# ---------------------------------------------------------------------------
# Lokasi berkas
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("MSHA_DATA_DIR", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.environ.get("MSHA_ARTIFACTS_DIR", "artifacts")

# HDFS (single-node / pseudo-distributed). Contoh: export MSHA_HDFS_URI=hdfs://localhost:9000
# Jika kosong, lapisan storage menulis ke filesystem lokal (file:///) dengan tata letak yang sama.
HDFS_URI = os.environ.get("MSHA_HDFS_URI", "").rstrip("/")
HDFS_BASE_DIR = os.environ.get("MSHA_HDFS_BASE_DIR", "/user/msha")

ACCIDENTS_FILE = os.path.join(DATA_DIR, "Accidents.txt")
VIOLATIONS_FILE = os.path.join(DATA_DIR, "Violations.txt")
MINES_FILE = os.path.join(DATA_DIR, "Mines.txt")

# ---------------------------------------------------------------------------
# Label (DEGREE_INJURY_CD) -> IS_LOST_TIME_OR_SEVERE
# Pilihan B: kode 01-05 = Lost-Time / Severe, kode 00, 06, 10 = No Lost-Time.
# Kode 07, 08, 09, dan nilai kosong/'?' dieksklusi.
# ---------------------------------------------------------------------------
TARGET_COL = "IS_LOST_TIME_OR_SEVERE"
HIGH_SEVERITY_CODES = ["01", "02", "03", "04", "05"]
LOW_SEVERITY_CODES = ["00", "06", "10"]

# ---------------------------------------------------------------------------
# Fitur
# ---------------------------------------------------------------------------
CAT_COLS = ["SUBUNIT", "COAL_METAL_IND", "ACCIDENT_TYPE", "MINING_EQUIP", "ACTIVITY", "SHIFT_PERIOD"]
NUM_COLS = ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER"]
HIGH_CARD_COLS = ["MINING_EQUIP", "ACTIVITY"]
TOP_K_CATEGORIES = 15
MISSING_TOKEN = "MISSING"
OTHER_TOKEN = "OTHER"

# ---------------------------------------------------------------------------
# Partisi temporal
# ---------------------------------------------------------------------------
TRAIN_YEARS = (2018, 2021)
VAL_YEAR = 2022
TEST_YEAR = 2023

# Jendela graph analytics (Accidents & Violations harus memakai periode yang sama)
GRAPH_START_YR = 2020
GRAPH_END_YR = 2023

# ---------------------------------------------------------------------------
# Hyperparameter (disamakan untuk Scikit-Learn dan Spark MLlib)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
LR_C = 1.0               # Scikit-Learn C; Spark regParam = 1 / (C * n_train)
LR_MAX_ITER = 1000
RF_N_TREES = 100
RF_MAX_DEPTH = 15

# Grid ambang batas untuk tuning pada validation set
THRESHOLD_GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05 .. 0.95


def java_home_fallback():
    """Set JAVA_HOME hanya jika belum ada DAN path macOS default memang tersedia."""
    mac_jdk = "/Library/Java/JavaVirtualMachines/jdk-11.jdk/Contents/Home"
    if "JAVA_HOME" not in os.environ and os.path.isdir(mac_jdk):
        os.environ["JAVA_HOME"] = mac_jdk
