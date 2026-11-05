"""
run_all.py
Menjalankan seluruh pipeline secara berurutan dari root repository:

    python src/run_all.py            # semua tahap
    python src/run_all.py --skip-spark --skip-storage   # tanpa Java/Spark

Urutan ini penting: models_spark membutuhkan sklearn_results.json,
dan geo_and_viz_pipeline membutuhkan artefak graph_analytics.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ACCIDENTS_FILE, VIOLATIONS_FILE, MINES_FILE  # noqa: E402


def check_inputs():
    missing = [p for p in (ACCIDENTS_FILE, VIOLATIONS_FILE, MINES_FILE) if not os.path.isfile(p)]
    if missing:
        sys.exit(
            "Dataset belum tersedia: " + ", ".join(missing)
            + "\nUnduh dari MSHA Open Government Data dan letakkan di folder data/ (lihat README.md)."
        )


def step(name, fn):
    print(f"\n######## {name} ########")
    t0 = time.perf_counter()
    fn()
    print(f"######## {name} selesai dalam {time.perf_counter() - t0:.1f} detik ########")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-spark", action="store_true", help="Lewati Spark MLlib")
    parser.add_argument("--skip-storage", action="store_true", help="Lewati konversi Parquet")
    args = parser.parse_args()

    check_inputs()

    from data_pipeline import load_and_preprocess_msha_accidents
    from models_baseline import run_sklearn_pipeline
    from graph_analytics import run_graph_analytics
    from geo_and_viz_pipeline import generate_geo_and_network_artifacts

    step("1. Data pipeline", load_and_preprocess_msha_accidents)
    step("2. Scikit-Learn", run_sklearn_pipeline)
    if not args.skip_spark:
        from models_spark import run_spark_pipeline
        step("3. Spark MLlib", run_spark_pipeline)
    if not args.skip_storage:
        from hadoop_storage import implement_hadoop_storage
        step("4. Parquet storage", implement_hadoop_storage)
    step("5. Graph analytics", run_graph_analytics)
    step("6. Geo & visual network", generate_geo_and_network_artifacts)


if __name__ == "__main__":
    main()
