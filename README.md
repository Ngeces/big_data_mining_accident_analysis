# MSHA Mine Accident Analysis (COMP8035041 Big Data Analytics)

Klasifikasi tingkat keparahan cedera pada insiden tambang yang dilaporkan (MSHA Accidents),
perbandingan Scikit-Learn vs Spark MLlib, graph analytics Mine–Standard–Accident Type,
dan dashboard Streamlit.

## Dataset

Dataset tidak disimpan di repository karena ukurannya besar (±2 GB). Unduh dari
[MSHA Open Government Data](https://arlweb.msha.gov/opengovernmentdata/ogimsha.asp),
ekstrak, lalu letakkan di folder `data/`:

```
data/
├── Accidents.txt
├── Violations.txt
└── Mines.txt
```

Folder lain bisa dipakai dengan variabel lingkungan `MSHA_DATA_DIR=/path/ke/data`.
Catat tanggal unduh di laporan, karena MSHA memperbarui file setiap minggu.

## Instalasi

Butuh Python 3.10+ dan Java 11/17 (untuk PySpark).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Menjalankan pipeline

Dari root repository:

```bash
python src/run_all.py
```

Urutan tahap (bisa juga dijalankan satu per satu dengan `python src/<file>.py`):

| Tahap | File | Output utama di `artifacts/` |
|---|---|---|
| 1 | `data_pipeline.py` | `data/processed/{train,val,test}.csv`, `preprocessing_metadata.json` |
| 2 | `models_baseline.py` | `sklearn_results.json`, `best_model.joblib`, `preprocessor.joblib`, `model_metadata.json` |
| 3 | `models_spark.py` | `benchmark_results.json` (gabungan sklearn + Spark) |
| 4 | `hadoop_storage.py` | `hadoop_storage_summary.json`, `data/hdfs_warehouse/` |
| 5 | `graph_analytics.py` | `graph_nodes_summary.csv`, `graph_top_edges.json`, `mine_accident_closeness.json` |
| 6 | `geo_and_viz_pipeline.py` | `mine_geo_accidents.csv`, `graph_network_layout.json`, `accident_annual_trend.csv` |

Tanpa Java: `python src/run_all.py --skip-spark --skip-storage`.

### Google Colab

```python
!git clone https://github.com/Ngeces/big_data_mining_accident_analysis.git
%cd big_data_mining_accident_analysis
!pip -q install -r requirements.txt
# salin Accidents.txt, Violations.txt, Mines.txt ke ./data/ (mis. dari Google Drive)
!python src/run_all.py
```

## Dashboard

```bash
streamlit run app.py
```

Dashboard hanya membaca artefak di `artifacts/`; tidak ada training saat aplikasi berjalan.

## Metodologi (ringkas)

Semua parameter ada di `src/config.py`.

- **Label** `IS_LOST_TIME_OR_SEVERE`: kode `DEGREE_INJURY_CD` 01–05 = 1; 00, 06, 10 = 0; 07, 08, 09, dan kosong dieksklusi.
- **Partisi temporal**: train 2018–2021, validation 2022, test 2023.
- **Tanpa leakage**: median, Top-15 kategori, encoder, dan scaler di-fit hanya pada train.
  Kolom pasca-kejadian (`DAYS_LOST`, `NATURE_INJURY`, dll.) tidak dibaca.
- **Pemilihan model dan threshold** hanya memakai validation set; test set dipakai sekali.
- **Perbandingan adil sklearn vs Spark**: fitur, encoding, class weight, hyperparameter,
  dan fungsi metrik (`src/evaluation.py`) sama untuk kedua framework.
- **Graph**: jendela 2020–2023 untuk Accidents dan Violations; node `mine::`, `standard::`, `accident_type::`.
  Weighted degree menunjukkan frekuensi laporan, bukan risiko yang dinormalisasi terhadap jam kerja.
