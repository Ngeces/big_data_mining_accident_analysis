"""
data_pipeline.py
Pipeline pemrosesan data resmi MSHA Accidents dengan metodologi kedap kebocoran data (leakage-free).
"""

import os
import json
import pandas as pd

from config import (
    ACCIDENTS_FILE, PROCESSED_DIR, ARTIFACTS_DIR, TARGET_COL,
    HIGH_SEVERITY_CODES, LOW_SEVERITY_CODES, CAT_COLS, NUM_COLS,
    HIGH_CARD_COLS, TOP_K_CATEGORIES, MISSING_TOKEN, OTHER_TOKEN,
    TRAIN_YEARS, VAL_YEAR, TEST_YEAR,
)

# Kolom yang secara sengaja TIDAK dipakai sebagai prediktor (didokumentasikan untuk laporan).
LEAKAGE_COLS = [
    "DAYS_LOST", "DAYS_RESTRICT", "SCHEDULE_CHARGE",
    "RETURN_TO_WORK_DT", "TRANS_TERM", "CLOSED_DOC_NO",
    "INVEST_BEGIN_DT", "DEGREE_INJURY_CD", "DEGREE_INJURY",
    "NATURE_INJURY_CD", "NATURE_INJURY", "INJ_BODY_PART_CD", "INJ_BODY_PART",
    "INJURY_SOURCE_CD", "INJURY_SOURCE", "IMMED_NOTIFY_CD", "IMMED_NOTIFY",
    "NARRATIVE",
]

RAW_CAT_COLS = [c for c in CAT_COLS if c != "SHIFT_PERIOD"]
META_COLS = ["MINE_ID", "DOCUMENT_NO", "CAL_YR"]
READ_COLS = META_COLS + ["DEGREE_INJURY_CD", "SHIFT_BEGIN_TIME"] + RAW_CAT_COLS + NUM_COLS


def parse_shift_time(time_val):
    """Mengonversi nilai waktu HHMM menjadi kategori shift kerja."""
    if pd.isna(time_val):
        return "UNKNOWN"
    val_str = str(time_val).strip().replace('"', "").replace("'", "")
    if not val_str.isdigit() or len(val_str) > 4:
        return "UNKNOWN"
    val_str = val_str.zfill(4)
    hour, minute = int(val_str[:2]), int(val_str[2:])
    if hour > 24 or minute > 59:
        return "UNKNOWN"
    if 6 <= hour < 14:
        return "DAY"
    if 14 <= hour < 22:
        return "EVENING"
    return "NIGHT"  # 22:00-06:00 (termasuk 2400)


def normalize_injury_code(series):
    """'1' -> '01', '1.0' -> '01'; nilai non-numerik dibiarkan apa adanya."""
    s = series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    return s.where(~s.str.fullmatch(r"\d+", na=False), s.str.zfill(2))


def clean_strings(df, cols):
    """Strip spasi/kutip; string kosong menjadi NaN (bukan string 'nan')."""
    for col in cols:
        s = df[col].astype("string").str.strip().str.replace('"', "", regex=False)
        df[col] = s.mask((s == "").fillna(False))
    return df


def load_and_preprocess_msha_accidents(
    file_path=ACCIDENTS_FILE,
    output_dir=PROCESSED_DIR,
    artifacts_dir=ARTIFACTS_DIR,
):
    """
    Membaca berkas Accidents.txt MSHA asli, melakukan profiling label,
    pemetaan target biner IS_LOST_TIME_OR_SEVERE, pencegahan leakage,
    serta partisi data temporal: Train (2018-2021), Val (2022), Test (2023).
    """
    print(f"[*] Membaca data MSHA dari {file_path}...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Semua kolom dibaca sebagai string agar MINE_ID / kode tidak kehilangan leading zero.
    df = pd.read_csv(
        file_path, sep="|", usecols=READ_COLS, dtype=str,
        quotechar='"', encoding="latin1", low_memory=False,
    )
    print(f"[+] Total data mentah terbaca: {len(df):,} baris")

    df = clean_strings(df, READ_COLS)
    df["DEGREE_INJURY_CD"] = normalize_injury_code(df["DEGREE_INJURY_CD"])
    df["CAL_YR"] = pd.to_numeric(df["CAL_YR"], errors="coerce").astype("Int64")

    # 1. Profiling Label DEGREE_INJURY_CD
    print("\n--- [PROFILING LABEL: DEGREE_INJURY_CD] ---")
    deg_counts = df["DEGREE_INJURY_CD"].value_counts(dropna=False)
    label_profile = []
    for code, count in deg_counts.items():
        pct = count / len(df) * 100
        print(f"  Kode {str(code):>4}: {count:>8,} baris ({pct:5.2f}%)")
        label_profile.append({"code": None if pd.isna(code) else str(code), "rows": int(count), "pct": round(pct, 2)})

    # 2. Pemetaan Target (Pilihan B: Lost-Time / Severe Injury Outcome)
    valid_codes = HIGH_SEVERITY_CODES + LOW_SEVERITY_CODES
    initial_len = len(df)
    df = df[df["DEGREE_INJURY_CD"].isin(valid_codes)].copy()
    excluded_len = initial_len - len(df)
    print(f"\n[+] Data valid target dipertahankan: {len(df):,} baris ({len(df)/initial_len*100:.2f}%)")
    print(f"[-] Data dieksklusi (07, 08, 09, missing/lainnya): {excluded_len:,} baris ({excluded_len/initial_len*100:.2f}%)")

    df[TARGET_COL] = df["DEGREE_INJURY_CD"].isin(HIGH_SEVERITY_CODES).astype(int)

    # 3. Filter Jendela Waktu Penelitian
    start_yr, end_yr = TRAIN_YEARS[0], TEST_YEAR
    df_window = df[df["CAL_YR"].between(start_yr, end_yr)].copy()
    print(f"\n[+] Data dalam jendela {start_yr}-{end_yr}: {len(df_window):,} baris")

    dist = df_window[TARGET_COL].value_counts()
    print("[+] Distribusi Target (dalam jendela):")
    print(f"  Class 0 (No Lost-Time):       {dist.get(0, 0):,} ({dist.get(0, 0)/len(df_window)*100:.2f}%)")
    print(f"  Class 1 (Lost-Time / Severe): {dist.get(1, 0):,} ({dist.get(1, 0)/len(df_window)*100:.2f}%)")

    # 4. Transformasi Fitur Waktu & Sanitasi Kolom
    print("[*] Melakukan rekayasa fitur waktu (Shift categorization)...")
    df_window["SHIFT_PERIOD"] = df_window["SHIFT_BEGIN_TIME"].apply(parse_shift_time)
    for col in NUM_COLS:
        df_window[col] = pd.to_numeric(df_window[col], errors="coerce")
    for col in RAW_CAT_COLS:
        df_window[col] = df_window[col].fillna(MISSING_TOKEN)

    # 5. Hanya kolom prediktor yang aman yang dipertahankan (kolom leakage tidak pernah dibaca).
    print(f"[!] {len(LEAKAGE_COLS)} kolom leakage tidak dipakai sebagai prediktor.")
    df_clean = df_window[META_COLS + [TARGET_COL] + CAT_COLS + NUM_COLS].copy()

    # 6. Partisi Temporal
    train_df = df_clean[df_clean["CAL_YR"].between(*TRAIN_YEARS)].copy()
    val_df = df_clean[df_clean["CAL_YR"] == VAL_YEAR].copy()
    test_df = df_clean[df_clean["CAL_YR"] == TEST_YEAR].copy()

    print("\n[+] Partisi Data Temporal:")
    print(f"  Training Set   ({TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}): {len(train_df):,} baris")
    print(f"  Validation Set ({VAL_YEAR}):       {len(val_df):,} baris")
    print(f"  Test Set       ({TEST_YEAR}):       {len(test_df):,} baris")

    # 7. Training-Only Parameter Fitting
    print(f"\n[*] Menghitung median dan Top-{TOP_K_CATEGORIES} kategori hanya dari Training Set...")
    medians = {col: float(train_df[col].median()) for col in NUM_COLS}
    print(f"  Median Pengalaman (Train): {medians}")
    for part in (train_df, val_df, test_df):
        for col in NUM_COLS:
            part[col] = part[col].fillna(medians[col])

    top_categories = {}
    for col in HIGH_CARD_COLS:
        top_k = train_df[col].value_counts().nlargest(TOP_K_CATEGORIES).index.tolist()
        top_categories[col] = top_k
        print(f"  Top-{TOP_K_CATEGORIES} '{col}' dari Train: {top_k[:2]} ... (total {len(top_k)})")
        for part in (train_df, val_df, test_df):
            part[col] = part[col].where(part[col].isin(top_k), OTHER_TOKEN)

    # Simpan partisi
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        part.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    print(f"\n[✓] Berhasil menyimpan partisi ke {output_dir}/ (train.csv, val.csv, test.csv)")

    # Metadata preprocessing: dipakai dashboard agar input diproses sama seperti saat training.
    metadata = {
        "label_mapping": {"high_severity_codes": HIGH_SEVERITY_CODES, "low_severity_codes": LOW_SEVERITY_CODES},
        "label_profile_raw": label_profile,
        "rows_raw": int(initial_len),
        "rows_excluded_label": int(excluded_len),
        "split": {
            "train_years": list(TRAIN_YEARS), "val_year": VAL_YEAR, "test_year": TEST_YEAR,
            "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
            "class1_ratio": {
                "train": round(float(train_df[TARGET_COL].mean()), 4),
                "val": round(float(val_df[TARGET_COL].mean()), 4),
                "test": round(float(test_df[TARGET_COL].mean()), 4),
            },
        },
        "commodity_counts": {str(k): int(v) for k, v in df_clean["COAL_METAL_IND"].value_counts().items()},
        "numeric_medians_train": medians,
        "top_categories_train": top_categories,
        "leakage_cols_excluded": LEAKAGE_COLS,
    }
    meta_path = os.path.join(artifacts_dir, "preprocessing_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[✓] Menyimpan metadata preprocessing ke: {meta_path}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    load_and_preprocess_msha_accidents()
