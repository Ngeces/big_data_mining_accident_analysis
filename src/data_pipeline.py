"""
data_pipeline.py
Pipeline pemrosesan data resmi MSHA Accidents dengan metodologi kedap kebocoran data (leakage-free).
"""

import os
import pandas as pd
import numpy as np


def parse_shift_time(time_val):
    """Mengonversi nilai waktu HHMM menjadi kategori shift kerja."""
    if pd.isna(time_val) or time_val == "" or time_val == "?":
        return "UNKNOWN"
    try:
        val_str = str(time_val).strip().replace('"', '').replace("'", "")
        if not val_str.isdigit():
            return "UNKNOWN"
        val_str = val_str.zfill(4)
        hour = int(val_str[:2])
        if 6 <= hour < 14:
            return "DAY"
        elif 14 <= hour < 22:
            return "EVENING"
        elif 0 <= hour < 6 or 22 <= hour <= 24:
            return "NIGHT"
        else:
            return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def load_and_preprocess_msha_accidents(
    file_path="data/Accidents.txt",
    output_dir="data/processed"
):
    """
    Membaca berkas Accidents.txt MSHA asli, melakukan profiling label,
    pemetaan target biner IS_LOST_TIME_OR_SEVERE, pencegahan leakage,
    serta partisi data temporal: Train (2018-2021), Val (2022), Test (2023).
    """
    print(f"[*] Membaca data MSHA dari {file_path}...")
    os.makedirs(output_dir, exist_ok=True)

    dtype_dict = {
        "MINE_ID": str,
        "DOCUMENT_NO": str,
        "CAL_YR": int,
        "DEGREE_INJURY_CD": str,
        "SUBUNIT_CD": str,
        "SUBUNIT": str,
        "COAL_METAL_IND": str,
        "ACCIDENT_TYPE_CD": str,
        "ACCIDENT_TYPE": str,
        "CLASSIFICATION_CD": str,
        "CLASSIFICATION": str,
        "MINING_EQUIP_CD": str,
        "MINING_EQUIP": str,
        "ACTIVITY_CD": str,
        "ACTIVITY": str,
        "SHIFT_BEGIN_TIME": str,
        "ACCIDENT_TIME": str,
    }

    df = pd.read_csv(
        file_path,
        sep="|",
        dtype=dtype_dict,
        low_memory=False,
        quotechar='"',
        encoding="latin1"
    )
    print(f"[+] Total data mentah terbaca: {len(df):,} baris")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.replace('"', '')

    # 1. Profiling Label DEGREE_INJURY_CD
    print("\n--- [PROFILING LABEL: DEGREE_INJURY_CD] ---")
    deg_counts = df["DEGREE_INJURY_CD"].value_counts(dropna=False)
    for code, count in deg_counts.items():
        pct = (count / len(df)) * 100
        print(f"  Kode {code:>3}: {count:>8,} baris ({pct:5.2f}%)")

    # 2. Pemetaan Target (Pilihan B: Lost-Time / Severe Injury Outcome)
    high_severity_codes = ["01", "02", "03", "04", "05"]
    low_severity_codes = ["00", "06", "10"]
    valid_codes = high_severity_codes + low_severity_codes

    initial_len = len(df)
    df = df[df["DEGREE_INJURY_CD"].isin(valid_codes)].copy()
    excluded_len = initial_len - len(df)
    print(f"\n[+] Data valid target dipertahankan: {len(df):,} baris ({len(df)/initial_len*100:.2f}%)")
    print(f"[-] Data dieksklusi (07, 08, 09, missing): {excluded_len:,} baris ({excluded_len/initial_len*100:.2f}%)")

    df["IS_LOST_TIME_OR_SEVERE"] = df["DEGREE_INJURY_CD"].isin(high_severity_codes).astype(int)

    dist = df["IS_LOST_TIME_OR_SEVERE"].value_counts()
    print("\n[+] Distribusi Target Final:")
    print(f"  Class 0 (Minor / No Lost-Time): {dist.get(0, 0):,} ({dist.get(0, 0)/len(df)*100:.2f}%)")
    print(f"  Class 1 (Lost-Time / Severe):    {dist.get(1, 0):,} ({dist.get(1, 0)/len(df)*100:.2f}%)")

    # 3. Filter Jendela Waktu Penelitian (2018 - 2023)
    df_window = df[df["CAL_YR"].between(2018, 2023)].copy()
    print(f"\n[+] Data dalam jendela 2018-2023: {len(df_window):,} baris")

    # 4. Transformasi Fitur Waktu & Sanitasi Kolom
    print("[*] Melakukan rekayasa fitur waktu (Shift categorization)...")
    df_window["SHIFT_PERIOD"] = df_window["SHIFT_BEGIN_TIME"].apply(parse_shift_time)

    num_cols = ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER"]
    for col in num_cols:
        df_window[col] = pd.to_numeric(df_window[col], errors="coerce")

    # 5. Eksklusi Ketat Seluruh Fitur Leakage
    leakage_cols = [
        "DAYS_LOST", "DAYS_RESTRICT", "SCHEDULE_CHARGE",
        "RETURN_TO_WORK_DT", "TRANS_TERM", "CLOSED_DOC_NO",
        "INVEST_BEGIN_DT", "DEGREE_INJURY_CD", "DEGREE_INJURY",
        "NATURE_INJURY_CD", "NATURE_INJURY", "INJ_BODY_PART_CD", "INJ_BODY_PART",
        "INJURY_SOURCE_CD", "INJURY_SOURCE", "IMMED_NOTIFY_CD", "IMMED_NOTIFY",
        "NARRATIVE"
    ]
    available_leakage = [c for c in leakage_cols if c in df_window.columns]
    print(f"[!] Mengeksklusi {len(available_leakage)} kolom leakage dari prediktor.")

    cat_cols = ["SUBUNIT", "COAL_METAL_IND", "ACCIDENT_TYPE", "MINING_EQUIP", "ACTIVITY", "SHIFT_PERIOD"]
    meta_cols = ["MINE_ID", "DOCUMENT_NO", "CAL_YR"]
    target_col = ["IS_LOST_TIME_OR_SEVERE"]

    selected_cols = meta_cols + target_col + cat_cols + num_cols
    df_clean = df_window[selected_cols].copy()

    # 6. Partisi Temporal (Train: 2018-2021, Val: 2022, Test: 2023)
    train_mask = df_clean["CAL_YR"].between(2018, 2021)
    val_mask = df_clean["CAL_YR"] == 2022
    test_mask = df_clean["CAL_YR"] == 2023

    train_df = df_clean[train_mask].copy()
    val_df = df_clean[val_mask].copy()
    test_df = df_clean[test_mask].copy()

    print(f"\n[+] Partisi Data Temporal:")
    print(f"  Training Set   (2018-2021): {len(train_df):,} baris")
    print(f"  Validation Set (2022):       {len(val_df):,} baris")
    print(f"  Test Set       (2023):       {len(test_df):,} baris")

    # 7. Training-Only Parameter Fitting
    print("\n[*] Menghitung parameter imputasi dan Top-15 kategori hanya dari Training Set...")
    medians = {col: float(train_df[col].median()) for col in num_cols}
    print(f"  Median Pengalaman (Train): {medians}")

    for col in num_cols:
        train_df[col] = train_df[col].fillna(medians[col])
        val_df[col] = val_df[col].fillna(medians[col])
        test_df[col] = test_df[col].fillna(medians[col])

    high_card_cols = ["MINING_EQUIP", "ACTIVITY"]
    top_categories = {}
    for col in high_card_cols:
        top_15 = train_df[col].value_counts().nlargest(15).index.tolist()
        top_categories[col] = top_15
        top_set = set(top_15)
        print(f"  Top-15 Kategori '{col}' dari Train: {top_15[:2]} ... (total {len(top_15)})")

        train_df[col] = train_df[col].apply(lambda x: x if x in top_set else "OTHER")
        val_df[col] = val_df[col].apply(lambda x: x if x in top_set else "OTHER")
        test_df[col] = test_df[col].apply(lambda x: x if x in top_set else "OTHER")

    # Simpan hasil partisi ke CSV
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\n[✓] Berhasil menyimpan partisi ke {output_dir}/ (train.csv, val.csv, test.csv)")

    return train_df, val_df, test_df


if __name__ == "__main__":
    load_and_preprocess_msha_accidents()
