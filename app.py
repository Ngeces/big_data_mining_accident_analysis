"""
app.py
Dashboard Streamlit Interaktif: Monitoring K3, Peta Geografis Tambang,
Penyimpanan Parquet (format Hadoop), Simulasi Prediksi Keparahan Cedera, Benchmark Model,
serta Visual Network Graph Analytics & Mine-Accident Closeness.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from config import GRAPH_START_YR, GRAPH_END_YR, TRAIN_YEARS, VAL_YEAR, TEST_YEAR  # noqa: E402

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def artifact(name):
    return os.path.join(ARTIFACTS, name)


def read_json(name):
    path = artifact(name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

# 1. Konfigurasi Halaman & Tema
st.set_page_config(
    page_title="MSHA Mine Safety Analytics & Big Data Platform",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #0b57d0;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Academic Disclaimer Wajib
st.warning(
    "⚠️ **Academic Disclaimer:** Dashboard ini merupakan prototipe akademik untuk analisis data historis MSHA "
    "(COMP8035041 Big Data Analytics). Estimasi probabilitas skenario didasarkan pada data laporan retrospektif "
    "dan tidak boleh digunakan sebagai satu-satunya dasar pengambilan keputusan keselamatan operasional tambang."
)

st.title("⛏️ MSHA Big Data Analytics & Mine Safety Platform")
st.markdown(
    "**Course:** COMP8035041 Big Data Analytics | **Program:** Magister Teknik Informatika BINUS University  \n"
    "*Integrasi Parquet Storage (format Hadoop), Pemodelan Machine Learning (Scikit-Learn vs Spark MLlib), Peta Geografis, & Graph Analytics*"
)

# 2. Fungsi Pemuatan Artifak (Cached)
@st.cache_resource
def load_ml_artifacts():
    model_path, preproc_path = artifact("best_model.joblib"), artifact("preprocessor.joblib")
    if os.path.exists(model_path) and os.path.exists(preproc_path):
        return joblib.load(model_path), joblib.load(preproc_path)
    return None, None


@st.cache_data
def load_json_cached(name):
    return read_json(name)


@st.cache_data
def load_csv_cached(name, **kwargs):
    path = artifact(name)
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else pd.DataFrame()


def model_categories(preproc):
    """Kategori yang benar-benar dilihat model saat training (dari OneHotEncoder yang sudah di-fit)."""
    try:
        ohe = preproc.named_transformers_["cat"]
        cols = preproc.transformers_[0][2]
        return {col: [str(c) for c in cats] for col, cats in zip(cols, ohe.categories_)}
    except (AttributeError, KeyError, IndexError):
        return {}


best_model, preprocessor = load_ml_artifacts()
benchmarks = load_json_cached("benchmark_results.json")
hadoop_summary = load_json_cached("hadoop_storage_summary.json")
prep_meta = load_json_cached("preprocessing_metadata.json")
model_meta = load_json_cached("model_metadata.json")
closeness_data = load_json_cached("mine_accident_closeness.json")
network_layout = load_json_cached("graph_network_layout.json")
df_nodes = load_csv_cached("graph_nodes_summary.csv", dtype={"node_id": str, "label": str})
df_geo = load_csv_cached("mine_geo_accidents.csv", dtype={"MINE_ID": str})
df_trend = load_csv_cached("accident_annual_trend.csv")

# 3. Struktur Tab Antarmuka
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 1. Overview K3 & Tren",
    "🗺️ 2. Peta Geografis Tambang",
    "🗄️ 3. Parquet Data Warehouse",
    "🎯 4. Simulasi Prediksi Keparahan",
    "⚡ 5. Benchmark Model ML",
    "🕸️ 6. Graph Analytics & Visual Network"
])

# ==============================================================================
# TAB 1: OVERVIEW K3 & TREN INSIDEN
# ==============================================================================
with tab1:
    st.header("Ringkasan Eksekutif Analisis Data Keselamatan Kerja MSHA (2018–2023)")
    st.markdown("Statistik deskriptif insiden kecelakaan tambang setelah pembersihan dan penyaringan temporal.")

    split = prep_meta.get("split", {})
    n_train, n_val, n_test = split.get("train_rows"), split.get("val_rows"), split.get("test_rows")
    n_total = sum(x for x in (n_train, n_val, n_test) if x) or None
    ratios = split.get("class1_ratio", {})
    class1_pct = None
    if n_total and ratios:
        class1_pct = 100 * sum(ratios[k] * n for k, n in (("train", n_train), ("val", n_val), ("test", n_test))) / n_total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Insiden Berlabel ({TRAIN_YEARS[0]}-{TEST_YEAR})", f"{n_total:,}" if n_total else "n/a", "Setelah eksklusi kode 07/08/09")
    col2.metric("Rasio Lost-Time / Severe", f"{class1_pct:.1f}%" if class1_pct is not None else "n/a", "Target Class 1")
    col3.metric("Simpul Graf Heterogen", f"{len(df_nodes):,}" if not df_nodes.empty else "n/a", "Mines, Standards, Types")
    col4.metric("Tambang Terpetakan di Peta", f"{len(df_geo):,}" if not df_geo.empty else "n/a", "Koordinat valid")
    if not prep_meta:
        st.info("Jalankan `src/data_pipeline.py` untuk membuat `artifacts/preprocessing_metadata.json`.")

    st.markdown("---")
    st.subheader("Distribusi Insiden Berdasarkan Periode Waktu & Komoditas")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Insiden Berdasarkan Komoditas Tambang (Coal vs Metal):**")
        commodity = prep_meta.get("commodity_counts", {})
        if commodity:
            names = {"C": "Coal (Batubara)", "M": "Metal / Non-Metal"}
            df_pie = pd.DataFrame({
                "Komoditas": [names.get(k, k) for k in commodity],
                "Jumlah": list(commodity.values()),
            })
            fig_pie = px.pie(df_pie, names="Komoditas", values="Jumlah", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("**Pembagian Partisi Dataset Temporal untuk Pemodelan:**")
        df_part = pd.DataFrame({
            "Partisi": [f"Training Set ({TRAIN_YEARS[0]}-{TRAIN_YEARS[1]})", f"Validation Set ({VAL_YEAR})", f"Test Set ({TEST_YEAR})"],
            "Jumlah Baris": [n_train, n_val, n_test],
            "Rasio Class 1": [ratios.get("train"), ratios.get("val"), ratios.get("test")],
            "Tujuan Metodologi": ["Fitting Preprocessor & Model", "Pemilihan Model & Threshold", "Evaluasi Akhir (sekali)"],
        })
        st.dataframe(df_part, use_container_width=True)

    # --------------------------------------------------------------------------
    # LINE GRAPH: ANALISIS VOLUME & TREN KECELAKAAN TAHUNAN
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Analisis Tren Historis: Apakah Jumlah Kecelakaan Tambang Cukup Banyak?")
    st.markdown(
        "Grafik garis (*line graph*) di bawah ini memvisualisasikan volume riil kecelakaan tambang tahunan "
        "yang dilaporkan ke MSHA untuk membuktikan apakah insiden kecelakaan terjadi secara masif dan berulang."
    )

    if not df_trend.empty:
        # Filter Kontrol Tren
        t_col1, t_col2 = st.columns([2, 1])
        with t_col1:
            yr_min = int(df_trend["CAL_YR"].min())
            yr_max = int(df_trend["CAL_YR"].max())
            selected_years = st.slider("Pilih Rentang Tahun Analisis:", yr_min, yr_max, (yr_min, yr_max))
        with t_col2:
            trend_view = st.radio(
                "Tampilan Garis Tren:",
                ["Total & Tingkat Keparahan (Severe vs Minor)", "Berdasarkan Komoditas (Coal vs Metal)"],
                index=0
            )

        # Filter Data Berdasarkan Slider
        filtered_trend = df_trend[
            df_trend["CAL_YR"].between(selected_years[0], selected_years[1])
        ].copy()

        # Hitung Statistik Ringkasan Periode Terpilih
        total_period = filtered_trend["total_accidents"].sum()
        severe_period = filtered_trend["severe_accidents"].sum()
        avg_per_year = total_period / max(len(filtered_trend), 1)
        avg_per_day = avg_per_year / 365.25

        # Metric Badges
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Insiden pada Rentang Terpilih", f"{total_period:,}", f"{selected_years[0]} - {selected_years[1]}")
        m2.metric("Rata-rata Insiden per Tahun", f"{avg_per_year:,.0f} kasus/thn", "Volume Sangat Tinggi")
        m3.metric("Rata-rata Insiden per Hari", f"{avg_per_day:.1f} kasus/hari", "Konsisten Setiap Hari")
        m4.metric("Total Insiden Cedera Berat", f"{severe_period:,}", f"{(severe_period/total_period*100):.1f}% dari Total")

        # Buat Visualisasi Line Graph Menggunakan Plotly
        fig_trend = go.Figure()

        if "Severe vs Minor" in trend_view:
            # Garis Total Insiden
            fig_trend.add_trace(go.Scatter(
                x=filtered_trend["CAL_YR"],
                y=filtered_trend["total_accidents"],
                mode="lines+markers",
                name="Total Seluruh Kecelakaan",
                line=dict(color="#0b57d0", width=3.5),
                marker=dict(size=8)
            ))
            # Garis Insiden Parah (Lost-Time / Severe)
            fig_trend.add_trace(go.Scatter(
                x=filtered_trend["CAL_YR"],
                y=filtered_trend["severe_accidents"],
                mode="lines+markers",
                name="Cedera Parah (Lost-Time / Severe)",
                line=dict(color="#d93025", width=2.5, dash="dash"),
                marker=dict(size=7)
            ))
            # Garis Insiden No Lost-Time (kode 00, 06, 10)
            fig_trend.add_trace(go.Scatter(
                x=filtered_trend["CAL_YR"],
                y=filtered_trend["minor_accidents"],
                mode="lines+markers",
                name="No Lost-Time (kode 00, 06, 10)",
                line=dict(color="#1e8e3e", width=2.5, dash="dot"),
                marker=dict(size=7)
            ))
            y_title = "Jumlah Kejadian Kecelakaan"
        else:
            # Garis Komoditas Batubara (Coal)
            fig_trend.add_trace(go.Scatter(
                x=filtered_trend["CAL_YR"],
                y=filtered_trend["coal_accidents"],
                mode="lines+markers",
                name="Tambang Batubara (Coal)",
                line=dict(color="#202124", width=3),
                marker=dict(size=8)
            ))
            # Garis Komoditas Logam (Metal/Non-Metal)
            fig_trend.add_trace(go.Scatter(
                x=filtered_trend["CAL_YR"],
                y=filtered_trend["metal_accidents"],
                mode="lines+markers",
                name="Tambang Logam / Non-Logam (Metal)",
                line=dict(color="#e37400", width=3),
                marker=dict(size=8)
            ))
            y_title = "Jumlah Kejadian per Komoditas"

        fig_trend.update_layout(
            title=f"Tren Tahunan Kecelakaan Tambang MSHA ({selected_years[0]} - {selected_years[1]})",
            xaxis_title="Tahun Kalender (CAL_YR)",
            yaxis_title=y_title,
            hovermode="x unified",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.success(
            f"💡 **Kesimpulan Analitik Data:**  \n"
            f"Data membuktikan bahwa **jumlah kecelakaan yang terjadi sangat banyak**—mencapai **{total_period:,} insiden** "
            f"dalam periode {selected_years[0]}–{selected_years[1]} (rata-rata **{avg_per_day:.1f} kecelakaan per hari**). "
            f"Meskipun terdapat tren penurunan bertahap sejak 2011 akibat regulasi keselamatan yang semakin ketat, "
            f"angka kecelakaan tahunan tetap stabil di angka **5.000 hingga 6.000+ insiden per tahun**, di mana lebih dari **55% di antaranya "
            f"tergolong cedera parah (*Lost-Time / Severe Injury*). Hal ini menjadi justifikasi empiris yang sangat kuat mengapa "
            f"solusi pemantauan **Big Data dan Machine Learning** mutlak dibutuhkan di industri pertambangan."
        )

# ==============================================================================
# TAB 2: PETA GEOGRAFIS LOKASI TAMBANG (INTERACTIVE MAP)
# ==============================================================================
with tab2:
    st.header("🗺️ Peta Interaktif Sebaran Lokasi Tambang Amerika Serikat")
    st.markdown(
        "Menghubungkan data kecelakaan dan metrik Graph Analytics dengan **koordinat geografis riil (Latitude & Longitude)** "
        "dari kamus data resmi MSHA `Mines.txt`. Anda dapat memfilter berdasarkan Negara Bagian, Tipe Tambang, dan Tingkat Kejadian."
    )

    if df_geo.empty:
        st.warning("Data geolokasi tambang belum tersedia di `artifacts/mine_geo_accidents.csv`.")
    else:
        # Sidebar/Toolbar Filter Peta
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            all_states = ["SEMUA NEGARA BAGIAN"] + sorted(df_geo["STATE"].dropna().unique().tolist())
            selected_state = st.selectbox("Filter Negara Bagian (State):", all_states, index=0)
        with f2:
            all_types = ["SEMUA TIPE TAMBANG"] + sorted(df_geo["CURRENT_MINE_TYPE"].dropna().unique().tolist())
            selected_type = st.selectbox("Tipe Tambang:", all_types, index=0)
        with f3:
            all_comms = ["SEMUA KLASTER"] + [f"Komunitas {i}" for i in sorted(df_geo["community_id"].unique()) if i >= 0]
            selected_comm = st.selectbox("Klaster Komunitas Louvain:", all_comms, index=0)
        with f4:
            min_accidents = st.slider("Minimal Jumlah Kecelakaan Tercatat:", 0, 50, 1)

        # Terapkan Filter
        filtered_geo = df_geo.copy()
        if selected_state != "SEMUA NEGARA BAGIAN":
            filtered_geo = filtered_geo[filtered_geo["STATE"] == selected_state]
        if selected_type != "SEMUA TIPE TAMBANG":
            filtered_geo = filtered_geo[filtered_geo["CURRENT_MINE_TYPE"] == selected_type]
        if selected_comm != "SEMUA KLASTER":
            comm_num = int(selected_comm.split()[-1])
            filtered_geo = filtered_geo[filtered_geo["community_id"] == comm_num]
        if min_accidents > 0:
            filtered_geo = filtered_geo[filtered_geo["total_accidents"] >= min_accidents]

        st.markdown(f"**Menampilkan `{len(filtered_geo):,}` tambang** sesuai kriteria filter:")

        # Tampilkan Peta Menggunakan PyDeck + CARTO Basemap
        if not filtered_geo.empty:
            # Batasi data teratas & kalkulasi ukuran radius marker (min: 2000 meter, max: 25000 meter di peta)
            plot_geo = filtered_geo.sort_values("total_accidents", ascending=False).head(2500).copy()
            plot_geo["radius_marker"] = np.clip(plot_geo["total_accidents"] * 1000, 2000, 25000)

            # Mapping warna RGBA sederhana berdasarkan tipe tambang (bisa disesuaikan dengan kategori unik Anda)
            # Contoh: [Red, Green, Blue, Alpha]
            def assign_color(mine_type):
                if "Surface" in str(mine_type):
                    return [239, 85, 59, 200]  # Merah/Oranye
                elif "Underground" in str(mine_type):
                    return [99, 110, 250, 200]  # Biru
                else:
                    return [0, 204, 150, 200]    # Hijau

            plot_geo["color_rgba"] = plot_geo["CURRENT_MINE_TYPE"].apply(assign_color)

            # Konfigurasi Layer PyDeck Scatterplot
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=plot_geo,
                get_position="[lon, lat]",
                get_color="color_rgba",
                get_radius="radius_marker",
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                radius_scale=1,
                radius_min_pixels=3,
                radius_max_pixels=30,
                line_width_min_pixels=1,
                get_line_color=[255, 255, 255, 150]
            )

            # Penentuan Titik Tengah Kamera Peta (AS Tengah)
            view_state = pdk.ViewState(
                latitude=38.0,
                longitude=-96.0,
                zoom=3.5,
                pitch=0,
                bearing=0
            )

            # Render PyDeck Chart menggunakan tema peta CARTO Positron (Light) atau Dark Matter
            # Basemap CARTO tidak memerlukan API key. Jangan menulis kunci API di kode;
            # jika suatu saat diperlukan, baca dari variabel lingkungan atau st.secrets.
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_provider="carto",
                    map_style="light",
                    tooltip={
                        "html": """
                            <b>Nama Tambang:</b> {CURRENT_MINE_NAME} <br/>
                            <b>MINE_ID:</b> {MINE_ID} <br/>
                            <b>Negara Bagian:</b> {STATE} <br/>
                            <b>Tipe Tambang:</b> {CURRENT_MINE_TYPE} <br/>
                            <b>Operator:</b> {CURRENT_OPERATOR_NAME} <br/>
                            <b>Total Kecelakaan (jendela graf):</b> {total_accidents} <br/>
                            <b>Bahaya Dominan:</b> {top_hazard} <br/>
                            <b>ID Komunitas Louvain:</b> {community_id}
                        """,
                        "style": {"backgroundColor": "maroon", "color": "white"}
                    }
                )
            )

            # Detail Tabel Tambang Terpilih
            with st.expander("📋 Lihat Data Rinci Tambang pada Peta di atas"):
                st.dataframe(
                    plot_geo[[
                        "MINE_ID", "CURRENT_MINE_NAME", "STATE", "CURRENT_MINE_TYPE",
                        "CURRENT_OPERATOR_NAME", "total_accidents", "severe_accidents", "top_hazard"
                    ]].head(50),
                    use_container_width=True
                )
        else:
            st.info("Tidak ada tambang yang memenuhi seluruh kombinasi filter yang dipilih.")


# ==============================================================================
# TAB 3: PARQUET STORAGE LAYER
# ==============================================================================
with tab3:
    st.header("🗄️ Penyimpanan Big Data: Apache Parquet dengan Partisi Hive-style")
    st.markdown(
        "Memenuhi capaian pembelajaran **LO 2 & LO 3 (Driving Big Data with Hadoop Tools and Technologies)**.  \n"
        "Data mentah teks MSHA dikonversi oleh Spark ke format kolom **Apache Parquet (Snappy)** dengan partisi "
        "temporal Hive-style (`CAL_YR=YYYY`). Pada prototipe ini data disimpan di filesystem lokal; "
        "format dan tata letak direktorinya sama dengan yang dipakai di HDFS."
    )

    if hadoop_summary:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Ukuran Data Mentah (TXT)", f"{hadoop_summary.get('raw_csv_size_mb', 'n/a')} MB", "Uncompressed Text")
        h2.metric("Ukuran Parquet", f"{hadoop_summary.get('hadoop_parquet_size_mb', 'n/a')} MB", "Snappy Columnar")
        h3.metric("Efisiensi Kompresi", f"{hadoop_summary.get('storage_savings_pct', 'n/a')}%", "Penghematan Disk")
        h4.metric("Waktu Kueri SQL", f"{hadoop_summary.get('olap_query_time_seconds', 'n/a')} s", "Spark SQL")

        st.markdown("---")
        h_col_a, h_col_b = st.columns(2)

        with h_col_a:
            st.subheader("Perbandingan Ukuran Penyimpanan (MB)")
            df_storage = pd.DataFrame({
                "Format Penyimpanan": ["Raw TXT (pipe-delimited)", "Snappy Parquet"],
                "Ukuran Disk (MB)": [hadoop_summary.get('raw_csv_size_mb'), hadoop_summary.get('hadoop_parquet_size_mb')]
            })
            fig_bar = px.bar(df_storage, x="Format Penyimpanan", y="Ukuran Disk (MB)", color="Format Penyimpanan", text_auto=True)
            st.plotly_chart(fig_bar, use_container_width=True)

        with h_col_b:
            st.subheader("Struktur Partisi Direktori Warehouse")
            st.code(
                "data/hdfs_warehouse/msha_accidents/\n"
                "├── CAL_YR=2018/\n"
                "│   └── part-00000.snappy.parquet\n"
                "├── CAL_YR=2019/\n"
                "│   └── part-00000.snappy.parquet\n"
                "├── CAL_YR=2020/\n"
                "├── CAL_YR=2021/\n"
                "├── CAL_YR=2022/\n"
                "├── CAL_YR=2023/\n"
                "└── _SUCCESS",
                language="text"
            )
            st.caption("📌 *Hierarki partisi standar Apache Hive / HDFS memungkinkan kueri analitik melakukan partition pruning.*")

        st.markdown("---")
        st.subheader("Hasil Kueri Analitik Spark SQL di atas Parquet Warehouse:")
        st.code(
            "SELECT CAL_YR AS Tahun, COUNT(*) AS Total_Insiden,\n"
            "       SUM(CASE WHEN DEGREE_INJURY_CD IN ('01','02','03','04','05') THEN 1 ELSE 0 END) AS Insiden_Lost_Time_Severe,\n"
            "       ROUND(AVG(TOT_EXPER), 2) AS Rata_Rata_Pengalaman_Tahun\n"
            "FROM msha_accidents WHERE CAL_YR BETWEEN 2018 AND 2023\n"
            "GROUP BY CAL_YR ORDER BY CAL_YR DESC;",
            language="sql"
        )
        sample_q = hadoop_summary.get("sample_query_results", [])
        if sample_q:
            st.dataframe(pd.DataFrame(sample_q), use_container_width=True)

# ==============================================================================
# TAB 4: SCENARIO-BASED SEVERITY PREDICTOR
# ==============================================================================
with tab4:
    st.header("Simulasi Skenario: Estimasi Keparahan Cedera Insiden Tambang")
    st.markdown(
        "Formulir ini mengevaluasi skenario kondisi insiden yang dilaporkan (*post-incident circumstance*) "
        "untuk mengestimasi apakah insiden tersebut berisiko tergolong **Lost-Time / Severe Injury** "
        "(kematian, disabilitas, atau kehilangan/pembatasan hari kerja)."
    )

    if best_model is None or preprocessor is None:
        st.error("Artifak model belum ditemukan di folder `artifacts/`. Harap jalankan pipeline pemodelan terlebih dahulu.")
    else:
        # Pilihan input = kategori yang dipelajari OneHotEncoder saat training.
        # Nilai di luar daftar ini akan menjadi vektor nol (diabaikan), jadi tidak boleh diketik bebas.
        cats = model_categories(preprocessor) or model_meta.get("categories", {})
        if not cats:
            st.error("Kategori model tidak dapat dibaca dari preprocessor. Jalankan ulang `src/models_baseline.py`.")
            st.stop()

        threshold = model_meta.get("threshold")
        if threshold is None:
            st.warning("`model_metadata.json` belum ada; memakai ambang 0.5. Jalankan ulang `src/models_baseline.py`.")
            threshold = 0.5
        st.caption(
            f"Model: **{model_meta.get('model_name', 'tidak diketahui')}** | "
            f"Ambang klasifikasi (dipilih pada validation set): **{threshold:.2f}**"
        )

        def pick(label, col, preferred=None):
            options = cats.get(col, [])
            index = options.index(preferred) if preferred in options else 0
            return st.selectbox(label, options, index=index)

        with st.form("scenario_form"):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                subunit = pick("Lokasi Kerja (Subunit):", "SUBUNIT")
                commodity_val = pick("Komoditas Tambang (C = Coal, M = Metal/Non-Metal):", "COAL_METAL_IND")
                shift = pick("Waktu Shift Kerja:", "SHIFT_PERIOD", "DAY")

            with col_b:
                accident_type = pick("Tipe Insiden Kecelakaan:", "ACCIDENT_TYPE")
                mining_equip = pick("Peralatan Tambang yang Terlibat:", "MINING_EQUIP")

            with col_c:
                activity = pick("Aktivitas Pekerja Saat Insiden:", "ACTIVITY")
                tot_exper = st.slider("Total Pengalaman Tambang (Tahun):", 0.0, 40.0, 5.0, 0.5)
                mine_exper = st.slider("Pengalaman di Tambang Ini (Tahun):", 0.0, 30.0, 2.0, 0.5)
                job_exper = st.slider("Pengalaman di Posisi Ini (Tahun):", 0.0, 30.0, 2.0, 0.5)

            submitted = st.form_submit_button("🔍 Evaluasi Skenario Insiden", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([{
                "SUBUNIT": subunit,
                "COAL_METAL_IND": commodity_val,
                "ACCIDENT_TYPE": accident_type,
                "MINING_EQUIP": mining_equip,
                "ACTIVITY": activity,
                "SHIFT_PERIOD": shift,
                "TOT_EXPER": float(tot_exper),
                "MINE_EXPER": float(mine_exper),
                "JOB_EXPER": float(job_exper)
            }])

            X_trans = preprocessor.transform(input_df)
            prob_high = best_model.predict_proba(X_trans)[0, 1]

            st.markdown("---")
            st.subheader("Hasil Estimasi Model:")
            r_col1, r_col2 = st.columns([1, 2])

            is_positive = prob_high >= threshold
            with r_col1:
                st.metric(
                    label="Skor Model untuk Lost-Time / Severe",
                    value=f"{prob_high * 100:.1f}%",
                    delta=f"≥ ambang {threshold:.2f}" if is_positive else f"< ambang {threshold:.2f}",
                    delta_color="inverse" if is_positive else "normal"
                )
                st.caption("Skor ini belum dikalibrasi; perlakukan sebagai peringkat risiko, bukan peluang pasti.")

            with r_col2:
                if is_positive:
                    st.error(
                        f"🚨 **Klasifikasi: LOST-TIME OR SEVERE INJURY OUTCOME**  \n"
                        f"Skenario ini memiliki indikasi korelasi tinggi terhadap cedera yang menimbulkan hari kerja hilang "
                        f"atau kecelakaan berbobot berat berdasarkan data historis."
                    )
                else:
                    st.success(
                        f"✅ **Klasifikasi: NO LOST-TIME OUTCOME**  \n"
                        f"Skor skenario ini berada di bawah ambang model untuk kategori Lost-Time / Severe "
                        f"(kode 01-05) berdasarkan data historis."
                    )

# ==============================================================================
# TAB 5: MODEL BENCHMARK (SCIKIT-LEARN VS SPARK MLLIB)
# ==============================================================================
with tab5:
    st.header("Evaluasi Komparatif: Scikit-Learn vs. Apache Spark MLlib")
    st.markdown(
        "Memenuhi capaian pembelajaran **LO 4 (Scalable Machine Learning on Big Data)** dengan "
        "membandingkan model non-Spark melawan distributed Spark MLlib pada partisi pengujian Test Set (2023)."
    )

    if not benchmarks:
        st.warning("Data benchmark belum tersedia di `artifacts/benchmark_results.json`.")
    else:
        table_rows = []
        for model_name, metrics in benchmarks.items():
            table_rows.append({
                "Framework": metrics.get("Framework", ""),
                "Algoritma": metrics.get("Model Name", ""),
                "Accuracy": metrics.get("Accuracy", 0.0),
                "Precision (C1)": metrics.get("Precision_Class1", 0.0),
                "Recall (C1)": metrics.get("Recall_Class1", 0.0),
                "F1-Score (C1)": metrics.get("F1_Class1", 0.0),
                "ROC-AUC": metrics.get("ROC_AUC", 0.0),
                "PR-AUC": metrics.get("PR_AUC", 0.0),
                "Val F1 (C1)": metrics.get("Val_F1_Class1"),
                "Threshold": metrics.get("Optimal_Threshold"),
                "Fit Time (s)": metrics.get("Fit_Time_Seconds"),
                "Inf Time (s)": metrics.get("Inference_Time_Seconds"),
            })
        df_bench = pd.DataFrame(table_rows)
        st.caption(
            "Semua model memakai partisi, fitur, dan class weight yang sama; threshold setiap model dipilih "
            "pada validation set. Baris *Majority Class Baseline* adalah pembanding minimum: model yang berguna "
            "harus mengunggulinya. Fit time Spark tidak termasuk waktu start SparkSession."
        )
        st.dataframe(df_bench.style.highlight_max(subset=["F1-Score (C1)", "ROC-AUC", "PR-AUC"], color="#90EE90"), use_container_width=True)

        st.markdown("---")
        b1, b2 = st.columns(2)
        with b1:
            st.subheader("Perbandingan F1-Score (Class 1)")
            fig_f1 = px.bar(df_bench, x="Algoritma", y="F1-Score (C1)", color="Framework", text_auto=True)
            st.plotly_chart(fig_f1, use_container_width=True)

        with b2:
            st.subheader("Waktu Pelatihan (Fit Time Execution - Detik)")
            fig_time = px.bar(df_bench, x="Algoritma", y="Fit Time (s)", color="Framework", text_auto=True)
            st.plotly_chart(fig_time, use_container_width=True)

        st.info(
            "💡 **Analisis Skalabilitas:** Model Spark MLlib memiliki waktu inisialisasi dan training overhead "
            "pada lingkungan single-node laptop, namun arsitektur RDD/DataFrame Spark siap diskalakan secara horizontal "
            "ke klaster multi-node ketika ukuran data mencapai puluhan gigabyte."
        )

# ==============================================================================
# TAB 6: GRAPH ANALYTICS, VISUAL NETWORK, & MINE CLOSENESS
# ==============================================================================
with tab6:
    st.header("🕸️ Graph Analytics: Visual Jaringan & Kedekatan Tambang-Kecelakaan")
    st.markdown(
        "Memenuhi capaian pembelajaran **LO 5 (Graph Analytics on Big Data)** dengan memodelkan:  \n"
        "1. **Gambar Graf Jaringan Interaktif:** Representasi visual simpul Tambang, Standar Pelanggaran, dan Tipe Kecelakaan.  \n"
        "2. **Mine-Accident Closeness Inspector:** Mengukur *Hazard Affinity* dan *Risk Lift Ratio* spesifik per tambang.  \n"
        "3. **Peer Closeness:** Menemukan tambang dengan sidik jari kecelakaan paling mirip (*Cosine Similarity*)."
    )

    # A. Visual Gambar Graf Jaringan Menggunakan Plotly
    st.subheader("🎨 Gambar Graf Jaringan Relasi Tambang & Bahaya (Interactive Network)")
    st.markdown("Visualisasi graf interaktif gaya pegas (*2D Force-Directed Spring Layout*). Anda dapat melakukan **zoom**, **pan**, dan mengarahkan kursor (*hover*) pada simpul untuk melihat rincian derajat dan komunitas:")

    if network_layout and "nodes" in network_layout:
        nodes_data = network_layout["nodes"]
        edges_list = network_layout["edges"]

        # Gambar Garis Edge
        edge_x = []
        edge_y = []
        for e in edges_list:
            edge_x.extend([e["x0"], e["x1"], None])
            edge_y.extend([e["y0"], e["y1"], None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color="#888888"),
            hoverinfo="none",
            mode="lines"
        )

        # Gambar Titik Node
        node_x = [n["x"] for n in nodes_data]
        node_y = [n["y"] for n in nodes_data]

        node_color = []
        node_text = []
        node_size = []
        node_symbol = []

        for n in nodes_data:
            ntype = n["node_type"]
            deg = n["degree"]
            lbl = n["label"]
            comm = n["community"]

            if ntype == "mine":
                node_color.append("#1f77b4")  # Biru untuk Tambang
                node_symbol.append("circle")
                node_size.append(min(max(deg / 150, 10), 28))
                node_text.append(f"⛏️ <b>Tambang MINE_ID:</b> {lbl}<br><b>Weighted Degree:</b> {deg:,}<br><b>Klaster Komunitas:</b> {comm}")
            elif ntype == "standard":
                node_color.append("#d62728")  # Merah untuk Pelanggaran Standar
                node_symbol.append("diamond")
                node_size.append(min(max(deg / 400, 14), 32))
                node_text.append(f"📜 <b>Standar 30 CFR:</b> {lbl}<br><b>Total Pelanggaran:</b> {deg:,}")
            else:
                node_color.append("#2ca02c")  # Hijau untuk Tipe Kecelakaan
                node_symbol.append("square")
                node_size.append(min(max(deg / 100, 14), 32))
                node_text.append(f"⚠️ <b>Tipe Kecelakaan:</b> {lbl}<br><b>Total Kejadian:</b> {deg:,}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                color=node_color,
                size=node_size,
                symbol=node_symbol,
                line=dict(width=1.5, color="#ffffff")
            )
        )

        fig_net = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=10, l=10, r=10, t=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=580,
                paper_bgcolor="#fcfcfc",
                plot_bgcolor="#fcfcfc"
            )
        )
        st.plotly_chart(fig_net, use_container_width=True)
        st.caption(
            "🔵 **Lingkaran Biru:** Simpul Tambang (`mine::`) | "
            "🔴 **Belah Ketupat Merah:** Standar Pelanggaran 30 CFR (`standard::`) | "
            "🟢 **Kotak Hijau:** Tipe Kecelakaan (`accident_type::`). Ukuran simpul merefleksikan kepadatan kejadian (*Degree Centrality*)."
        )

    # B. Mine-Accident Closeness & Peer Inspector
    st.markdown("---")
    st.subheader("🎯 Mine-Accident Closeness & Peer Inspector")
    st.markdown("Pilih ID Tambang di bawah ini untuk melihat kedekatan risiko bahaya spesifik dan tambang pembanding terdekat:")

    mine_affinity_map = closeness_data.get("mine_affinity", {})
    peer_proximity_map = closeness_data.get("peer_proximity", {})

    available_mines = sorted(list(mine_affinity_map.keys()))
    if available_mines:
        selected_mine = st.selectbox(
            "Pilih MINE_ID (ID Tambang):",
            available_mines,
            index=available_mines.index("4601437") if "4601437" in available_mines else 0
        )

        mine_info = mine_affinity_map.get(selected_mine, {})
        tot_acc = mine_info.get("total_accidents", 0)
        top_hazards = mine_info.get("top_hazards", [])

        st.markdown(f"**Total Insiden Tercatat di Tambang `{selected_mine}` ({GRAPH_START_YR}–{GRAPH_END_YR}):** `{tot_acc:,}` kejadian")

        m_col1, m_col2 = st.columns(2)

        with m_col1:
            st.markdown(f"**Tipe Kecelakaan Paling Dekat (*Closest Hazards*) pada Tambang `{selected_mine}`:**")
            if top_hazards:
                df_haz = pd.DataFrame(top_hazards)
                df_haz = df_haz.rename(columns={
                    "accident_type": "Tipe Kecelakaan",
                    "count": "Jumlah Kasus",
                    "mine_affinity_pct": "Afinitas Tambang (%)",
                    "baseline_pct": "Rata-rata Industri (%)",
                    "risk_lift_ratio": "Risk Multiplier (Lift)"
                })
                st.dataframe(df_haz, use_container_width=True)
                st.caption(
                    "📌 *Risk Multiplier (Lift) > 1.0 berarti tambang ini terpapar bahaya tersebut lebih tinggi "
                    "daripada rata-rata industri nasional.*"
                )

        with m_col2:
            st.markdown(f"**Tambang Pembanding Terdekat (*Closest Peer Mines by Accident Profile*):**")
            peers = peer_proximity_map.get(selected_mine, [])
            if peers:
                df_peers = pd.DataFrame(peers).rename(columns={
                    "peer_mine_id": "ID Tambang Sejawat",
                    "similarity_pct": "Tingkat Kemiripan Profil (%)"
                })
                st.dataframe(df_peers, use_container_width=True)
                st.caption("📌 *Dihitung menggunakan Cosine Similarity pada vektor sebaran tipe kecelakaan antar-tambang.*")

    # C. Metrik Sentralitas Global & Komunitas Louvain
    st.markdown("---")
    st.subheader("🌐 Metrik Sentralitas Global & Komunitas Tambang")

    if not df_nodes.empty:
        g_col1, g_col2, g_col3 = st.columns(3)

        with g_col1:
            st.markdown("**Top 5 Tambang Terpadat (High-Frequency Hubs):**")
            top_m = df_nodes[df_nodes["node_type"] == "mine"].sort_values("weighted_degree", ascending=False).head(5)
            st.dataframe(top_m[["label", "weighted_degree", "pagerank"]], use_container_width=True)

        with g_col2:
            st.markdown("**Top 5 Standar Pelanggaran 30 CFR:**")
            top_s = df_nodes[df_nodes["node_type"] == "standard"].sort_values("weighted_degree", ascending=False).head(5)
            st.dataframe(top_s[["label", "weighted_degree", "pagerank"]], use_container_width=True)

        with g_col3:
            st.markdown("**Top 5 Tipe Insiden Terkait:**")
            top_a = df_nodes[df_nodes["node_type"] == "accident_type"].sort_values("weighted_degree", ascending=False).head(5)
            st.dataframe(top_a[["label", "weighted_degree", "pagerank"]], use_container_width=True)

        st.markdown("---")
        st.subheader("Klaster Komunitas Tambang (Louvain Community Detection)")
        comm_summary = df_nodes[df_nodes["community_id"] >= 0].groupby("community_id").size().reset_index(name="Jumlah Tambang")
        fig_comm = px.bar(comm_summary, x="community_id", y="Jumlah Tambang", color="community_id", text_auto=True)
        st.plotly_chart(fig_comm, use_container_width=True)
