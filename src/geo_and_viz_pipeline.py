"""
geo_and_viz_pipeline.py
Modul untuk:
1. Mengintegrasikan koordinat geografis Mines.txt dengan metrik Graph Analytics & Kecelakaan.
2. Menghasilkan layout visual jaringan graf interaktif (Plotly Spring Layout) untuk Tab Graph Analytics.
"""

import os
import json
import pandas as pd
import networkx as nx

from config import (
    MINES_FILE, ACCIDENTS_FILE, ARTIFACTS_DIR, HIGH_SEVERITY_CODES, LOW_SEVERITY_CODES,
    GRAPH_START_YR, GRAPH_END_YR, RANDOM_SEED,
)

TREND_START_YR, TREND_END_YR = 2010, 2023


def build_annual_trend(df_acc, artifacts_dir):
    """Tren tahunan untuk Tab Overview. 'other' = kode 07/08/09/kosong yang dieksklusi dari model."""
    df = df_acc[df_acc["CAL_YR"].between(TREND_START_YR, TREND_END_YR)].copy()
    df["is_severe"] = df["DEGREE_INJURY_CD"].isin(HIGH_SEVERITY_CODES)
    df["is_minor"] = df["DEGREE_INJURY_CD"].isin(LOW_SEVERITY_CODES)
    trend = df.groupby("CAL_YR").agg(
        total_accidents=("DEGREE_INJURY_CD", "size"),
        severe_accidents=("is_severe", "sum"),
        minor_accidents=("is_minor", "sum"),
        coal_accidents=("COAL_METAL_IND", lambda s: (s == "C").sum()),
        metal_accidents=("COAL_METAL_IND", lambda s: (s == "M").sum()),
    ).reset_index()
    trend["other_accidents"] = trend["total_accidents"] - trend["severe_accidents"] - trend["minor_accidents"]
    trend["severe_pct"] = (trend["severe_accidents"] / trend["total_accidents"] * 100).round(2)
    trend["CAL_YR"] = trend["CAL_YR"].astype(int)
    path = os.path.join(artifacts_dir, "accident_annual_trend.csv")
    trend.to_csv(path, index=False)
    print(f"[✓] Menyimpan tren tahunan ke: {path}")
    return trend


def generate_geo_and_network_artifacts(
    mines_file=MINES_FILE,
    accidents_file=ACCIDENTS_FILE,
    graph_nodes_file=os.path.join(ARTIFACTS_DIR, "graph_nodes_summary.csv"),
    graph_edges_file=os.path.join(ARTIFACTS_DIR, "graph_top_edges.json"),
    artifacts_dir=ARTIFACTS_DIR,
):
    print("=" * 60)
    print("[*] MEMULAI ENRICHMENT GEOGRAFIS & VISUAL NETWORK GRAPH LAYOUT")
    print("=" * 60)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Baca dan Bersihkan Data Geografis Mines.txt
    print(f"[*] Membaca data koordinat geografis dari {mines_file}...")
    geo_cols = [
        "MINE_ID", "CURRENT_MINE_NAME", "STATE", "FIPS_CNTY_NM",
        "LATITUDE", "LONGITUDE", "CURRENT_MINE_TYPE", "CURRENT_OPERATOR_NAME"
    ]
    df_m = pd.read_csv(
        mines_file,
        sep="|",
        usecols=geo_cols,
        dtype=str,
        encoding="latin1",
        quotechar='"'
    )
    for c in df_m.columns:
        df_m[c] = df_m[c].str.strip().str.replace('"', '')

    df_m["lat"] = pd.to_numeric(df_m["LATITUDE"], errors="coerce")
    df_m["lon"] = pd.to_numeric(df_m["LONGITUDE"], errors="coerce")
    # Filter koordinat valid Amerika Serikat (Continental US, Alaska, Hawaii)
    df_m = df_m[(df_m["lat"].between(18, 72)) & (df_m["lon"].between(-170, -65))].copy()
    print(f"[+] Total tambang dengan koordinat valid: {len(df_m):,} tambang")

    # 2. Gabungkan dengan Metrik Graph Analytics (Weighted Degree, PageRank, Community)
    print(f"[*] Menggabungkan dengan metrik graf dari {graph_nodes_file}...")
    # MINE_ID dibaca sebagai string agar leading zero cocok dengan Mines.txt
    df_nodes = pd.read_csv(graph_nodes_file, dtype={"node_id": str, "label": str})
    mine_nodes = df_nodes[df_nodes["node_type"] == "mine"].copy()
    mine_nodes["MINE_ID"] = mine_nodes["label"]

    df_acc = pd.read_csv(
        accidents_file,
        sep="|",
        usecols=["MINE_ID", "CAL_YR", "DEGREE_INJURY_CD", "ACCIDENT_TYPE", "COAL_METAL_IND"],
        dtype=str,
        encoding="latin1",
        quotechar='"',
    )
    for c in df_acc.columns:
        df_acc[c] = df_acc[c].str.strip().str.replace('"', '')
    df_acc["CAL_YR"] = pd.to_numeric(df_acc["CAL_YR"], errors="coerce")

    build_annual_trend(df_acc, artifacts_dir)

    # Statistik per tambang memakai jendela yang SAMA dengan graph analytics
    df_acc_win = df_acc[df_acc["CAL_YR"].between(GRAPH_START_YR, GRAPH_END_YR)].copy()

    acc_stats = df_acc_win.groupby("MINE_ID").agg(
        total_accidents=("DEGREE_INJURY_CD", "size"),
        severe_accidents=("DEGREE_INJURY_CD", lambda s: s.isin(HIGH_SEVERITY_CODES).sum()),
        top_hazard=("ACCIDENT_TYPE", lambda s: s.value_counts().index[0] if s.notna().any() else "None")
    ).reset_index()

    # Merge Geo + Graph + Accidents
    df_geo_merged = pd.merge(df_m, mine_nodes[["MINE_ID", "weighted_degree", "pagerank", "community_id"]], on="MINE_ID", how="inner")
    df_geo_merged = pd.merge(df_geo_merged, acc_stats, on="MINE_ID", how="left")
    df_geo_merged["total_accidents"] = df_geo_merged["total_accidents"].fillna(0).astype(int)
    df_geo_merged["severe_accidents"] = df_geo_merged["severe_accidents"].fillna(0).astype(int)
    df_geo_merged["top_hazard"] = df_geo_merged["top_hazard"].fillna("Tidak Ada Insiden")

    geo_output_path = os.path.join(artifacts_dir, "mine_geo_accidents.csv")
    df_geo_merged.to_csv(geo_output_path, index=False)
    print(f"[✓] Berhasil menyimpan dataset geolokasi ke: {geo_output_path} ({len(df_geo_merged):,} baris)")

    # 3. Bangun Layout Visual Network Graph (Spring Layout untuk Plotly)
    print("\n[*] Menghitung koordinat 2D Spring Layout untuk visualisasi graf Plotly...")
    with open(graph_edges_file, "r") as f:
        edges_data = json.load(f)

    top_edges = edges_data.get("top_edges", [])[:120]  # Ambil 120 relasi paling kuat agar visual bersih

    # Buat Subgraph NetworkX
    sub_G = nx.Graph()
    for e in top_edges:
        sub_G.add_edge(e["source"], e["target"], weight=e["weight"], edge_type=e["edge_type"])

    # Tambahkan atribut node
    node_attr_dict = df_nodes.set_index("node_id").to_dict(orient="index")
    for n in sub_G.nodes():
        attr = node_attr_dict.get(n, {})
        sub_G.nodes[n]["node_type"] = attr.get("node_type", "unknown")
        sub_G.nodes[n]["label"] = attr.get("label", n)
        sub_G.nodes[n]["degree"] = attr.get("weighted_degree", 1)
        sub_G.nodes[n]["community"] = attr.get("community_id", -1)

    # Hitung posisi 2D Spring Layout
    pos = nx.spring_layout(sub_G, k=0.45, iterations=60, seed=RANDOM_SEED)

    # Siapkan data koordinat untuk Plotly
    nodes_viz = []
    for n in sub_G.nodes():
        x, y = pos[n]
        attr = sub_G.nodes[n]
        nodes_viz.append({
            "id": n,
            "x": float(x),
            "y": float(y),
            "node_type": attr["node_type"],
            "label": str(attr["label"]),
            "degree": int(attr["degree"]),
            "community": int(attr["community"])
        })

    edges_viz = []
    for u, v, d in sub_G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edges_viz.append({
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "weight": int(d.get("weight", 1)),
            "edge_type": str(d.get("edge_type", ""))
        })

    layout_data = {
        "nodes": nodes_viz,
        "edges": edges_viz,
        "total_sub_nodes": len(nodes_viz),
        "total_sub_edges": len(edges_viz)
    }

    layout_output_path = os.path.join(artifacts_dir, "graph_network_layout.json")
    with open(layout_output_path, "w") as f:
        json.dump(layout_data, f, indent=2)
    print(f"[✓] Berhasil menyimpan layout visual graf ke: {layout_output_path}")

    return df_geo_merged, layout_data


if __name__ == "__main__":
    generate_geo_and_network_artifacts()
