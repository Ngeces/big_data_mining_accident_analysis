"""
graph_analytics.py
Implementasi Graph Analytics heterogen untuk Bab 6 Assignment II:
1. Hubungan Tambang (mine::), Standar Pelanggaran (standard::), dan Tipe Kecelakaan (accident_type::)
2. Metrik "How Close": Mine-Accident Affinity, Risk Lift Ratio, dan Mine-to-Mine Cosine Similarity.
3. Proyeksi Antar-Tambang dan Louvain Community Detection.
"""

import os
import json
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ACCIDENTS_FILE, VIOLATIONS_FILE, ARTIFACTS_DIR,
    GRAPH_START_YR, GRAPH_END_YR, RANDOM_SEED,
)

# Parameter desain graf (dokumentasikan alasannya di Bab 6)
MIN_ACCIDENTS_FOR_AFFINITY = 3   # tambang dengan < 3 insiden tidak dihitung afinitas/lift-nya
TOP_MINES_FOR_PROJECTION = 300   # proyeksi mine-to-mine hanya untuk 300 tambang dengan weighted degree tertinggi
JACCARD_THRESHOLD = 0.20         # sisi proyeksi dibuat jika Jaccard(tetangga) >= 0.20 (tanpa bobot)
EXPLORER_EDGES_PER_TYPE = 300    # jumlah edge terberat per tipe yang diekspor untuk explorer dashboard


def run_graph_analytics(
    accidents_file=ACCIDENTS_FILE,
    violations_file=VIOLATIONS_FILE,
    artifacts_dir=ARTIFACTS_DIR,
    start_yr=GRAPH_START_YR,
    end_yr=GRAPH_END_YR,
):
    print("=" * 60)
    print(f"[*] MEMULAI GRAPH ANALYTICS (JENDELA WAKTU {start_yr}-{end_yr})")
    print("=" * 60)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Agregasi Data Kecelakaan (Accidents)
    print(f"[*] Membaca data Accidents ({start_yr}-{end_yr})...")
    acc_cols = ["MINE_ID", "ACCIDENT_TYPE", "CAL_YR"]
    df_acc = pd.read_csv(
        accidents_file,
        sep="|",
        usecols=acc_cols,
        dtype=str,
        encoding="latin1",
        quotechar='"'
    )
    df_acc["CAL_YR"] = pd.to_numeric(df_acc["CAL_YR"], errors="coerce")
    df_acc = df_acc[df_acc["CAL_YR"].between(start_yr, end_yr)].copy()
    for c in ["MINE_ID", "ACCIDENT_TYPE"]:
        df_acc[c] = df_acc[c].str.strip().str.replace('"', '')

    df_acc = df_acc[(df_acc["MINE_ID"].str.len() > 0) & (~df_acc["ACCIDENT_TYPE"].isin(["?", "NO VALUE FOUND"]))]

    print(f"[+] Total insiden kecelakaan tervalidasi ({start_yr}-{end_yr}): {len(df_acc):,} baris")
    acc_agg = df_acc.groupby(["MINE_ID", "ACCIDENT_TYPE"]).size().reset_index(name="weight")
    print(f"[+] Total edge unik Mine <-> Accident_Type: {len(acc_agg):,}")

    # 2. Agregasi Data Pelanggaran (Violations) secara Chunking
    print(f"\n[*] Membaca & mengagregasi data Violations ({start_yr}-{end_yr}) secara chunking...")
    viol_chunks = []
    chunk_size = 200_000
    for chunk in pd.read_csv(
        violations_file,
        sep="|",
        usecols=["MINE_ID", "PART_SECTION", "CAL_YR"],
        dtype=str,
        encoding="latin1",
        quotechar='"',
        chunksize=chunk_size
    ):
        chunk["CAL_YR"] = pd.to_numeric(chunk["CAL_YR"], errors="coerce")
        chunk_filtered = chunk[chunk["CAL_YR"].between(start_yr, end_yr)].copy()
        if not chunk_filtered.empty:
            for c in ["MINE_ID", "PART_SECTION"]:
                chunk_filtered[c] = chunk_filtered[c].astype(str).str.strip().str.replace('"', '')
            chunk_filtered = chunk_filtered[
                (chunk_filtered["MINE_ID"].str.len() > 0) &
                (chunk_filtered["PART_SECTION"] != "") &
                (chunk_filtered["PART_SECTION"] != "?") &
                (chunk_filtered["PART_SECTION"] != "nan")
            ]
            agg_chunk = chunk_filtered.groupby(["MINE_ID", "PART_SECTION"]).size().reset_index(name="weight")
            viol_chunks.append(agg_chunk)

    df_viol_all = pd.concat(viol_chunks, ignore_index=True)
    viol_agg = df_viol_all.groupby(["MINE_ID", "PART_SECTION"])["weight"].sum().reset_index()
    print(f"[+] Total edge unik Mine <-> Violated_Standard: {len(viol_agg):,}")

    # 3. Analisis Hubungan Kedekatan (Closeness) Tambang terhadap Tipe Kecelakaan
    print("\n[*] Menghitung Metrik Kedekatan (Closeness & Affinity) Tambang-Kecelakaan...")
    crosstab_acc = pd.crosstab(df_acc["MINE_ID"], df_acc["ACCIDENT_TYPE"])
    active_mines = crosstab_acc[crosstab_acc.sum(axis=1) >= MIN_ACCIDENTS_FOR_AFFINITY]

    overall_counts = active_mines.sum()
    overall_dist = overall_counts / overall_counts.sum()

    # Hitung Closeness & Lift Ratio per tambang
    mine_affinity_dict = {}
    for mine_id, row in active_mines.iterrows():
        total_m = row.sum()
        if total_m > 0:
            mine_dist = row / total_m
            lift = mine_dist / overall_dist
            top_types = []
            for acc_t, count in row.sort_values(ascending=False).head(5).items():
                if count > 0:
                    top_types.append({
                        "accident_type": acc_t,
                        "count": int(count),
                        "mine_affinity_pct": round(float(mine_dist[acc_t]) * 100, 2),
                        "baseline_pct": round(float(overall_dist[acc_t]) * 100, 2),
                        "risk_lift_ratio": round(float(lift[acc_t]), 2)
                    })
            mine_affinity_dict[mine_id] = {
                "total_accidents": int(total_m),
                "top_hazards": top_types
            }

    # Hitung Kedekatan Antar-Tambang (Mine-to-Mine Cosine Proximity)
    print("[*] Menghitung Kedekatan Antar-Tambang (Cosine Similarity Profil Bahaya)...")
    cos_matrix = cosine_similarity(active_mines)
    df_cos = pd.DataFrame(cos_matrix, index=active_mines.index, columns=active_mines.index)

    peer_proximity_dict = {}
    for mine_id in active_mines.index:
        peers = df_cos.loc[mine_id].drop(mine_id).sort_values(ascending=False).head(5)
        peer_proximity_dict[mine_id] = [
            {"peer_mine_id": p_id, "similarity_pct": round(float(score) * 100, 1)}
            for p_id, score in peers.items()
        ]

    closeness_export_path = os.path.join(artifacts_dir, "mine_accident_closeness.json")
    with open(closeness_export_path, "w") as f:
        json.dump({
            "mine_affinity": mine_affinity_dict,
            "peer_proximity": peer_proximity_dict
        }, f, indent=2)
    print(f"[✓] Menyimpan metrik kedekatan tambang-kecelakaan ke: {closeness_export_path}")

    # 4. Bangun Graf Heterogen NetworkX
    print("\n[*] Membangun Graf Heterogen Multi-Modal...")
    G = nx.Graph()

    def add_bipartite_edges(agg, other_col, prefix, node_type, edge_type):
        mines = agg["MINE_ID"].astype(str)
        others = agg[other_col].astype(str)
        G.add_nodes_from(((f"mine::{m}", {"node_type": "mine", "label": m}) for m in mines.unique()))
        G.add_nodes_from(((f"{prefix}::{o}", {"node_type": node_type, "label": o}) for o in others.unique()))
        G.add_edges_from(
            (f"mine::{m}", f"{prefix}::{o}", {"weight": int(w), "edge_type": edge_type})
            for m, o, w in zip(mines, others, agg["weight"])
        )

    add_bipartite_edges(viol_agg, "PART_SECTION", "standard", "standard", "CITED_FOR")
    add_bipartite_edges(acc_agg, "ACCIDENT_TYPE", "accident_type", "accident_type", "REPORTED_ACCIDENT")

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"[+] Graf Heterogen Terbentuk: {num_nodes:,} simpul, {num_edges:,} sisi")

    # 5. Hitung Metrik Sentralitas
    degrees = dict(G.degree(weight="weight"))
    pageranks = nx.pagerank(G, weight="weight", max_iter=100)

    node_metrics = []
    for node, data in G.nodes(data=True):
        node_metrics.append({
            "node_id": node,
            "node_type": data.get("node_type", "unknown"),
            "label": data.get("label", node),
            "weighted_degree": degrees.get(node, 0),
            "pagerank": pageranks.get(node, 0.0)
        })

    df_metrics = pd.DataFrame(node_metrics)

    # 6. Proyeksi Antar-Tambang (Mine-to-Mine Projection) & Community Detection
    print("\n[*] Menjalankan Proyeksi Graf Antar-Tambang & Louvain Community...")
    top_active_mines = (
        df_metrics[df_metrics["node_type"] == "mine"]
        .sort_values("weighted_degree", ascending=False)
        .head(TOP_MINES_FOR_PROJECTION)["node_id"].tolist()
    )

    mine_features = {mine: set(G.neighbors(mine)) for mine in top_active_mines}
    G_proj = nx.Graph()
    for mine in top_active_mines:
        G_proj.add_node(mine, label=mine.replace("mine::", ""))

    jaccard_threshold = JACCARD_THRESHOLD
    for i in range(len(top_active_mines)):
        m1 = top_active_mines[i]
        feats1 = mine_features[m1]
        if not feats1:
            continue
        for j in range(i + 1, len(top_active_mines)):
            m2 = top_active_mines[j]
            feats2 = mine_features[m2]
            if not feats2:
                continue
            intersection = len(feats1 & feats2)
            union = len(feats1 | feats2)
            if union > 0 and (intersection / union) >= jaccard_threshold:
                G_proj.add_edge(m1, m2, weight=round(intersection / union, 3))

    communities = louvain_communities(G_proj, weight="weight", seed=RANDOM_SEED)
    isolated = sum(1 for c in communities if len(c) == 1)
    print(f"[+] Sisi proyeksi: {G_proj.number_of_edges():,} | komunitas berisi 1 tambang (terisolasi): {isolated}")
    print(f"[+] Ditemukan {len(communities)} klaster komunitas profil risiko!")

    mine_community_map = {}
    for comm_id, comm in enumerate(communities):
        for member in comm:
            mine_community_map[member] = comm_id

    df_metrics["community_id"] = df_metrics["node_id"].map(mine_community_map).fillna(-1).astype(int)

    summary_path = os.path.join(artifacts_dir, "graph_nodes_summary.csv")
    df_metrics.to_csv(summary_path, index=False)
    print(f"[✓] Menyimpan rangkuman simpul graf ke: {summary_path}")

    # Simpan Top Edges
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)[:150]
    top_edges_list = [
        {"source": u, "target": v, "weight": d.get("weight", 1), "edge_type": d.get("edge_type", "")}
        for u, v, d in sorted_edges
    ]

    # Edge terberat PER TIPE agar explorer dashboard memuat kedua relasi
    # (top edges global hampir seluruhnya CITED_FOR karena bobot pelanggaran jauh lebih besar).
    explorer_rows = []
    for etype in ("CITED_FOR", "REPORTED_ACCIDENT"):
        typed = [(u, v, d["weight"]) for u, v, d in G.edges(data=True) if d.get("edge_type") == etype]
        typed.sort(key=lambda x: x[2], reverse=True)
        for u, v, w in typed[:EXPLORER_EDGES_PER_TYPE]:
            src, dst = (u, v) if u.startswith("mine::") else (v, u)  # source selalu tambang
            explorer_rows.append({"source": src, "target": dst, "weight": int(w), "edge_type": etype})
    explorer_path = os.path.join(artifacts_dir, "graph_explorer_edges.csv")
    pd.DataFrame(explorer_rows).to_csv(explorer_path, index=False)
    print(f"[✓] Menyimpan edge explorer ({len(explorer_rows)} edge) ke: {explorer_path}")

    graph_viz_path = os.path.join(artifacts_dir, "graph_top_edges.json")
    with open(graph_viz_path, "w") as f:
        json.dump({
            "total_nodes": num_nodes,
            "total_edges": num_edges,
            "top_edges": top_edges_list,
            "num_communities": len(communities)
        }, f, indent=2)
    print(f"[✓] Menyimpan top edges visualisasi ke: {graph_viz_path}")

    return df_metrics


if __name__ == "__main__":
    run_graph_analytics()
