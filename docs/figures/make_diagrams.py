"""
make_diagrams.py
Membuat diagram laporan sebagai SVG (tanpa dependensi eksternal):

    python docs/figures/make_diagrams.py

Output: docs/figures/*.svg. Untuk PNG, buka SVG di browser lalu screenshot/ekspor,
atau gunakan PNG yang sudah dirender di folder yang sama.
"""

import os
from html import escape

OUT = os.path.dirname(os.path.abspath(__file__))
FONT = "Arial, Helvetica, 'DejaVu Sans', sans-serif"

# Warna per lapisan arsitektur
C = {
    "source": ("#e8f0fe", "#1a73e8"),
    "ingest": ("#fef7e0", "#e37400"),
    "storage": ("#e6f4ea", "#188038"),
    "process": ("#fce8e6", "#c5221f"),
    "viz": ("#f3e8fd", "#8430ce"),
    "muted": ("#f1f3f4", "#9aa0a6"),
}
TEXT = "#1f2328"
SUB = "#5f6b7a"
LINE = "#5f6368"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="#fff", stroke=LINE, rx=10, sw=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=14, weight="normal", color=TEXT, anchor="middle", italic=False):
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
                 f'text-anchor="{anchor}"{st}>{escape(s)}</text>')

    def box(self, x, y, w, h, layer, title, lines=(), title_size=15, line_size=12.5, dash=None):
        fill, stroke = C[layer]
        self.rect(x, y, w, h, fill, stroke, dash=dash)
        n = 1 + len(lines)
        block = 19 + 17 * (n - 1)
        ty = y + (h - block) / 2 + 14
        self.text(x + w / 2, ty, title, title_size, "bold")
        for i, ln in enumerate(lines):
            self.text(x + w / 2, ty + 19 + 17 * i, ln, line_size, color=SUB)

    def path(self, pts, color=LINE, sw=1.8, dash=None, arrow=True):
        d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        mk = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da}{mk}/>')

    def save(self, name):
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
            f'markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/>'
            "</marker></defs>"
            f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>'
        )
        path = os.path.join(OUT, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + "".join(self.parts) + "</svg>")
        print("saved", path)


def legend(s, x, y, items):
    for label, layer in items:
        fill, stroke = C[layer]
        s.rect(x, y - 13, 18, 18, fill, stroke, rx=4)
        s.text(x + 26, y + 1, label, 13, anchor="start", color=SUB)
        x += 34 + 8 * len(label)


# ----------------------------------------------------------------------------
# Gambar 2.1 Arsitektur
# ----------------------------------------------------------------------------
def architecture():
    s = SVG(1900, 640)
    s.text(950, 40, "Arsitektur Solusi Big Data – Prototipe Data Keselamatan Tambang MSHA", 22, "bold")

    # Sumber data
    srcs = [("Accidents.txt", "218 MB"), ("Violations.txt", "1.375 MB"), ("Mines.txt", "38 MB"),
            ("Inspections.txt", "332 MB")]
    for i, (t, sz) in enumerate(srcs):
        y = 150 + 85 * i
        s.box(30, y, 175, 62, "muted" if t == "Inspections.txt" else "source", t, [sz],
              dash="5 4" if t == "Inspections.txt" else None)
    s.text(117, 490, "Inspections: eksplorasi Assignment I", 11.5, color=SUB, italic=True)
    s.text(117, 135, "MSHA Open Government Data", 12, "bold", SUB)
    # bus sumber -> ingestion
    for i in range(3):
        y = 181 + 85 * i
        s.path([(205, y), (228, y)], arrow=False)
    s.path([(228, 181), (228, 351)], arrow=False)
    s.path([(228, 300), (258, 300)])

    s.box(260, 240, 185, 120, "ingest", "Ingestion", ["Python downloader", "unduh .zip MSHA",
                                                      "batch mingguan", "full refresh"])
    s.path([(445, 300), (488, 300)])
    s.box(490, 245, 170, 110, "storage", "Raw Zone", ["data/*.txt", "berkas asli,", "tidak diubah"])

    # cabang dari raw zone
    s.path([(660, 300), (695, 300)], arrow=False)
    s.path([(695, 125), (695, 475)], arrow=False)
    s.path([(695, 125), (728, 125)])
    s.path([(695, 300), (728, 300)])
    s.path([(695, 475), (728, 475)])

    # Baris A: storage
    s.box(730, 90, 200, 70, "process", "Spark SQL", ["konversi & kueri"])
    s.path([(930, 125), (978, 125)])
    s.box(980, 85, 220, 80, "storage", "Curated Zone", ["Parquet + Snappy", "partisi CAL_YR=YYYY"])

    # Baris B: machine learning
    s.box(730, 255, 200, 90, "process", "Preprocessing", ["pandas", "label · partisi temporal",
                                                          "fit hanya pada train"])
    s.path([(930, 300), (978, 300)])
    s.box(980, 260, 220, 80, "storage", "Processed Zone", ["train 2018–21 · val 2022", "test 2023"])
    s.path([(1200, 300), (1225, 300)], arrow=False)
    s.path([(1225, 252), (1225, 327)], arrow=False)
    s.path([(1225, 252), (1248, 252)])
    s.path([(1225, 327), (1248, 327)])
    s.box(1250, 225, 185, 55, "process", "Scikit-Learn", ["LR · Random Forest"], 14, 12)
    s.box(1250, 300, 185, 55, "process", "Spark MLlib", ["LR · Random Forest"], 14, 12)

    # Baris C: graph
    s.box(730, 440, 200, 70, "process", "Graph Analytics", ["NetworkX + Accidents/Violations"], 15, 11.5)

    # Artifacts & dashboard
    s.box(1485, 230, 190, 120, "storage", "Artifacts", ["model & preprocessor", "metrik benchmark",
                                                        "metrik graf", "ringkasan storage"])
    s.path([(1435, 252), (1483, 252)])
    s.path([(1435, 327), (1483, 327)])
    s.path([(1200, 125), (1580, 125), (1580, 228)], dash="6 4")
    s.path([(930, 475), (1580, 475), (1580, 352)])
    s.path([(1675, 290), (1708, 290)])
    s.box(1710, 235, 170, 110, "viz", "Streamlit", ["dashboard 6 halaman", "hanya membaca", "artifacts"])

    s.text(1390, 115, "ringkasan ukuran & kueri", 11.5, color=SUB, italic=True)

    legend(s, 30, 580, [("Sumber data", "source"), ("Ingestion", "ingest"), ("Storage", "storage"),
                        ("Processing", "process"), ("Visualisasi", "viz")])
    s.text(1870, 585, "Padanan cloud (Assignment I): Ingestion = EventBridge + Lambda · Raw/Curated = Amazon S3 · "
                      "Processing = AWS Glue Spark · Spark SQL = Redshift", 12, color=SUB, anchor="end")
    s.text(1870, 610, "Prototipe: Apache Spark local[*] pada satu mesin (laptop / Google Colab)", 12,
           color=SUB, anchor="end")
    s.save("gambar_2_1_arsitektur.svg")


# ----------------------------------------------------------------------------
# Gambar 4.2 Model data relasional
# ----------------------------------------------------------------------------
def table(s, x, y, w, name, rows, color):
    h = 38 + 26 * len(rows)
    s.rect(x, y, w, h, "#ffffff", color, rx=8, sw=1.8)
    s.add(f'<path d="M{x} {y + 8} a8 8 0 0 1 8 -8 h{w - 16} a8 8 0 0 1 8 8 v30 h{-w} z" fill="{color}"/>')
    s.text(x + w / 2, y + 25, name, 15, "bold", "#ffffff")
    for i, (col, key) in enumerate(rows):
        ry = y + 38 + 26 * i
        if i:
            s.add(f'<line x1="{x}" y1="{ry}" x2="{x + w}" y2="{ry}" stroke="#e3e7ee"/>')
        if key:
            kc = "#e37400" if key == "PK" else "#1a73e8"
            s.rect(x + 10, ry + 5, 30, 16, kc, kc, rx=4, sw=1)
            s.text(x + 25, ry + 17, key, 10.5, "bold", "#ffffff")
        s.text(x + 50, ry + 18, col, 13, "bold" if key else "normal", anchor="start",
               color=TEXT if key or not col.startswith("…") else SUB)
    return h


def row_y(y, i):
    return y + 38 + 26 * i + 13


def data_model():
    s = SVG(1220, 660)
    s.text(610, 36, "Model Data Relasional MSHA", 22, "bold")

    mines = [("MINE_ID", "PK"), ("CURRENT_MINE_NAME", ""), ("CURRENT_MINE_TYPE", ""),
             ("CURRENT_OPERATOR_NAME", ""), ("STATE", ""), ("LATITUDE, LONGITUDE", "")]
    acc = [("DOCUMENT_NO", "PK"), ("MINE_ID", "FK"), ("CAL_YR", ""), ("DEGREE_INJURY_CD", ""),
           ("ACCIDENT_TYPE", ""), ("MINING_EQUIP", ""), ("ACTIVITY", ""), ("TOT_EXPER, MINE_EXPER, JOB_EXPER", ""),
           ("… kolom lainnya", "")]
    insp = [("EVENT_NO", "PK"), ("MINE_ID", "FK"), ("… atribut inspeksi", "")]
    viol = [("VIOLATION_NO", "PK"), ("EVENT_NO", "FK"), ("MINE_ID", "FK"), ("PART_SECTION", ""),
            ("CAL_YR", "")]
    narr = [("DOCUMENT_NO", "FK"), ("NARRATIVE (teks bebas)", "")]

    table(s, 50, 220, 280, "MINES", mines, "#188038")
    table(s, 470, 70, 330, "ACCIDENTS", acc, "#c5221f")
    table(s, 940, 100, 240, "NARRATIVE", narr, "#8430ce")
    table(s, 470, 420, 330, "INSPECTIONS", insp, "#1a73e8")
    table(s, 940, 420, 240, "VIOLATIONS", viol, "#e37400")

    def lbl(x, y, t):
        s.text(x, y, t, 13, "bold", SUB)

    # MINES.MINE_ID -> ACCIDENTS / INSPECTIONS / VIOLATIONS
    my = row_y(220, 0)
    s.path([(330, my), (400, my)], arrow=False)
    s.path([(400, row_y(70, 1)), (400, 625)], arrow=False)
    s.path([(400, row_y(70, 1)), (468, row_y(70, 1))])
    s.path([(400, row_y(420, 1)), (468, row_y(420, 1))])
    s.path([(400, 625), (880, 625), (880, row_y(420, 2)), (938, row_y(420, 2))])
    lbl(345, my - 8, "1")
    lbl(455, row_y(70, 1) - 8, "N")
    lbl(455, row_y(420, 1) - 8, "N")
    lbl(925, row_y(420, 2) - 8, "N")

    # INSPECTIONS.EVENT_NO -> VIOLATIONS.EVENT_NO
    iy, vy = row_y(420, 0), row_y(420, 1)
    s.path([(800, iy), (850, iy), (850, vy), (938, vy)])
    lbl(815, iy - 8, "1")
    lbl(925, vy - 8, "N")

    # ACCIDENTS.DOCUMENT_NO -> NARRATIVE
    ay, ny = row_y(70, 0), row_y(100, 0)
    s.path([(800, ay), (870, ay), (870, ny), (938, ny)])
    lbl(815, ay - 8, "1")
    lbl(925, ny - 8, "1")
    s.text(1060, 200, "data tidak terstruktur", 12, color=SUB, italic=True)

    # legenda
    s.rect(50, 590, 30, 16, "#e37400", "#e37400", rx=4, sw=1)
    s.text(65, 602, "PK", 10.5, "bold", "#ffffff")
    s.text(88, 603, "primary key", 13, anchor="start", color=SUB)
    s.rect(190, 590, 30, 16, "#1a73e8", "#1a73e8", rx=4, sw=1)
    s.text(205, 602, "FK", 10.5, "bold", "#ffffff")
    s.text(228, 603, "foreign key", 13, anchor="start", color=SUB)
    s.text(50, 640, "Hanya kolom yang dipakai pada analisis yang ditampilkan.", 12, color=SUB, anchor="start",
           italic=True)
    s.save("gambar_4_2_model_data.svg")


# ----------------------------------------------------------------------------
# Gambar 5.1 Pipeline machine learning
# ----------------------------------------------------------------------------
def ml_pipeline():
    W = 1000
    s = SVG(W, 1230)
    s.text(W / 2, 38, "Alur Machine Learning Pipeline", 22, "bold")
    x0, w = 150, 700
    cx = x0 + w / 2

    def step(y, h, layer, title, lines=(), num=None, tag=None, tag_color=None):
        s.box(x0, y, w, h, layer, title, lines)
        if num is not None:
            s.add(f'<circle cx="{x0 - 38}" cy="{y + h / 2}" r="17" fill="#1f2328"/>')
            s.text(x0 - 38, y + h / 2 + 5, str(num), 14, "bold", "#ffffff")
        if tag:
            tw = 12 + 7.2 * len(tag)
            s.rect(x0 + w - tw - 10, y + 8, tw, 22, tag_color, tag_color, rx=11, sw=1)
            s.text(x0 + w - tw / 2 - 10, y + 23, tag, 11.5, "bold", "#ffffff")
        return y + h

    def down(y1, y2):
        s.path([(cx, y1), (cx, y2 - 2)])

    y = step(70, 60, "source", "Accidents.txt", ["data mentah MSHA Form 7000-1"], 1)
    down(y, y + 26)
    y = step(y + 26, 78, "process", "Pembacaan & normalisasi",
             ["semua kolom dibaca sebagai teks (MINE_ID, kode tetap utuh)",
              "kode cedera dinormalisasi 2 digit · kategori kosong menjadi MISSING"], 2)
    down(y, y + 26)
    y = step(y + 26, 78, "process", "Pemetaan label IS_LOST_TIME_OR_SEVERE",
             ["DEGREE_INJURY_CD 01–05 = 1 · 00, 06, 10 = 0",
              "07, 08, 09, dan kosong dieksklusi"], 3)
    down(y, y + 26)
    y = step(y + 26, 62, "process", "Filter CAL_YR 2018–2023 & rekayasa fitur",
             ["SHIFT_PERIOD dari SHIFT_BEGIN_TIME (DAY / EVENING / NIGHT / UNKNOWN)"], 4)

    # partisi temporal
    py = y + 34
    down(y, py)
    parts = [("Train", "2018–2021", "#188038"), ("Validation", "2022", "#e37400"), ("Test", "2023", "#c5221f")]
    pw, gap = 220, 20
    s.add(f'<circle cx="{x0 - 38}" cy="{py + 31}" r="17" fill="#1f2328"/>')
    s.text(x0 - 38, py + 36, "5", 14, "bold", "#ffffff")
    for i, (t, yr, col) in enumerate(parts):
        px = x0 + i * (pw + gap)
        s.rect(px, py, pw, 62, "#ffffff", col, sw=2.2)
        s.text(px + pw / 2, py + 27, t, 15, "bold", col)
        s.text(px + pw / 2, py + 46, yr, 12.5, color=SUB)
    s.text(cx, py + 82, "partisi temporal: model dilatih pada masa lalu dan diuji pada tahun berikutnya",
           12, color=SUB, italic=True)
    y = py + 62

    sy = y + 44
    s.path([(x0 + pw / 2, y), (x0 + pw / 2, sy - 2)])
    y = step(sy, 62, "process", "Fit preprocessing hanya pada Train",
             ["median imputasi · Top-15 kategori + OTHER · one-hot encoder · scaler (LR)"], 6,
             "Train", "#188038")

    # dua framework
    my = y + 40
    half = (w - 30) / 2
    s.path([(cx, y), (cx, y + 18)], arrow=False)
    s.path([(x0 + half / 2, y + 18), (x0 + w - half / 2, y + 18)], arrow=False)
    s.path([(x0 + half / 2, y + 18), (x0 + half / 2, my - 2)])
    s.path([(x0 + w - half / 2, y + 18), (x0 + w - half / 2, my - 2)])
    s.add(f'<circle cx="{x0 - 38}" cy="{my + 55}" r="17" fill="#1f2328"/>')
    s.text(x0 - 38, my + 60, "7", 14, "bold", "#ffffff")
    s.box(x0, my, half, 110, "viz", "Tanpa Spark: Scikit-Learn",
          ["ColumnTransformer + OneHotEncoder", "LogisticRegression", "RandomForestClassifier"], 14.5)
    s.box(x0 + half + 30, my, half, 110, "viz", "Dengan Spark: Spark MLlib",
          ["StringIndexer » OneHotEncoder » VectorAssembler", "LogisticRegression",
           "RandomForestClassifier"], 14.5)
    y = my + 110
    s.text(cx, y + 52, "label, fitur, partisi, class weight (balanced), dan hyperparameter sama",
           12, color=SUB, italic=True)

    jy = y + 64
    s.path([(x0 + half / 2, y), (x0 + half / 2, y + 30)], arrow=False)
    s.path([(x0 + w - half / 2, y), (x0 + w - half / 2, y + 30)], arrow=False)
    s.path([(x0 + half / 2, y + 30), (x0 + w - half / 2, y + 30)], arrow=False)
    s.path([(cx, y + 30), (cx, y + 38)], arrow=False)
    s.path([(cx, y + 60), (cx, jy + 18)])
    y = step(jy + 20, 62, "process", "Tuning threshold",
             ["threshold dengan F1 kelas 1 tertinggi · pemilihan model untuk dashboard"], 8,
             "Validation", "#e37400")
    down(y, y + 26)
    y = step(y + 26, 62, "process", "Evaluasi akhir (sekali)",
             ["Accuracy · Precision · Recall · F1 · ROC-AUC · PR-AUC · waktu · baseline mayoritas"], 9,
             "Test", "#c5221f")

    oy = y + 44
    s.path([(cx, y), (cx, y + 22)], arrow=False)
    s.path([(x0 + half / 2, y + 22), (x0 + w - half / 2, y + 22)], arrow=False)
    s.path([(x0 + half / 2, y + 22), (x0 + half / 2, oy - 2)])
    s.path([(x0 + w - half / 2, y + 22), (x0 + w - half / 2, oy - 2)])
    s.box(x0, oy, half, 62, "storage", "benchmark_results.json", ["tabel & grafik Bab 5"], 14)
    s.box(x0 + half + 30, oy, half, 62, "storage", "best_model.joblib", ["+ preprocessor, model_metadata"], 14)
    s.path([(x0 + w - half / 2, oy + 62), (x0 + w - half / 2, oy + 88)])
    s.box(x0 + half + 30, oy + 90, half, 50, "viz", "Dashboard: Simulasi Keparahan", [], 14)
    s.save("gambar_5_1_pipeline_ml.svg")


# ----------------------------------------------------------------------------
# Gambar 6.1 Skema graf heterogen
# ----------------------------------------------------------------------------
def node(s, kind, x, y, label, r=34, size=13):
    color = {"mine": "#1a73e8", "standard": "#c5221f", "accident": "#188038"}[kind]
    light = {"mine": "#e8f0fe", "standard": "#fce8e6", "accident": "#e6f4ea"}[kind]
    if kind == "mine":
        s.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{light}" stroke="{color}" stroke-width="2.4"/>')
    elif kind == "standard":
        s.add(f'<polygon points="{x},{y - r - 4} {x + r + 4},{y} {x},{y + r + 4} {x - r - 4},{y}" '
              f'fill="{light}" stroke="{color}" stroke-width="2.4"/>')
    else:
        s.rect(x - r, y - r, 2 * r, 2 * r, light, color, rx=6, sw=2.4)
    s.text(x, y + 5, label, size, "bold", color)


def edge(s, x1, y1, x2, y2, label=None, color=LINE, sw=2, dash=None, lx=None, ly=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    s.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{da}/>')
    if label:
        s.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 8,
               label, 12.5, "bold", color)


def graph_schema():
    s = SVG(1400, 640)
    s.text(700, 38, "Skema Graf Heterogen: Tambang – Standar Pelanggaran – Tipe Kecelakaan", 22, "bold")

    # Panel A: skema
    s.rect(30, 70, 640, 470, "#fbfcfe", "#e3e7ee", rx=14)
    s.text(350, 102, "(a) Skema simpul dan sisi", 15, "bold", SUB)
    node(s, "standard", 115, 300, "standard", 52, 13)
    node(s, "mine", 345, 300, "mine", 52, 15)
    node(s, "accident", 585, 300, "type", 46, 14)
    edge(s, 171, 300, 293, 300)
    edge(s, 397, 300, 539, 300)
    s.add('<line x1="171" y1="300" x2="293" y2="300" stroke="#c5221f" stroke-width="3"/>')
    s.add('<line x1="397" y1="300" x2="539" y2="300" stroke="#188038" stroke-width="3"/>')
    s.text(232, 290, "CITED_FOR", 11.5, "bold", "#c5221f")
    s.text(468, 290, "REPORTED_ACCIDENT", 11.5, "bold", "#188038")
    s.text(232, 318, "w = jumlah", 11, color=SUB)
    s.text(232, 332, "pelanggaran", 11, color=SUB)
    s.text(468, 318, "w = jumlah", 11, color=SUB)
    s.text(468, 332, "kecelakaan", 11, color=SUB)
    s.text(115, 385, "standard::<PART_SECTION>", 12, color=SUB)
    s.text(115, 402, "pasal 30 CFR (Violations)", 12, color=SUB)
    s.text(345, 385, "mine::<MINE_ID>", 12, color=SUB)
    s.text(345, 402, "tambang", 12, color=SUB)
    s.text(585, 385, "accident_type::<ACCIDENT_TYPE>", 12, color=SUB)
    s.text(585, 402, "tipe kecelakaan (Accidents)", 12, color=SUB)
    s.text(350, 455, "Graf tidak berarah, periode Accidents dan Violations sama (2020–2023).", 12.5, color=SUB)
    s.text(350, 475, "Metrik: weighted degree, PageRank, afinitas & lift, cosine similarity.", 12.5, color=SUB)
    s.text(350, 505, "Sisi = kejadian di tambang yang sama, bukan hubungan sebab-akibat.", 12.5, "bold", SUB)

    # Panel B: proyeksi mine-to-mine
    s.rect(700, 70, 670, 470, "#fbfcfe", "#e3e7ee", rx=14)
    s.text(1035, 102, "(b) Proyeksi mine-to-mine untuk deteksi komunitas", 15, "bold", SUB)
    # graf bipartit contoh
    mA, mB = (790, 200), (790, 400)
    nb = [("standard", 960, 160, "S1"), ("standard", 960, 260, "S2"), ("accident", 960, 360, "T1"),
          ("accident", 960, 450, "T2")]
    linksA, linksB = ["S1", "S2", "T1"], ["S2", "T1", "T2"]
    pos = {lab: (x, y) for _, x, y, lab in nb}
    for lab in linksA:
        edge(s, mA[0] + 26, mA[1], pos[lab][0] - 22, pos[lab][1], color="#9aa0a6", sw=1.8)
    for lab in linksB:
        edge(s, mB[0] + 26, mB[1], pos[lab][0] - 22, pos[lab][1], color="#9aa0a6", sw=1.8)
    node(s, "mine", *mA, "A", 26, 15)
    node(s, "mine", *mB, "B", 26, 15)
    for kind, x, y, lab in nb:
        node(s, kind, x, y, lab, 20, 12)
    s.path([(1010, 305), (1085, 305)])
    # hasil proyeksi
    node(s, "mine", 1150, 230, "A", 26, 15)
    node(s, "mine", 1150, 380, "B", 26, 15)
    edge(s, 1150, 256, 1150, 354, color="#1a73e8", sw=3.2)
    s.text(1275, 300, "Jaccard(A, B)", 13, "bold", "#1a73e8")
    s.text(1275, 320, "= |{S2, T1}| / |{S1, S2, T1, T2}|", 12, color=SUB)
    s.text(1275, 338, "= 2 / 4 = 0,50", 12, color=SUB)
    s.text(1035, 500, "300 tambang dengan weighted degree tertinggi · sisi jika Jaccard >= 0,20 ·", 12.5, color=SUB)
    s.text(1035, 520, "bobot = Jaccard · komunitas: Louvain (seed 42)", 12.5, color=SUB)

    # legenda
    y = 590
    node(s, "mine", 60, y, "", 13)
    s.text(82, y + 5, "tambang", 13, anchor="start", color=SUB)
    node(s, "standard", 200, y, "", 11)
    s.text(222, y + 5, "standar 30 CFR", 13, anchor="start", color=SUB)
    node(s, "accident", 380, y, "", 12)
    s.text(402, y + 5, "tipe kecelakaan", 13, anchor="start", color=SUB)
    s.save("gambar_6_1_skema_graf.svg")


if __name__ == "__main__":
    architecture()
    data_model()
    ml_pipeline()
    graph_schema()
