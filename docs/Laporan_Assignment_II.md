<!--
CATATAN PENYUSUNAN (HAPUS SEBELUM DIKUMPULKAN)

Penanda di dokumen ini:
  ⟦CEK⟧  = angka dari eksekusi pipeline versi lama. Cocokkan dengan output terbaru `python src/run_all.py`.
  ⟦ISI⟧  = angka atau gambar yang harus diisi dari artefak terbaru.
  ✍️ BAGIAN MAHASISWA = bagian yang menurut Ketentuan Penggunaan AI Assignment II wajib ditulis
     sendiri (alasan pemilihan, interpretasi hasil, perbandingan model, analisis graf, kesimpulan).
     Pertanyaan pemandu di bawahnya hanya membantu menyusun argumen; tidak berisi jawaban.

Daftar Isi, Daftar Gambar, dan Daftar Tabel dibuat otomatis di Word/Google Docs
(gunakan Heading 1–3 untuk judul bab dan sub-bab, serta Caption untuk gambar dan tabel).
Batas maksimal laporan: 40 halaman.
-->

# Assignment II Big Data Analytics

**Klasifikasi Tingkat Keparahan Cedera pada Insiden Tambang yang Dilaporkan dan Analisis Jaringan Tambang–Standar Pelanggaran–Tipe Kecelakaan Menggunakan Machine Learning dan Graph Analytics**

EMMANUEL CANDRA SULISTIAWAN
2702715744

GRADUATE PROGRAM
MASTER OF COMPUTER SCIENCE
BINA NUSANTARA UNIVERSITY
2026

- Link Video: ⟦ISI⟧
- Link PPT: ⟦ISI⟧
- Link Dataset: https://arlweb.msha.gov/opengovernmentdata/ogimsha.asp
- Link Source Code: https://github.com/Ngeces/big_data_mining_accident_analysis
- Link Notebook (eksekusi): ⟦ISI: link Colab yang sudah dijalankan⟧

---

## DAFTAR ISI

*(dibuat otomatis)*

## DAFTAR GAMBAR

- Gambar 2.1 Arsitektur solusi Big Data (prototipe yang diimplementasikan)
- Gambar 4.1 Tren tahunan laporan kecelakaan MSHA 2010–2023
- Gambar 4.2 Model data relasional MSHA
- Gambar 5.1 Alur machine learning pipeline
- Gambar 5.2 Perbandingan metrik evaluasi pada test set
- Gambar 5.3 Confusion matrix model terpilih
- Gambar 5.4 Waktu komputasi Scikit-Learn vs Spark MLlib
- Gambar 6.1 Skema graf heterogen
- Gambar 6.2 Visualisasi jaringan tambang, standar, dan tipe kecelakaan
- Gambar 6.3 Sentralitas simpul (weighted degree dan PageRank)
- Gambar 6.4 Komunitas Louvain pada proyeksi mine-to-mine

## DAFTAR TABEL

- Tabel 2.1 Komponen prototipe dan padanannya pada rancangan cloud
- Tabel 4.1 Ringkasan berkas dataset MSHA
- Tabel 4.2 Pemetaan kode DEGREE_INJURY_CD ke label biner
- Tabel 4.3 Ringkasan dataset untuk machine learning
- Tabel 5.1 Kolom yang dieksklusi dari prediktor
- Tabel 5.2 Fitur prediktor
- Tabel 5.3 Hyperparameter model
- Tabel 5.4 Padanan implementasi Scikit-Learn dan Spark MLlib
- Tabel 5.5 Hasil evaluasi pada test set 2023
- Tabel 5.6 Waktu komputasi
- Tabel 6.1 Tipe simpul dan sisi pada graf
- Tabel 6.2 Statistik graf
- Tabel 6.3 Simpul dengan weighted degree tertinggi

---

# BAB 1. PENDAHULUAN

## 1.1 Latar Belakang Masalah

Kebutuhan energi dunia terus meningkat. Perkembangan teknologi, terutama *Artificial Intelligence* (AI), memerlukan energi yang sangat besar. Perusahaan teknologi seperti Google, Microsoft, dan Meta menggunakan *data center* untuk melatih dan menjalankan model AI. *Data center* memproses data dalam jumlah sangat besar sehingga membutuhkan pasokan listrik yang besar pula. Kebutuhan ini menambah beban permintaan energi global yang sudah tertekan oleh krisis pasokan.

Konflik di Timur Tengah membuat energi semakin sulit diperoleh. Rantai pasokan minyak dan gas alam terganggu karena rusaknya infrastruktur transportasi dan blokade Selat Hormuz. Negara-negara kemudian mencari sumber energi lain yang lebih mudah didapatkan.

Batu bara menjadi salah satu pilihan untuk menutup kekurangan energi tersebut karena biayanya rendah dan infrastrukturnya lebih mudah disiapkan dibandingkan energi rendah emisi seperti nuklir, angin, dan surya. Namun, sektor batu bara memiliki tingkat kecelakaan dan dampak kesehatan yang lebih tinggi dibandingkan sumber energi bersih (Markandya & Wilkinson, 2007). Kecelakaan pada kegiatan penambangan terjadi secara rutin.

Oleh karena itu, diperlukan pemantauan untuk memahami dan mencegah kecelakaan tersebut. Pendekatan *Big Data* diperlukan karena jumlah tambang yang dipantau banyak, laporan kecelakaan masuk setiap hari, dan laporan tersebut terdiri atas data terstruktur (kode, tanggal, angka) serta data tidak terstruktur (narasi kejadian).

## 1.2 Market and Business Driver

**Market driver**

- Peningkatan kebutuhan listrik *data center*. Goldman Sachs Research memperkirakan permintaan listrik *data center* tumbuh sekitar 160% pada 2030 dibandingkan 2023, sehingga kebutuhan pembangkit listrik, termasuk yang berbasis batu bara, ikut meningkat (Goldman Sachs, 2024).

**Business driver**

- Manajemen risiko untuk mengurangi kecelakaan kerja dan kerugian akibat hilangnya hari kerja.
- Kepatuhan terhadap regulasi keselamatan untuk menghindari penghentian operasi atau denda.

<!-- Catatan: versi Assignment I menyebut investasi data center AS 500 miliar USD naik menjadi 1 triliun USD sebelum 2027. Angka ini tidak ditemukan pada laporan Goldman Sachs yang dirujuk, jadi diganti dengan angka 160% yang terdapat pada sumber tersebut. Jika Anda punya sumber lain untuk angka investasi, tambahkan beserta rujukannya. -->

## 1.3 Pemangku Kepentingan yang Terlibat

- Perusahaan tambang dan pembangkit listrik berbahan bakar batu bara.
- Perusahaan pemilik *data center* sebagai konsumen energi.
- Regulator keselamatan tambang (di Amerika Serikat: Mine Safety and Health Administration / MSHA).
- Dinas sosial dan instansi ketenagakerjaan yang menangani dampak kecelakaan kerja.

---

# BAB 2. RANCANGAN SOLUSI BIG DATA

Assignment I mengusulkan arsitektur berbasis AWS (EventBridge, Lambda, S3, Glue, dan Redshift). Pada Assignment II, solusi direalisasikan sebagai prototipe yang berjalan di satu mesin (laptop atau Google Colab) dengan Apache Spark mode lokal. Setiap komponen prototipe memiliki padanan langsung pada rancangan cloud Assignment I (Tabel 2.1), sehingga prototipe dapat dipindahkan ke cloud tanpa mengubah logika pemrosesan.

![Gambar 2.1 Arsitektur solusi Big Data](figures/gambar_2_1_arsitektur.png)

*Gambar 2.1 Arsitektur solusi Big Data (prototipe yang diimplementasikan).*

*Tabel 2.1 Komponen prototipe dan padanannya pada rancangan cloud*

| Lapisan | Prototipe (Assignment II) | Rancangan cloud (Assignment I) |
|---|---|---|
| Penjadwalan ingestion | Skrip Python dijalankan manual/terjadwal | Amazon EventBridge + AWS Lambda |
| Raw zone | Folder `data/` (berkas `.txt` asli) | Amazon S3 (raw zone) |
| Curated zone | Apache Parquet + Snappy, partisi Hive-style `CAL_YR=YYYY` di filesystem lokal | Amazon S3 (Parquet) / HDFS |
| Pemrosesan terdistribusi | Apache Spark `local[*]` (Spark SQL, MLlib) | AWS Glue Spark Job |
| Gudang data analitik | Tabel Spark SQL di atas Parquet | Amazon Redshift |
| Visualisasi | Streamlit | Dashboard (Final Project) |

## 2.1 Metode Pengumpulan Data

Data diperoleh dari portal data terbuka U.S. Department of Labor, yaitu Mine Safety and Health Administration (MSHA) Open Government Data. Menurut dokumentasi MSHA, dataset diperbarui setiap minggu. Karena itu, strategi pengumpulan data menggunakan *weekly batch ingestion* dengan metode *full refresh*: MSHA menyediakan berkas utuh (format `.txt` dengan pemisah `|`) yang diganti seluruhnya pada setiap rilis, bukan aliran data (*data stream*).

Alur pengumpulan data:

1. **Unduh berkas.** Skrip Python mengunduh arsip `.zip` dari server MSHA dan mengekstrak berkas `.txt` ke *raw zone* (`data/`). Berkas mentah tidak diubah agar setiap tahap dapat diulang dari sumber yang sama.
2. **Pembacaan dengan aturan parsing yang ketat.** Seluruh kolom dibaca sebagai teks dengan *encoding* Latin-1 dan karakter kutip `"`. Pembacaan sebagai teks mencegah hilangnya angka nol di depan pada `MINE_ID` dan kode MSHA (misalnya `01`). Pada eksplorasi Assignment I ditemukan bahwa kolom teks bebas seperti `NARRATIVE` dapat mengandung karakter pemisah atau baris baru yang menggeser kolom (*column shifting*); pada notebook eksplorasi, baris yang gagal dibaca dicatat dan dihitung.
3. **Konversi ke format kolom.** Spark membaca berkas Accidents dan menuliskannya ke *curated zone* dalam format Parquet terpartisi per tahun (Subbab 2.2).

## 2.2 Teknologi Penyimpanan Data

Penyimpanan dibagi menjadi beberapa zona sesuai fungsi dan beban kerjanya:

- **Raw zone (`data/*.txt`).** Menyimpan berkas asli MSHA tanpa perubahan. Pada rancangan cloud, zona ini berada di Amazon S3, yang menjamin durabilitas data 99,999999999%.
- **Curated zone (Parquet).** Data Accidents dikonversi oleh Spark ke Apache Parquet dengan kompresi Snappy dan dipartisi per tahun kalender (`CAL_YR=YYYY`). Parquet menyimpan data per kolom sehingga kueri analitik hanya membaca kolom yang diperlukan, dan partisi per tahun memungkinkan *partition pruning* (Spark hanya membaca direktori tahun yang difilter). Pada prototipe, data disimpan di filesystem lokal (`file:///`). Format dan tata letak direktorinya sama dengan yang digunakan di HDFS atau Amazon S3. Hasil pengukuran ukuran penyimpanan: berkas mentah ⟦ISI: raw_csv_size_mb⟧ MB menjadi ⟦ISI: hadoop_parquet_size_mb⟧ MB dalam Parquet (penghematan ⟦ISI: storage_savings_pct⟧%), dengan ⟦ISI: num_partitions⟧ partisi tahun.
- **Processed zone (`data/processed/`).** Menyimpan partisi data *training*, *validation*, dan *test* yang sudah dibersihkan. Partisi yang sama dibaca oleh Scikit-Learn dan Spark MLlib agar perbandingan kedua framework adil.
- **Artifacts (`artifacts/`).** Menyimpan model terlatih, preprocessor, metadata, hasil evaluasi, dan hasil graph analytics yang dibaca oleh dashboard.

## 2.3 Processing Framework

Pemrosesan menggunakan Apache Spark (PySpark) dalam mode `local[*]`, yaitu Spark memakai seluruh inti CPU pada satu mesin. Kode yang sama dapat dijalankan di klaster (misalnya AWS Glue atau Amazon EMR) dengan mengganti konfigurasi *master*.

Apache Spark dipilih karena memproses data secara *in-memory* menggunakan *Resilient Distributed Dataset* (RDD) dan DataFrame (Zaharia et al., 2016). Berbeda dengan Hadoop MapReduce yang menulis hasil antara (*intermediate results*) ke disk pada setiap tahap Map dan Reduce, Spark menyimpan hasil antara di memori dan melakukan *spill to disk* bila memori tidak cukup.

Komponen pemrosesan yang digunakan:

| Komponen | Fungsi |
|---|---|
| Spark SQL | Konversi ke Parquet dan kueri agregasi per tahun |
| Spark MLlib | Model Logistic Regression dan Random Forest terdistribusi (Bab 5) |
| pandas + Scikit-Learn | Pembersihan data, partisi temporal, dan model pembanding tanpa Spark (Bab 5) |
| NetworkX | Pembentukan graf dan perhitungan sentralitas serta komunitas (Bab 6) |
| Streamlit | Dashboard interaktif yang membaca artefak hasil pemrosesan |

---

# BAB 3. TATA KELOLA IT

## 3.1 Security

Perubahan data oleh pihak yang tidak berwenang dapat memengaruhi hasil model. Oleh karena itu, akses ditulis dan dibaca dipisahkan:

- **Pemisahan hak akses.** Raw zone hanya dapat ditulis oleh proses ingestion. Analis hanya memiliki akses baca. Pada rancangan cloud, pembatasan ini diterapkan dengan kebijakan AWS IAM.
- **Verifikasi sumber.** Hasil olahan dapat diverifikasi ulang dengan mengunduh kembali berkas dari situs MSHA karena raw zone tidak pernah diubah.
- **Pengelolaan rahasia.** Kunci API dan kredensial tidak boleh ditulis di dalam kode sumber atau disimpan di repository. Kredensial dibaca dari variabel lingkungan atau *secret manager* (misalnya `st.secrets` pada Streamlit atau AWS Secrets Manager).

## 3.2 Compliance

Dataset MSHA merupakan data pemerintah Amerika Serikat yang dipublikasikan sebagai data terbuka dan bebas digunakan untuk analisis. Pada rancangan cloud, data disimpan di region bucket yang dipilih sehingga lokasi penyimpanan dapat dikendalikan. Sumber, tanggal unduh, dan periode data dicatat dalam laporan agar hasil dapat direproduksi.

## 3.3 Data Privacy

Dataset merupakan data publik yang tidak memuat nama pekerja. Meskipun demikian, kolom narasi (`NARRATIVE`) tidak digunakan sebagai fitur dan tidak ditampilkan di dashboard, dan dashboard hanya menampilkan data agregat per tambang.

## 3.4 Data Quality

Data merupakan laporan kecelakaan, sehingga tidak setiap hari memiliki laporan dan sebagian kolom tidak selalu terisi. Penanganan kualitas data yang diterapkan:

| Masalah | Penanganan |
|---|---|
| Sel kosong pada fitur kategorikal | Diberi label `MISSING` |
| Penanda MSHA untuk nilai tidak diketahui (`?`, `NO VALUE FOUND`) | Pada label: dieksklusi. Pada fitur: dipertahankan sebagai kategori tersendiri. Pada graf: tipe kecelakaan dan standar dengan penanda ini dibuang |
| Angka nol di depan pada `MINE_ID` dan kode MSHA | Seluruh kolom dibaca sebagai teks; kode cedera dinormalisasi menjadi dua digit (`1` → `01`) |
| Nilai kosong pada fitur numerik pengalaman kerja | Diisi median yang dihitung **hanya dari data training** |
| Kategori dengan jumlah sangat banyak (peralatan, aktivitas) | 15 kategori terbanyak pada data training dipertahankan, sisanya digabung menjadi `OTHER` |
| Kode cedera di luar cakupan (07, 08, 09) atau kosong | Dieksklusi dan jumlahnya dicatat (Tabel 4.2) |
| Pergeseran kolom akibat teks bebas | Dideteksi pada notebook eksplorasi (baris gagal dibaca dicatat); kolom `NARRATIVE` tidak dibaca oleh pipeline machine learning |
| Waktu shift berformat HHMM yang tidak valid | Dikategorikan sebagai `UNKNOWN` |

Kode eksplorasi kualitas data: https://colab.research.google.com/drive/1IKOqwyx-5LudkYKyRNBlRFOzULQMKCMS?usp=sharing

---

# BAB 4. DATA

## 4.1 Deskripsi Data

Data diperoleh dari portal data terbuka U.S. Department of Labor (MSHA). Penelitian ini menggunakan empat berkas `.txt`:

*Tabel 4.1 Ringkasan berkas dataset MSHA*

| Berkas | Ukuran | Isi |
|---|---|---|
| Mines.txt | 37,55 MB | Profil tambang: lokasi, status, jenis tambang, operator, koordinat |
| Accidents.txt | 218,08 MB | Laporan kecelakaan, cedera, dan penyakit (MSHA Form 7000-1): waktu, lokasi, peralatan, aktivitas, tingkat cedera, narasi kejadian |
| Inspections.txt | 331,97 MB | Laporan inspeksi oleh MSHA |
| Violations.txt | 1.375,41 MB | Pelanggaran standar keselamatan yang ditemukan pada inspeksi |

Data sebagian besar berbentuk tabel terstruktur, dengan satu kolom tidak terstruktur yaitu narasi kecelakaan.

**Karakteristik 5V**

- **Volume:** sekitar 2 GB dengan sekitar 4,6 juta baris.
- **Variety:** tanggal, lokasi dan koordinat, kode kategori, angka pengukuran (pengalaman kerja, hari kerja hilang), dan teks narasi.
- **Velocity:** tidak *real-time*, tetapi data baru dirilis setiap minggu.
- **Value:** data dapat digunakan untuk memahami pola kecelakaan, tingkat keparahan cedera, dan pelanggaran yang ditemukan pada inspeksi.
- **Veracity:** tingkat nilai kosong berbeda antar kolom (hasil profiling pada notebook Assignment I), dan beberapa kolom memakai penanda khusus seperti `?` atau `NO VALUE FOUND` untuk nilai yang tidak diketahui.

Gambar 4.1 menunjukkan jumlah laporan kecelakaan per tahun pada 2010–2023.

⟦ISI: tangkapan layar grafik "Tren tahunan laporan kecelakaan" dari halaman Overview dashboard⟧

*Gambar 4.1 Tren tahunan laporan kecelakaan MSHA 2010–2023*

## 4.2 Model Data

Data terstruktur disimpan sebagai tabel relasional. Relasi antartabel:

- `MINE_ID` terdapat pada keempat tabel dan menghubungkan kecelakaan, inspeksi, dan pelanggaran dengan profil tambang.
- Inspections dan Violations dihubungkan melalui `EVENT_NO`.
- Narasi kecelakaan (data tidak terstruktur) tetap terhubung ke data terstruktur melalui `DOCUMENT_NO`.

![Gambar 4.2 Model data relasional MSHA](figures/gambar_4_2_model_data.png)

*Gambar 4.2 Model data relasional MSHA*

Untuk Graph Analytics (Bab 6), relasi tersebut juga dimodelkan sebagai graf: tambang, standar yang dilanggar, dan tipe kecelakaan menjadi simpul, sedangkan frekuensi kejadian menjadi bobot sisi.

## 4.3 Dataset untuk Diolah Menggunakan Machine Learning

**Data yang dipilih.** Machine learning menggunakan berkas **Accidents.txt** (data terstruktur). Graph Analytics menggunakan Accidents.txt dan Violations.txt, sedangkan Mines.txt dipakai untuk koordinat dan atribut tambang di dashboard.

**Unit data.** Satu baris adalah satu laporan insiden (satu formulir MSHA 7000-1).

**Label.** Target biner `IS_LOST_TIME_OR_SEVERE` dibentuk dari kolom `DEGREE_INJURY_CD`:

*Tabel 4.2 Pemetaan kode DEGREE_INJURY_CD ke label biner*

| Kode | Deskripsi MSHA | Label | Jumlah baris (seluruh data) |
|---|---|---|---|
| 01 | Fatality | 1 | ⟦ISI⟧ |
| 02 | Permanent total or permanent partial disability | 1 | ⟦ISI⟧ |
| 03 | Days away from work only | 1 | ⟦ISI⟧ |
| 04 | Days away from work and restricted activity | 1 | ⟦ISI⟧ |
| 05 | Days restricted activity only | 1 | ⟦ISI⟧ |
| 00 | Accident only | 0 | ⟦ISI⟧ |
| 06 | No days away from work, no restrictions | 0 | ⟦ISI⟧ |
| 10 | All other cases (termasuk *first aid*) | 0 | ⟦ISI⟧ |
| 07 | Occupational illness not degree 1–6 | Dieksklusi | ⟦ISI⟧ |
| 08 | Injuries due to natural causes | Dieksklusi | ⟦ISI⟧ |
| 09 | Injuries involving non-employees | Dieksklusi | ⟦ISI⟧ |
| kosong / `?` | Tidak ada nilai | Dieksklusi | ⟦ISI⟧ |

*Sumber jumlah baris: `label_profile_raw` pada `artifacts/preprocessing_metadata.json`. Cocokkan deskripsi kode dengan `Accidents_Definition_File.txt` dari MSHA.*

**Periode dan partisi.** Data dibatasi pada tahun kalender 2018–2023 dan dibagi secara temporal:

*Tabel 4.3 Ringkasan dataset untuk machine learning*

| Partisi | Periode | Jumlah baris | Proporsi label 1 | Fungsi |
|---|---|---|---|---|
| Training | 2018–2021 | 23.484 ⟦CEK⟧ | ⟦ISI⟧ | Fit preprocessing dan model |
| Validation | 2022 | 5.682 ⟦CEK⟧ | ⟦ISI⟧ | Pemilihan model dan *threshold* |
| Test | 2023 | 5.765 ⟦CEK⟧ | ⟦ISI⟧ | Evaluasi akhir (sekali) |
| **Total** | 2018–2023 | 34.931 ⟦CEK⟧ | ⟦ISI⟧ | |

*Sumber: `split` pada `artifacts/preprocessing_metadata.json`.*

> ✍️ **BAGIAN MAHASISWA — Deskripsi masalah dan alasan pemilihan data**
>
> Tuliskan dengan kata-kata Anda sendiri:
> - Masalah apa yang diselesaikan, dan mengapa ini klasifikasi keparahan **setelah insiden dilaporkan**, bukan prediksi apakah kecelakaan akan terjadi.
> - Mengapa Anda memilih Accidents.txt untuk machine learning.
> - Mengapa kode 05 (*restricted activity only*) Anda masukkan ke label 1, dan mengapa kode 07, 08, 09 dieksklusi.
> - Mengapa Anda memilih periode 2018–2023 dan pembagian temporal, bukan pembagian acak.

---

# BAB 5. MACHINE LEARNING UNTUK BIG DATA

Gambar 5.1 menunjukkan alur pipeline yang diimplementasikan. Kode sumber berada di folder `src/` pada repository, dan seluruh tahap dapat dijalankan dengan `python src/run_all.py`.

![Gambar 5.1 Alur machine learning pipeline](figures/gambar_5_1_pipeline_ml.png)

*Gambar 5.1 Alur machine learning pipeline*

## 5.1 Data Preprocessing

Preprocessing diimplementasikan di `src/data_pipeline.py`:

1. **Pembacaan.** Hanya kolom yang diperlukan dibaca, seluruhnya sebagai teks. Spasi dan tanda kutip dibuang, dan string kosong diubah menjadi nilai kosong.
2. **Normalisasi kode cedera.** `DEGREE_INJURY_CD` dinormalisasi menjadi dua digit, lalu distribusi seluruh kode dicatat (Tabel 4.2).
3. **Pembentukan label dan eksklusi.** Baris dengan kode 01–05 diberi label 1, kode 00, 06, dan 10 diberi label 0, dan baris lainnya dieksklusi.
4. **Filter periode.** Hanya baris dengan `CAL_YR` 2018–2023 yang digunakan.
5. **Pembersihan fitur.** Nilai kategorikal yang kosong diberi label `MISSING`. Nilai pengalaman kerja diubah menjadi angka; nilai yang tidak valid menjadi kosong.
6. **Partisi temporal.** Data dibagi menjadi training (2018–2021), validation (2022), dan test (2023).
7. **Fit hanya pada training.** Median pengalaman kerja dan daftar 15 kategori terbanyak dihitung dari data training saja, lalu diterapkan ke ketiga partisi. Dengan cara ini, informasi dari tahun 2022 dan 2023 tidak bocor ke proses training.
8. **Penyimpanan.** Ketiga partisi disimpan sebagai CSV, dan parameter preprocessing disimpan di `preprocessing_metadata.json`.

**Pencegahan target leakage.** Kolom yang nilainya ditentukan oleh tingkat cedera atau baru diketahui setelah penanganan medis tidak dibaca sama sekali sebagai prediktor:

*Tabel 5.1 Kolom yang dieksklusi dari prediktor*

| Kolom | Alasan eksklusi |
|---|---|
| `DEGREE_INJURY_CD`, `DEGREE_INJURY` | Sumber label |
| `DAYS_LOST`, `DAYS_RESTRICT`, `SCHEDULE_CHARGE` | Dihitung dari tingkat cedera |
| `RETURN_TO_WORK_DT`, `TRANS_TERM`, `CLOSED_DOC_NO` | Baru diketahui setelah kasus ditutup |
| `INVEST_BEGIN_DT`, `IMMED_NOTIFY_CD`, `IMMED_NOTIFY` | Tindak lanjut yang bergantung pada keparahan |
| `NATURE_INJURY`, `INJ_BODY_PART`, `INJURY_SOURCE` (dan kodenya) | Diagnosis klinis setelah kejadian |
| `NARRATIVE` | Teks bebas yang sering menyebut hasil cedera |

## 5.2 Feature Engineering

*Tabel 5.2 Fitur prediktor*

| Fitur | Tipe | Transformasi |
|---|---|---|
| `SUBUNIT` | Kategorikal | One-hot encoding |
| `COAL_METAL_IND` | Kategorikal | One-hot encoding |
| `ACCIDENT_TYPE` | Kategorikal | One-hot encoding |
| `MINING_EQUIP` | Kategorikal | Top-15 dari training + `OTHER`, lalu one-hot |
| `ACTIVITY` | Kategorikal | Top-15 dari training + `OTHER`, lalu one-hot |
| `SHIFT_PERIOD` | Kategorikal (turunan) | Dari `SHIFT_BEGIN_TIME` (HHMM): DAY 06–14, EVENING 14–22, NIGHT 22–06, UNKNOWN |
| `TOT_EXPER` | Numerik | Imputasi median training; standardisasi untuk LR |
| `MINE_EXPER` | Numerik | Imputasi median training; standardisasi untuk LR |
| `JOB_EXPER` | Numerik | Imputasi median training; standardisasi untuk LR |

Detail implementasi:

- **One-hot encoding** mempertahankan semua kategori yang dilihat pada training. Kategori baru pada validation atau test diubah menjadi vektor nol (`handle_unknown="ignore"` di Scikit-Learn dan `handleInvalid="keep"` + `dropLast=True` di Spark).
- **Standardisasi** (rata-rata 0, simpangan baku 1) hanya diterapkan pada fitur numerik untuk Logistic Regression. Random Forest tidak memerlukan standardisasi karena pemisahan pada pohon keputusan tidak bergantung pada skala.
- **Ketidakseimbangan kelas** ditangani dengan *class weight* "balanced": bobot kelas *c* = jumlah baris training / (2 × jumlah baris kelas *c*). Rumus yang sama dipakai di kedua framework.

> ✍️ **BAGIAN MAHASISWA — Alasan feature engineering**
>
> Jelaskan alasan Anda untuk:
> - Pemilihan setiap fitur, terutama mengapa peralatan (`MINING_EQUIP`) dan aktivitas dipakai.
> - Pembatasan Top-15 kategori dan penggabungan `OTHER`.
> - Pembagian shift menjadi DAY, EVENING, dan NIGHT dengan batas jam tersebut.
> - Penggunaan median untuk imputasi.
> - Penggunaan *class weight*, bukan *oversampling* atau *undersampling*.

## 5.3 Algoritma Machine Learning

Dua algoritma klasifikasi biner digunakan, masing-masing diimplementasikan di Scikit-Learn dan Spark MLlib.

**Logistic Regression.** Model linier yang memodelkan peluang label 1 sebagai *p* = σ(**w**·**x** + *b*), dengan σ fungsi sigmoid. Parameter dipelajari dengan meminimalkan *log-loss* berbobot ditambah regularisasi L2. Koefisien **w** dapat dibaca langsung sebagai arah dan besar pengaruh setiap fitur.

**Random Forest.** *Ensemble* dari banyak pohon keputusan (Breiman, 2001). Setiap pohon dilatih pada sampel *bootstrap* dan pada setiap pemisahan hanya mempertimbangkan subset acak fitur (√jumlah fitur). Prediksi adalah rata-rata peluang dari seluruh pohon. Random Forest dapat menangkap interaksi antarfitur yang tidak linier.

*Tabel 5.3 Hyperparameter model (sama untuk kedua framework, diatur di `src/config.py`)*

| Model | Hyperparameter | Nilai |
|---|---|---|
| Logistic Regression | Regularisasi | L2, C = 1,0 (Spark: `regParam` = 1 / (C × jumlah bobot training)) |
| Logistic Regression | Iterasi maksimum | 1.000 |
| Random Forest | Jumlah pohon | 100 |
| Random Forest | Kedalaman maksimum | 15 |
| Random Forest | Fitur per pemisahan | √jumlah fitur |
| Keduanya | *Class weight* | balanced |
| Keduanya | *Random seed* | 42 |
| Keduanya | Grid *threshold* | 0,05–0,95 dengan langkah 0,01 |

> ✍️ **BAGIAN MAHASISWA — Alasan pemilihan algoritma**
>
> Jelaskan mengapa Anda memilih Logistic Regression dan Random Forest untuk masalah ini, apa kelebihan dan kekurangan masing-masing terhadap karakteristik data Anda, dan mengapa nilai hyperparameter pada Tabel 5.3 dipilih.

## 5.4 Proses Pengembangan Model dengan dan tanpa Spark MLlib

**Tanpa Spark (Scikit-Learn) — `src/models_baseline.py`**

1. Partisi training, validation, dan test dibaca dengan pandas.
2. `ColumnTransformer` menggabungkan `OneHotEncoder` untuk fitur kategorikal dan `StandardScaler` (khusus LR) untuk fitur numerik. Transformer di-fit pada training, lalu diterapkan ke validation dan test.
3. `LogisticRegression` dan `RandomForestClassifier` dilatih pada training. Waktu fit diukur.
4. Peluang pada validation dihitung, lalu *threshold* dengan F1 kelas 1 tertinggi dipilih.
5. Model dengan F1 validation tertinggi disimpan sebagai model dashboard (`best_model.joblib`, `preprocessor.joblib`, `model_metadata.json`).
6. Peluang pada test dihitung sekali dan dievaluasi dengan *threshold* hasil langkah 4. Waktu inferensi diukur.

**Dengan Spark MLlib — `src/models_spark.py`**

1. `SparkSession` dibuat dalam mode `local[*]` (memori driver 4 GB).
2. Partisi yang sama dibaca sebagai Spark DataFrame tanpa inferensi skema; kolom numerik di-*cast* secara eksplisit.
3. Kolom `classWeight` ditambahkan pada training dengan rumus *balanced*.
4. Spark ML `Pipeline` berisi `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, dan `StandardScaler` (hanya untuk fitur numerik LR). Pipeline di-fit pada training.
5. `LogisticRegression` dan `RandomForestClassifier` dari `pyspark.ml.classification` dilatih dengan `weightCol="classWeight"`. Data training di-*cache* di memori sebelum training.
6. Kolom `probability` diubah menjadi array; peluang kelas 1 pada validation dan test diambil, lalu *threshold* dipilih dan metrik dihitung dengan fungsi yang sama seperti Scikit-Learn (`src/evaluation.py`).

*Tabel 5.4 Padanan implementasi Scikit-Learn dan Spark MLlib*

| Aspek | Scikit-Learn | Spark MLlib |
|---|---|---|
| Struktur data | pandas DataFrame / sparse matrix | Spark DataFrame (terdistribusi, *lazy*) |
| Encoding kategorikal | `OneHotEncoder(handle_unknown="ignore")` | `StringIndexer(handleInvalid="keep")` + `OneHotEncoder(dropLast=True)` |
| Penggabungan fitur | `ColumnTransformer` | `VectorAssembler` |
| Standardisasi (LR) | `StandardScaler` pada numerik | `StandardScaler(withMean=True)` pada numerik |
| Class weight | `class_weight="balanced"` | `weightCol` dengan rumus yang sama |
| Eksekusi | Satu proses (RF memakai semua inti via `n_jobs=-1`) | Tugas dibagi ke *executor* (di sini semua inti lokal) |
| Pemilihan *threshold* | Validation set | Validation set |
| Perhitungan metrik | `src/evaluation.py` | `src/evaluation.py` |

**Pengukuran waktu.** *Fit time* adalah waktu `fit()` model saja. Waktu inisialisasi `SparkSession` dan fit pipeline preprocessing Spark dicatat terpisah (`Spark_Init_Seconds`, `Preprocess_Fit_Seconds`). *Inference time* adalah waktu menghasilkan prediksi untuk seluruh test set.

**Lingkungan eksekusi:** ⟦ISI: CPU, RAM, sistem operasi, versi Python, Java, PySpark, Scikit-Learn⟧.

## 5.5 Evaluasi Model

Kelas positif adalah label 1 (Lost-Time / Severe). Dengan TP, FP, TN, dan FN dari *confusion matrix*:

- **Accuracy** = (TP + TN) / total.
- **Precision (kelas 1)** = TP / (TP + FP): dari insiden yang diprediksi Lost-Time / Severe, berapa yang benar.
- **Recall (kelas 1)** = TP / (TP + FN): dari insiden Lost-Time / Severe, berapa yang terdeteksi.
- **F1 (kelas 1)** = 2 × Precision × Recall / (Precision + Recall).
- **F1 Macro**: rata-rata F1 kelas 0 dan kelas 1.
- **ROC-AUC**: luas area di bawah kurva TPR terhadap FPR untuk semua *threshold*; tidak bergantung pada *threshold* yang dipilih.
- **PR-AUC** (*average precision*): luas area di bawah kurva Precision–Recall. Nilai acuan model acak sama dengan proporsi kelas 1 pada test set (Saito & Rehmsmeier, 2015).

Sebagai pembanding minimum, digunakan **baseline kelas mayoritas** yang selalu memprediksi kelas terbanyak pada training.

*Tabel 5.5 Hasil evaluasi pada test set 2023 (sumber: `artifacts/benchmark_results.json`)*

| Model | Threshold | Val F1 | Accuracy | Precision (C1) | Recall (C1) | F1 (C1) | F1 Macro | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baseline kelas mayoritas | – | – | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | 0,5000 | ⟦ISI⟧ |
| Scikit-Learn Logistic Regression | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |
| Scikit-Learn Random Forest | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |
| Spark MLlib Logistic Regression | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |
| Spark MLlib Random Forest | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |

*Tabel 5.6 Waktu komputasi (detik)*

| Model | Fit | Inferensi test | Inisialisasi Spark | Fit preprocessing Spark |
|---|---|---|---|---|
| Scikit-Learn Logistic Regression | ⟦ISI⟧ | ⟦ISI⟧ | – | – |
| Scikit-Learn Random Forest | ⟦ISI⟧ | ⟦ISI⟧ | – | – |
| Spark MLlib Logistic Regression | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |
| Spark MLlib Random Forest | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |

⟦ISI: tangkapan layar halaman Benchmark, tab Metrik⟧

*Gambar 5.2 Perbandingan metrik evaluasi pada test set*

⟦ISI: tangkapan layar tab Confusion matrix untuk model terpilih⟧

*Gambar 5.3 Confusion matrix model terpilih*

⟦ISI: tangkapan layar tab Waktu komputasi⟧

*Gambar 5.4 Waktu komputasi Scikit-Learn vs Spark MLlib*

**Model untuk dashboard.** Model yang dipakai di halaman Simulasi Keparahan adalah ⟦ISI: `model_name` dari `model_metadata.json`⟧ dengan *threshold* ⟦ISI⟧, dipilih berdasarkan F1 kelas 1 pada validation set.

## 5.6 Analisis

> ✍️ **BAGIAN MAHASISWA — Interpretasi dan perbandingan model**
>
> Bagian ini wajib ditulis sendiri. Pertanyaan pemandu:
> - Bandingkan setiap model dengan baseline kelas mayoritas pada Tabel 5.5. Metrik mana yang paling relevan untuk masalah ini, dan mengapa?
> - Apa arti *threshold* yang terpilih terhadap keseimbangan precision dan recall? Apa konsekuensi FN (insiden severe yang terlewat) dibandingkan FP dalam konteks keselamatan tambang?
> - Bagaimana perbedaan hasil Logistic Regression dan Random Forest? Apa kemungkinan penyebabnya?
> - Apakah hasil Scikit-Learn dan Spark MLlib untuk algoritma yang sama berbeda? Jika ya, faktor implementasi apa yang dapat menjelaskannya (lihat Tabel 5.4)?
> - Bagaimana Anda menjelaskan perbedaan waktu komputasi pada Tabel 5.6 untuk ukuran data ini, dan kapan Spark akan lebih menguntungkan?
> - Apa keterbatasan model ini (fitur yang tersedia, periode data, kemungkinan perubahan pola antartahun)?

---

# BAB 6. GRAPH ANALYTICS

## 6.1 Pengembangan Graph

Graph analytics diimplementasikan di `src/graph_analytics.py` dengan NetworkX (Hagberg et al., 2008).

**Sumber dan periode.** Accidents.txt dan Violations.txt dibatasi pada periode yang sama, 2020–2023, agar sisi graf menggambarkan kejadian pada rentang waktu yang sama. Violations.txt (±1,4 GB) dibaca per *chunk* 200.000 baris dan langsung diagregasi, sehingga tidak seluruh berkas dimuat ke memori.

**Skema graf.** Graf heterogen tidak berarah dengan tiga tipe simpul dan dua tipe sisi. Setiap ID simpul diberi awalan tipe untuk mencegah tabrakan ID.

*Tabel 6.1 Tipe simpul dan sisi pada graf*

| Elemen | Tipe | Sumber kolom | Keterangan |
|---|---|---|---|
| Simpul | `mine::<MINE_ID>` | Accidents, Violations | Tambang |
| Simpul | `standard::<PART_SECTION>` | Violations | Pasal standar 30 CFR yang dilanggar |
| Simpul | `accident_type::<ACCIDENT_TYPE>` | Accidents | Tipe kecelakaan |
| Sisi | `CITED_FOR` (tambang – standar) | Violations | Bobot = jumlah pelanggaran standar tersebut di tambang itu |
| Sisi | `REPORTED_ACCIDENT` (tambang – tipe) | Accidents | Bobot = jumlah kecelakaan tipe tersebut di tambang itu |

![Gambar 6.1 Skema graf heterogen](figures/gambar_6_1_skema_graf.png)

*Gambar 6.1 Skema graf heterogen*

**Analisis yang dijalankan.**

1. **Weighted degree**: jumlah bobot sisi pada setiap simpul, yaitu total pelanggaran dan kecelakaan yang terhubung.
2. **PageRank** berbobot (Page et al., 1999): simpul bernilai tinggi jika terhubung dengan simpul lain yang juga penting.
3. **Afinitas dan *lift* tambang–tipe kecelakaan**: untuk tambang dengan minimal 3 kecelakaan, proporsi setiap tipe kecelakaan di tambang dibandingkan dengan proporsi rata-rata seluruh tambang tersebut. *Lift* = proporsi di tambang ÷ proporsi rata-rata.
4. **Kemiripan antartambang**: *cosine similarity* antar vektor jumlah tipe kecelakaan per tambang; 5 tambang paling mirip disimpan untuk setiap tambang.
5. **Proyeksi mine-to-mine dan deteksi komunitas**: 300 tambang dengan weighted degree tertinggi diproyeksikan menjadi graf tambang–tambang. Dua tambang dihubungkan jika indeks Jaccard himpunan tetangganya (standar dan tipe kecelakaan) ≥ 0,20, dengan bobot sisi = nilai Jaccard. Komunitas dicari dengan algoritma Louvain (Blondel et al., 2008), *seed* 42.

*Tabel 6.2 Statistik graf (sumber: `graph_top_edges.json`, `graph_nodes_summary.csv`)*

| Statistik | Nilai |
|---|---|
| Jumlah simpul | 16.463 ⟦CEK⟧ |
| – tambang | 13.168 ⟦CEK⟧ |
| – standar 30 CFR | 3.253 ⟦CEK⟧ |
| – tipe kecelakaan | 42 ⟦CEK⟧ |
| Jumlah sisi | 165.982 ⟦CEK⟧ |
| Tambang pada proyeksi | 300 |
| Komunitas Louvain | 4 ⟦CEK⟧ (berukuran 141, 98, 44, dan 17 tambang ⟦CEK⟧) |
| Tambang dengan profil afinitas (≥ 3 kecelakaan) | 1.526 ⟦CEK⟧ |

**Batasan interpretasi.**

- Sisi menunjukkan bahwa pelanggaran dan kecelakaan tercatat di tambang yang sama pada periode yang sama, bukan hubungan sebab-akibat antara pelanggaran tertentu dan kecelakaan tertentu.
- Weighted degree adalah frekuensi laporan dan belum dinormalisasi terhadap jumlah pekerja, jam kerja, atau jumlah inspeksi. Tambang besar atau yang lebih sering diinspeksi cenderung memiliki weighted degree lebih tinggi.
- Proyeksi Jaccard tidak memakai bobot sisi; dua tambang dianggap mirip berdasarkan kesamaan jenis standar dan tipe kecelakaan, bukan jumlahnya.
- Tambang di luar 300 teratas tidak memiliki label komunitas.

> ✍️ **BAGIAN MAHASISWA — Alasan desain graf**
>
> Jelaskan mengapa Anda memilih periode 2020–2023, batas minimal 3 kecelakaan, 300 tambang teratas, dan ambang Jaccard 0,20.

## 6.2 Analisis dan Insight yang Diperoleh

*Tabel 6.3 Simpul dengan weighted degree tertinggi (sumber: halaman Graph Analytics, tab Sentralitas)*

| Peringkat | Tambang | Weighted degree | Standar 30 CFR | Weighted degree | Tipe kecelakaan | Weighted degree |
|---|---|---|---|---|---|---|
| 1 | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |
| 2 | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |
| 3 | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |
| 4 | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |
| 5 | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ | ⟦ISI⟧ |

⟦ISI: tangkapan layar tab Jaringan⟧

*Gambar 6.2 Visualisasi jaringan tambang, standar, dan tipe kecelakaan*

⟦ISI: tangkapan layar tab Sentralitas⟧

*Gambar 6.3 Sentralitas simpul (weighted degree dan PageRank)*

⟦ISI: tangkapan layar tab Komunitas⟧

*Gambar 6.4 Komunitas Louvain pada proyeksi mine-to-mine*

> ✍️ **BAGIAN MAHASISWA — Analisis graph analytics**
>
> Bagian ini wajib ditulis sendiri. Pertanyaan pemandu:
> - Standar 30 CFR dan tipe kecelakaan apa yang paling banyak terhubung? Apa makna praktisnya bagi inspektur atau operator tambang?
> - Apakah urutan berdasarkan weighted degree berbeda dengan urutan PageRank? Apa artinya?
> - Apa karakteristik setiap komunitas Louvain (negara bagian, tipe tambang, standar atau tipe kecelakaan yang dominan)? Apakah komunitas tersebut masuk akal secara operasional?
> - Pilih satu atau dua tambang di tab Inspector. Tipe kecelakaan apa yang memiliki *lift* tinggi, dan seberapa andal angka tersebut mengingat jumlah kasusnya?
> - Bagaimana batasan pada Subbab 6.1 memengaruhi kesimpulan Anda?

---

# KESIMPULAN

> ✍️ **BAGIAN MAHASISWA**
>
> Rangkum jawaban atas masalah pada Subbab 4.3, hasil utama Bab 5 dan Bab 6, keterbatasan, dan rencana pengembangan untuk Final Project (dashboard dan insight).

---

# REFERENSI

Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics: Theory and Experiment, 2008*(10), P10008. https://doi.org/10.1088/1742-5468/2008/10/P10008

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Goldman Sachs. (2024). *Generational growth: AI, data centers and the coming US power demand surge*. Goldman Sachs Research. https://www.goldmansachs.com/insights/goldman-sachs-research/generational-growth-ai-data-centers-and-the-coming-us-power-demand-surge

Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. In *Proceedings of the 7th Python in Science Conference (SciPy 2008)* (pp. 11–15).

International Energy Agency. (2023). *Coal 2023: Analysis and forecast to 2026*. https://www.iea.org/reports/coal-2023

International Energy Agency. (2024). *Electricity 2024: Analysis and forecast to 2026*. https://www.iea.org/reports/electricity-2024

Markandya, A., & Wilkinson, P. (2007). Electricity generation and health. *The Lancet, 370*(9591), 979–990. https://doi.org/10.1016/S0140-6736(07)61253-7

Meng, X., Bradley, J., Yavuz, B., Sparks, E., Venkataraman, S., Liu, D., ... & Talwalkar, A. (2016). MLlib: Machine learning in Apache Spark. *Journal of Machine Learning Research, 17*(34), 1–7.

Mine Safety and Health Administration. (n.d.). *Open government initiative portal: Data sets*. U.S. Department of Labor. Diakses ⟦ISI: tanggal unduh⟧, dari https://arlweb.msha.gov/opengovernmentdata/ogimsha.asp

Mine Safety and Health Administration. (n.d.). *Mine accident, injury and illness report (MSHA Form 7000-1)*. U.S. Department of Labor. https://www.msha.gov/support-resources/forms-online-filing/7000-1/mine-accident-injury-and-illness-report

Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank citation ranking: Bringing order to the web* (Technical Report 1999-66). Stanford InfoLab.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

U.S. Energy Information Administration. (2023). *World oil transit chokepoints*. https://www.eia.gov/international/analysis/specialtopics/World_Oil_Transit_Chokepoints

Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Meng, X., ... & Stoica, I. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM, 59*(11), 56–65. https://doi.org/10.1145/2934664

<!--
Dua referensi dari Assignment I tidak dapat diverifikasi dan sudah dihapus:
  - Song, Jiao, & Dong (2020), IEEE Access 8, doi 10.1109/ACCESS.2020.3029123
  - Zhang, Wang, & Liu (2022), Safety Science 148, 105658, doi 10.1016/j.ssci.2021.105658
Pencarian DOI tersebut tidak menemukan artikel dengan judul dan penulis itu. Buka DOI-nya di https://doi.org/
Jika artikelnya ada, tambahkan kembali. Jika tidak ada, jangan dipakai.
-->

---

# LAMPIRAN A. CARA MENJALANKAN KODE

```bash
git clone https://github.com/Ngeces/big_data_mining_accident_analysis.git
cd big_data_mining_accident_analysis
pip install -r requirements.txt
# letakkan Accidents.txt, Violations.txt, Mines.txt di folder data/
python src/run_all.py
streamlit run app.py
```

# LAMPIRAN B. AI USE DECLARATION FORM (DRAF)

| | |
|---|---|
| Name | Emmanuel Candra Sulistiawan |
| Student ID | 2702715744 |
| Course Code | COMP8035041 |
| Course Title | Big Data Analytics |
| Assignment Title | Assignment II – Pengolahan Big Data Menggunakan Machine Learning |
| Assignment Type | ⟦ISI⟧ |

| Assignment part | Used AI tool? | Tool(s) used | Key prompt |
|---|---|---|---|
| Idea and content discussion | ⟦Yes/No⟧ | ⟦ISI⟧ | ⟦ISI⟧ |
| Text/Script generation | Yes | Kiro (Claude) ⟦tambahkan tool lain yang dipakai⟧ | Menulis dan me-review kode Python (pipeline, Spark MLlib, NetworkX, Streamlit); menyusun draf bagian deskriptif laporan |
| Image/Video/Graphics generation | Yes | Kiro (Claude) | Membuat grafik dan visualisasi dashboard |
| Translation, grammar, and spelling check | Yes | Kiro (Claude) | Memperbaiki tata bahasa Bab 1–4 |
| Others: debugging | Yes | Kiro (Claude) | Menjelaskan error dan memperbaiki bug kode |

Approximate contribution of AI in the final submission of assignment is ⟦ISI⟧%.

<!-- Batas Assignment II adalah 50%. Isi persentase dengan jujur sesuai porsi akhir. -->
