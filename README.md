# Structural Damage Classification — Gempa Nepal 2015

**Tugas 1 Supervised Machine Learning, Sains Data ITS · 010_Constantine**

Prediksi tingkat kerusakan bangunan akibat gempa Nepal 2015 menggunakan data survei bangunan, dengan lima label `damage_grade` (1 sampai 5). Metrik kompetisi adalah **Macro F1**, bukan accuracy. Proyek ini dikerjakan tanpa neural network, sesuai ketentuan tugas.

[Kompetisi Kaggle](https://www.kaggle.com/t/a9196b92680b4492aeb06aeaee94f654) · [Instruksi tugas](https://docs.google.com/document/d/1e8SbPqX3BX6_hsmawrtFu9kn80WotSfbORxYNg6cNBE/edit)

## Hasil yang saya pertahankan

| Catatan | Hasil | Keterangan |
|---|---:|---|
| **Best public leaderboard** | **0,60627** | Submission ke-10, `010_Constantine(10).csv` |
| Posisi pada tangkapan layar 13 Oktober 2026 | **1 dari 25 peserta** | Posisi sementara, bukan hasil private |
| Submission terakhir yang terlihat pada tangkapan layar | 0,60621 
| Private leaderboard | **Belum diketahui** | Baru tersedia setelah penutupan kompetisi |

Saya memilih tetap memakai **submission ke-10** sebagai acuan. Ada kandidat lanjutan pada `submissions/011_Constantine_candidate.csv`, tetapi kenaikan pada pengecekan lokal sangat kecil dan belum cukup kuat untuk menyimpulkan bahwa hasil Kaggle akan lebih baik. Skor 0,60627 adalah *skor publik yang dilaporkan dan diperlihatkan lewat leaderboard*, bukan keluaran dari menjalankan ulang `scripts/train.py`.

## Kenapa kasus ini tidak sederhana?

Dataset kompetisi terdiri dari **533.465 baris train** dan **228.629 baris test**. Kelas kerusakannya tidak seimbang. Dalam evaluasi lokal historis, kelas 2 dan 3 sering saling tertukar; justru dua kelas itulah yang paling menahan Macro F1. Karena itu saya tidak hanya mengejar accuracy, melainkan membandingkan F1 per kelas, confusion matrix, perubahan keputusan kelas, dan konsistensi antarfold.

Pendekatan yang saya coba berkembang dari model boosting dengan fitur karakteristik bangunan, lalu variasi penanganan kelas dan lokasi, sampai blending dan koreksi keputusan berdasarkan kelompok. Tidak semua percobaan menghasilkan peningkatan. Saya tetap mencatat yang gagal agar alasan mempertahankan submission terbaik bisa ditelusuri.

Bacaan utama:
- [Metode: data, fitur, model, validasi, dan pemeriksaan leakage](reports/METODOLOGI.md)
- [Catatan eksperimen: termasuk percobaan yang tidak naik](reports/CATATAN_EKSPERIMEN.md)
- [Hasil akhir dan batas kesimpulan](reports/HASIL_AKHIR.md)
- [Audit kebocoran data](reports/LEAKAGE_AUDIT.md)
- [Rincian eksperimen historis](reports/EXPERIMENT_RESULTS.md) dan [submission 10–11](reports/SUBMISSION_10_AND_NEXT.md)

## Gambaran proses

1. Memastikan data train/test dan format submission cocok, terutama kolom `building_id` dan `damage_grade`.
2. EDA: distribusi target, missing value, kategori material, serta karakteristik bangunan sebelum gempa.
3. Menyusun fitur rasio dan interaksi yang masuk akal dari segi kerentanan bangunan. Prefix ID dan kode wilayah diuji sebagai proxy, bukan dianggap koordinat atau bukti geologi.
4. Mencoba boosting (LightGBM, CatBoost), class weighting, target encoding berlipat, variasi kompleksitas, dan blending probabilitas. Sebagian pengujian menggunakan model atau cache eksperimen historis.
5. Menyeleksi kandidat memakai Macro F1 pada data validasi, lalu memeriksa apakah perubahan tetap konsisten pada fold historis. Koreksi kecil yang interval ketidakpastiannya masih melintasi nol **tidak otomatis menggantikan model lama**.
6. Mengecek integritas CSV submission, mengarsipkan versi terbaik, dan memisahkan skor Kaggle dari skor lokal.

**Penting tentang reproduksi:** `scripts/train.py` adalah **pipeline eksperimen modular yang dapat dijalankan ulang**, bukan rekonstruksi persis ensemble historis yang menghasilkan public score 0,60627. CSV submission 10–11 serta laporan skor tersimpan, tetapi beberapa checkpoint/model sementara dan prediksi validasi per baris dari eksperimen historis tidak lagi tersedia. Saya tidak mengklaim bahwa satu perintah training di repo ini langsung menghasilkan skor 0,60627.

## Struktur repo

```text
Structural-Damage-Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                       # salinan ZIP kompetisi masih ada; CSV lain diabaikan Git
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_submission.ipynb
│   ├── 04_experiment_review.ipynb
│   ├── 05_submission10_and_refinement.ipynb
│   └── 06_hasil_akhir.ipynb
├── scripts/
│   ├── feature_engineering.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── inspect_results.py
├── tests/
├── models/                     # checkpoint lokal, tidak dilacak Git
├── submissions/
│   ├── 010_Constantine(10).csv
│   └── 011_Constantine_candidate.csv
└── reports/
    ├── METODOLOGI.md
    ├── CATATAN_EKSPERIMEN.md
    ├── HASIL_AKHIR.md
    ├── EXPERIMENT_RESULTS.md
    ├── SUBMISSION_10_AND_NEXT.md
    ├── LEAKAGE_AUDIT.md
    ├── experiment_history.json
    └── submission10_followup_results.json
```

## Cara menjalankan

Butuh Python 3.10+; semua langkah dari root repository.

```bash
git clone https://github.com/Constantine1507/Structural-Damage-Classification.git
cd Structural-Damage-Classification
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Dataset ZIP kompetisi `data/structural-damage-classification.zip` masih tersimpan di repository ini (sekitar 17 MB); ini perlu ditinjau lagi jika aturan redistribusi kompetisi melarangnya. Alternatif yang disarankan: unduh dataset dari Kaggle setelah menerima rules, lalu letakkan `train.csv`, `test.csv`, dan `sample_submission.csv` dalam `data/`. Dataset CSV diabaikan Git. Jika menggunakan ZIP yang tersimpan, berikan **path ZIP-nya langsung** pada argumen `--data` (bukan hanya folder `data/`).

```bash
# Ringkasan hasil historis dan pemeriksaan dua CSV tanpa melatih ulang
python -m scripts.inspect_results

# Pemeriksaan alur cepat: tidak mewakili model akhir
python -m scripts.train --data data/structural-damage-classification.zip --output outputs/smoke --quick

# Eksperimen modular penuh; memerlukan CPU/RAM dan waktu lebih banyak
python -m scripts.train --data data/structural-damage-classification.zip --output outputs/full

# Prediksi ulang dari bundle yang benar-benar dihasilkan oleh perintah sebelumnya
python -m scripts.predict --data data/structural-damage-classification.zip --model outputs/full/model_bundle.joblib --output outputs/full/submission.csv
```

Notebook 01–03 mendemonstrasikan alur EDA–modeling–submission untuk **model modular baru** dan memerlukan dataset. Notebook 04–06 meninjau **hasil eksperimen yang telah dicatat**; notebook tersebut tidak diam-diam melatih ulang model lama.

Jika ingin memeriksa format dan perbedaan dua prediksi yang tersedia, jalankan `python -m scripts.inspect_results --data data`. Tool akan memeriksa kecocokan ID terhadap sampel resmi jika dataset tersedia, tetapi **tidak bisa menghitung Macro F1 test** tanpa label tersembunyi.

## Interpretasi singkat

Pada evaluasi historis 60.000 baris, F1 kelas 2 sekitar **0,473** dan kelas 3 sekitar **0,491**, sementara kelas 5 sekitar **0,761**. Perbedaan tersebut menjelaskan mengapa model dengan prediksi kelas mayoritas yang baik belum tentu unggul pada Macro F1. Kelas bertetangga sering tertukar sehingga pendekatan ordinal layak diuji, tetapi **belum bisa dinyatakan lebih bagus** tanpa eksperimen lanjutan.

Ada juga risiko menghafal lokasi. Pipeline modular baru mengeluarkan ID lengkap. Namun audit notebook asli menemukan `id_numeric` dalam fitur ensemble historis; penggunaan ID dan konteks transduktif dibahas terpisah pada audit lanjutan. Koreksi yang bergantung pada target, termasuk target encoding atau offset kelompok, harus dipelajari hanya dari bagian training/calibration yang sesuai. Detailnya ada di laporan audit.

## Hal yang sengaja saya pisahkan

| Yang diketahui | Yang belum terbukti |
|---|---|
| Skor publik terbaik 0,60627 dan tangkapan layar peringkat 1/25 | Skor private serta peringkat final |
| Perubahan 685 baris antara dua file CSV tersimpan | Bahwa kandidat 011 lebih bagus di Kaggle |
| Riwayat eksperimen dan ringkasan validasi terdokumentasi | Reproduksi byte-identical hasil terbaik dari raw-data-only tanpa cache/model historis |
| Pipeline sklearn dengan `ColumnTransformer` tersedia dan bisa dilatih ulang | Bahwa skor pipeline modular sama dengan public 0,60627 |

Tugas ini menilai **public leaderboard (15%)**, **private leaderboard (35%)**, serta **struktur/workflow repo, Pipeline + ColumnTransformer, docstring, petunjuk penggunaan, dan interpretasi (50%)**. Penilaian akhir tetap mengikuti dosen dan skor private, bukan asumsi dari ranking sementara.

## Audit mendalam setelah kandidat mendapat 0,60621

[Laporan terbaru](reports/deep_audit/ANALYSIS_060621.md) memuat pemeriksaan ulang data, enam uji perubahan label held-out yang lulus, analisis 685 perpindahan kelas, temuan konteks train+test serta ID numerik pada notebook asli, dan rencana validasi lanjutan. Kandidat 011 mendapat public **0,60621**, di bawah acuan **0,60627**. Tidak ditemukan bukti label test langsung pada jalur fitur yang diuji; ini bukan sertifikasi seluruh ensemble bebas leakage. Laporan dan script tersedia di `reports/deep_audit/`.

## Pembaruan reproduksibilitas & confusion matrix (14 Oktober 2026)

- Notebook **mandiri** [07_standalone_reproduction.ipynb](notebooks/07_standalone_reproduction.ipynb) berisi seluruh definisi model/preprocessing (tanpa impor `scripts/`), visualisasi confusion, full refit, save/replay model dengan `joblib`, dan verifikasi CSV byte-identical. Input ZIP kompetisi; `QUICK=False` untuk full.
- [Validasi historis yang diaudit](reports/HISTORICAL_VALIDATION_20261014.md) memuat matrix 60.000 baris, Grade 1–5, F1 per kelas dan keterbatasan metode. [Visualisasi SVG](reports/figures/historical_v7_confusion.svg) dan [angka JSON](reports/historical_counts_20261014.json).
- Script verifikasi: `python -m scripts.historical_checkpoint_audit --data data/structural-damage-classification.zip --checkpoint /path/to/Structural_Damage_NonNeural_Rebuild_Checkpoint.zip`. Checkpoint besar dari sesi riset lama harus tersedia secara terpisah.
- **Jangan samakan** replay kandidat *baru* dengan submission 010 yang terbaik di Kaggle (public 0,60627). Seluruh state ensemble historis belum tersedia untuk rekonstruksi byte-identical 010. Tidak ada label test tersembunyi untuk confusion matrix Kaggle, dan validasi historis sudah terkena adaptivity.

## Submission ke-10: verifikasi file dan validasi historis

Submission final adalah [`010_Constantine(10).csv`](submissions/010_Constantine(10).csv), dengan hash SHA-256 **`2764f5f59a3924b982a72584edb4facdd8aeabde16519a8ce0e04e45975d0767`**. Integritas seluruh 228.629 prediksi dan urutan ID telah diverifikasi. Ringkasan pengujian, confusion matrix historis dan penelusuran 439 selisih prediksi dari replay native v7 terdapat pada [laporan verifikasi akhir](reports/SUBMISSION_10_VERIFICATION.md).

Notebook mandiri: [`notebooks/08_submission10_verified.ipynb`](notebooks/08_submission10_verified.ipynb). Kode audit: [`scripts/audit_submission10.py`](scripts/audit_submission10.py). Jalankan `python -m scripts.audit_submission10 --data data/structural-damage-classification.zip --submission "submissions/010_Constantine(10).csv"`. Untuk audit historis, tambahkan `--checkpoint /path/to/Structural_Damage_NonNeural_Rebuild_Checkpoint.zip`.

Pemisahan hasil: CSV final dapat diverifikasi byte-identical, evaluasi historis v6/v7 dapat direkalkulasi dari checkpoint, dan notebook baseline sudah lulus `Run All` mode smoke. Reproduksi **training historis final 010** yang byte-identical dari data mentah masih belum terbukti; hasil mode full belum lulus eksekusi. Skor Kaggle private dan confusion matrix pada hidden test tidak tersedia. Tidak memakai neural network.


## Verified end-to-end execution (14 October 2026)

[Notebook 09 — executed evidence](notebooks/09_end_to_end_verified.ipynb) includes **saved cell outputs** from a real run on the original competition dataset and historical checkpoint: SHA/ID checks passed, historical v6/v7 Macro F1 0.600795670/0.601198136, and a fresh non-neural pipeline trained on 9,600 examples, tested on 2,400, with a successful joblib prediction replay (smoke Macro F1 0.404740). The latter is **not** the public 0.60627 ensemble. CI re-executes it against the original dataset on GitHub Actions and uploads an executed notebook artifact (external 454 MB archive is optional and skipped in CI).

**Provenance boundary:** native v7 replay and final submission 010 still differ on **439 rows**. Byte-identical full retraining of the original best ensemble is **not established**. Do not confuse integrity of a preserved CSV with reproducibility of its complete modeling recipe.

Read [the checkpoint-to-submission audit](reports/PROVENANCE_010.md) and [machine-readable execution results](reports/verified_evidence_20261014.json) for exact hashes, test conditions, limits and replay instructions.
