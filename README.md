# Structural Damage Classification — Gempa Nepal 2015

**Tugas 1 Supervised Machine Learning · Sains Data ITS · 010_Constantine**

Proyek ini memprediksi tingkat kerusakan bangunan akibat gempa Nepal 2015 ke dalam lima kelas, dari Grade 1 sampai Grade 5. Saya menggunakan model untuk data tabular, terutama LightGBM dan CatBoost. Tidak ada neural network yang digunakan.

[Kompetisi Kaggle](https://www.kaggle.com/t/a9196b92680b4492aeb06aeaee94f654) · [Ketentuan tugas](https://docs.google.com/document/d/1e8SbPqX3BX6_hsmawrtFu9kn80WotSfbORxYNg6cNBE/edit) · [Hasil akhir](reports/HASIL_AKHIR.md) · [Audit dua submission](reports/SUBMISSION_FINAL_REVIEW.md)

## Hasil utama: submission 010

**Submission yang saya pertahankan sebagai hasil terbaik adalah [`010_Constantine(10).csv`](submissions/010_Constantine(10).csv).** Public Macro F1 yang tercatat untuk submission ke-10 adalah **0,60627**. Pada tangkapan layar leaderboard tanggal 13 Oktober 2026, posisinya sementara **1 dari 25 peserta**. Angka tersebut berasal dari catatan/tangkapan layar Kaggle, bukan dari menghitung label test yang tersembunyi. Bukti unggahan yang menghubungkan skor publik dengan hash CSV belum tersimpan di repository.

| Berkas | Peran | Hasil yang bisa dilaporkan | Keputusan |
|---|---|---|---|
| [`010_Constantine(10).csv`](submissions/010_Constantine(10).csv) | **Submission utama** | Public Macro F1 **0,60627** (catatan leaderboard) | Tetap digunakan sebagai best public |
| [`010_Constantine(11).csv`](submissions/010_Constantine(11).csv) | Arsip kandidat sebelumnya | Skor pengiriman 0,60621 pernah dilaporkan; hubungan dengan hash berkas 011 belum terverifikasi | Bukan pengganti 010 |
| [`010_Constantine(12).csv`](submissions/010_Constantine(12).csv) | **Eksperimen lanjutan** dengan pemisahan evaluasi lebih jelas | Macro F1 **0,602359** pada audit lokal; public Kaggle **0,60449** (hasil unggahan dilaporkan 14 Oktober 2026) | Dicatat sebagai kandidat, bukan best public |

Skor publik kandidat 012 sebesar **0,60449** lebih rendah **0,00178** daripada submission 010 (**0,60627**), sehingga 010 tetap menjadi hasil terbaik. Sumber skor terbaru adalah laporan hasil unggahan; bukti penghubung hash CSV dengan halaman unggahan Kaggle belum disimpan.

Skor audit lokal 012 tidak dibandingkan langsung dengan public leaderboard 010: data evaluasi dan proses pemilihan modelnya berbeda. Private leaderboard juga belum tersedia dalam bukti proyek. Karena itu saya tidak menyebut 012 lebih unggul di Kaggle.

## Data, metrik, dan masalah yang ditemui

Dataset berisi **533.465 baris train** dan **228.629 baris test**. Targetnya lima kelas kerusakan yang tidak seimbang. Penilaian menggunakan **Macro F1**, yaitu rata-rata F1 untuk setiap kelas, bukan accuracy. Model bisa cukup baik di kelas yang dominan tetapi tetap kehilangan banyak poin ketika Grade 2 dan Grade 3 saling tertukar.

Saya memulai dari pemeriksaan skema, distribusi kelas, karakteristik bangunan sebelum gempa, dan risiko kebocoran data. Fitur yang diuji meliputi usia, luas, jumlah lantai, tinggi, material bangunan, rasio fisik, serta penanda wilayah. Potongan `building_id` hanya digunakan sebagai proxy pola lokal; kedekatan nomor ID tidak sama dengan jarak fisik bangunan.

Beberapa keputusan yang penting selama pengerjaan:

- Menjaga `building_id` sebagai string agar angka nol di depan tidak berubah, lalu memastikan urutan prediksi mengikuti `sample_submission.csv`.
- Memisahkan fitur sebelum gempa dari kolom yang mungkin menggambarkan kondisi sesudah gempa.
- Menggunakan `Pipeline` dan `ColumnTransformer` untuk preprocessing, model, serta inferensi dari bundle yang tersimpan.
- Memeriksa F1 per kelas, confusion matrix, perbedaan antarkandidat, dan apakah perbaikan pada validasi cukup konsisten untuk dipertahankan.
- Menjaga submission 010 tetap utuh ketika eksperimen berikutnya belum memiliki bukti skor Kaggle yang lebih tinggi.

Penjelasan metode dan percobaan yang tidak berhasil ada di [METODOLOGI.md](reports/METODOLOGI.md), [CATATAN_EKSPERIMEN.md](reports/CATATAN_EKSPERIMEN.md), dan [EXPERIMENT_RESULTS.md](reports/EXPERIMENT_RESULTS.md).

## Eksperimen lanjutan: submission 012

Setelah meninjau kembali risiko target encoding dan penggunaan ID, saya menjalankan eksperimen baru dengan prosedur yang lebih terpisah. Statistik fitur kelompok dibentuk dari bagian training, sedangkan fitur target tetangga dan target encoding memakai cross-fitting. ID numerik utuh tidak dimasukkan sebagai fitur. Namun fragmen ID masih dipakai sebagai proxy, sehingga potensi ketergantungan pada pola wilayah tetap perlu diuji.

Run ini melatih **tiga LightGBM dan satu CatBoost non-neural**, kemudian membandingkan **69 kandidat keputusan**. Resep yang terpilih menggunakan **25% LightGBM (`lgb47`) dan 75% CatBoost (`cat_numeric`)**, ditambah offset kelas hasil kalibrasi.

| Bagian data | Jumlah baris | Kegunaan |
|---|---:|---|
| FIT | 426.772 | Pelatihan dan cross-fitting |
| CAL | 37.342 | Early stopping dan kalibrasi |
| SEL | 37.342 | Memilih konfigurasi, bobot, dan offset |
| AUDIT | 32.009 | Evaluasi setelah pilihan dikunci |

Pada bagian AUDIT yang sama, baseline LightGBM menghasilkan Macro F1 **0,590189**, sementara kandidat 012 mencapai **0,602359** (selisih **+0,012170**). Kenaikan terutama terlihat pada Grade 2, 3, dan 4. Interval bootstrap berpasangan yang dilaporkan untuk selisih terhadap baseline adalah sekitar **[+0,008584; +0,015711]**. Ini bukan interval untuk selisih dengan skor Kaggle 010.

Kandidat 012 menghasilkan **20.831 prediksi berbeda** dibandingkan file 010. Jumlah perubahan itu tidak membuktikan prediksinya lebih baik. Saya menyimpan 012 untuk menunjukkan eksperimen dan evaluasinya, bukan menggantikan submission utama.

Catatan selengkapnya: [analisis 012](reports/fixed_run/ANALYSIS.md) · [metrik audit](reports/fixed_run/metrics.json) · [notebook dengan output analisis](notebooks/10_leakage_fixed_experiment.ipynb) · [pemeriksaan reproduksi bundle](reports/fixed_run/bundle_verification.json).

## Interpretasi dan keterbatasan

Pada validasi historis v7, F1 Grade 2 sekitar **0,473** dan Grade 3 sekitar **0,491**. Pada audit 012 nilainya sekitar **0,479** dan **0,496**. Angka-angka itu berasal dari split berbeda, sehingga saya tidak mengartikannya sebagai peningkatan langsung dari v7 ke 012. Keduanya justru menunjukkan persoalan yang sama: tingkat kerusakan menengah sulit dibedakan.

Uji geografis tambahan untuk 012 menunjukkan penurunan Macro F1 dari **0,518197** pada random split ke **0,441427** pada split prefix wilayah yang tidak bertumpang tindih. Ukuran data dan distribusi holdout pada uji itu berbeda dari eksperimen utama. Hasilnya mengingatkan bahwa model masih bergantung pada pola lokal, bukan memberikan angka koreksi langsung untuk leaderboard.

Saya juga tidak menganggap seluruh risiko leakage sudah hilang hanya karena beberapa pemeriksaan lulus. Riwayat pengembangan model memakai data ini berkali-kali; karena itu holdout pada run baru tidak saya klaim sebagai test yang benar-benar belum pernah disentuh sepanjang proyek. Penjelasan lengkap ada di [audit kebocoran](reports/LEAKAGE_AUDIT.md) dan [analisis 012](reports/fixed_run/ANALYSIS.md).

## Reproduksibilitas: apa yang sudah dan belum terbukti

| Hal yang diperiksa | Status |
|---|---|
| Integritas dan urutan seluruh 228.629 prediksi submission 010 | Terverifikasi, termasuk checksum file |
| Training ulang persis ensemble historis yang menghasilkan 010 | **Belum terbukti**; sebagian checkpoint historis tidak tersedia |
| Pipeline modular dari dataset mentah | Source code tersedia; smoke test pada GitHub Actions berhasil |
| Pelatihan, pemilihan model, full refit, dan output analisis 012 | Log, konfigurasi, metrik, serta notebook analisis tersimpan |
| Replay submission 012 dari bundle model yang disimpan | Dilaporkan **byte-identical** untuk seluruh test pada lingkungan eksperimen |
| Bundle terlatih 012 tersedia di repository public | **Tidak**; bundle disimpan terpisah dan tidak dilacak Git |
| Training 012 dari clone bersih di CI, lalu membandingkan CSV baru | **Belum diuji**; CI tidak menjalankan full training 012 |
| Skor Kaggle untuk 012 | **Public 0,60449** (dilaporkan 14 Oktober 2026); private belum diketahui. Bukti unggahan yang mengaitkan skor dengan hash CSV belum diarsipkan |

[![Reproducibility checks](https://github.com/Constantine1507/Structural-Damage-Classification/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/Constantine1507/Structural-Damage-Classification/actions/workflows/reproducibility.yml)

[GitHub Actions run #5 (14 Oktober 2026)](https://github.com/Constantine1507/Structural-Damage-Classification/actions/runs/38513775622) berhasil: **19 pengujian lulus**, pemeriksaan CSV 010 lulus, dan notebook 09 dijalankan sebagai smoke test. Output eksekusi notebook 09 tersedia sebagai workflow artifact. Run CI ini bukan reproduksi pelatihan akhir 010 maupun full refit 012.

Untuk 010, file final memiliki SHA-256 `2764f5f59a3924b982a72584edb4facdd8aeabde16519a8ce0e04e45975d0767`. Untuk 012, SHA-256 CSV yang dicatat adalah `b10072cca29ffbfa36d07228d1000139054cfc17193f2f61aedc7a8d3c73a396`. Audit yang membedakan data mentah, checkpoint historis, dan submission ada di [PROVENANCE_010.md](reports/PROVENANCE_010.md) serta [EVIDENCE_INDEX.md](reports/EVIDENCE_INDEX.md).

## Struktur repository

```text
Structural-Damage-Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                         # arsip ZIP kompetisi
├── notebooks/
│   ├── 01_eda.ipynb              # eksplorasi
│   ├── 02_modeling.ipynb         # pipeline modular
│   ├── 03_submission.ipynb       # pemeriksaan format
│   ├── 04_experiment_review.ipynb
│   ├── 05_submission10_and_refinement.ipynb
│   ├── 06_hasil_akhir.ipynb
│   ├── 07_standalone_reproduction.ipynb
│   ├── 08_submission10_verified.ipynb
│   ├── 09_end_to_end_verified.ipynb
│   └── 10_leakage_fixed_experiment.ipynb  # analisis lanjutan 012
├── scripts/                      # preprocessing, training, audit, inferensi
├── tests/
├── models/                       # hanya placeholder; checkpoint privat
├── submissions/
│   ├── 010_Constantine(10).csv   # HASIL UTAMA
│   ├── 010_Constantine(11).csv
│   └── 010_Constantine(12).csv   # EKSPERIMEN LANJUTAN
└── reports/
    ├── HASIL_AKHIR.md
    ├── METODOLOGI.md
    ├── EVIDENCE_INDEX.md
    └── fixed_run/               # laporan, log, gambar, metrik 012
```

## Cara menjalankan

Gunakan Python 3.10+ untuk pipeline modular. Versi paket yang dipakai saat eksperimen 012 dicatat terpisah dalam [`reports/fixed_run/requirements-run.txt`](reports/fixed_run/requirements-run.txt). Jalankan perintah dari root repository.

```bash
git clone https://github.com/Constantine1507/Structural-Damage-Classification.git
cd Structural-Damage-Classification
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt nbclient nbconvert ipykernel
python -m pytest -q
```

**Memeriksa submission utama 010** tanpa melatih ulang model historis:

```bash
python -m scripts.audit_submission10 --data data/structural-damage-classification.zip --submission 'submissions/010_Constantine(10).csv'
python -m scripts.inspect_results
```

**Menjalankan pipeline modular** (ini eksperimen terpisah, bukan rekonstruksi 010):

```bash
python -m scripts.train --data data/structural-damage-classification.zip --output artifacts/modular_smoke --quick
```

**Mengulang eksperimen 012:** ekstrak `train.csv`, `test.csv`, dan `sample_submission.csv` dari ZIP kompetisi ke folder `data/` terlebih dahulu. Perintah berikut melakukan pelatihan dan full refit; membutuhkan sumber daya lebih besar daripada smoke test:

```bash
python -m scripts.run_fixed_experiment --data data --output artifacts/fixed_run
```

Jika pemrosesan full refit terputus setelah seleksi model tersimpan, tahap akhir dapat dilanjutkan tanpa memilih kandidat ulang:

```bash
python -m scripts.resume_fixed_refit --data data --output artifacts/fixed_run --threads 8
```

Untuk menguji bundle 012 yang **benar-benar sudah dihasilkan** pada mesin tersebut:

```bash
python -m scripts.verify_fixed_delivery --data data --output artifacts/fixed_run
python -m scripts.predict_fixed --bundle artifacts/fixed_run/model_bundle.joblib --test data/test.csv --sample data/sample_submission.csv --output artifacts/replayed_012.csv
```

File 010 dalam `submissions/` tidak ditimpa oleh perintah-perintah di atas. Jika mengunduh data dari Kaggle, tetap patuhi ketentuan kompetisi. Arsip ZIP kompetisi yang sekarang ada di repository perlu dipastikan boleh didistribusikan ulang.

## Keputusan akhir

Sampai ada hasil Kaggle yang terverifikasi dan lebih baik, **010 tetap submission utama**. Eksperimen 012 saya pertahankan sebagai catatan perbaikan metode, pengujian kebocoran data, serta reproduksi model baru. Pemisahan ini penting supaya peningkatan dokumentasi dan validasi tidak disalahartikan sebagai peningkatan leaderboard.
