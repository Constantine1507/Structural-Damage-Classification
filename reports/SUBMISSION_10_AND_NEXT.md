# Submission 10 dan kandidat berikutnya

> Pembaruan: skor leaderboard **0,60621** tercatat sesudah hasil terbaik **0,60627**, tetapi hubungan skor tersebut dengan hash CSV kandidat 011 belum dibuktikan dari arsip unggahan. Lihat [audit lanjutan](deep_audit/ANALYSIS_060621.md).

**Acuan terbaik yang dipertahankan adalah `010_Constantine(10).csv` (public Macro F1 tercatat 0,60627).** Angka public didokumentasikan dari tangkapan layar/laporan Kaggle, bukan diambil dari API. Belum ada bukti unggahan yang menghubungkan skor dengan hash CSV secara independen; private score belum diketahui.

## Identitas file

| File | SHA-256 | Status |
|:--|:--|:--|
| `010_Constantine(10).csv` | `2764f5f59a3924b982a72584edb4facdd8aeabde16519a8ce0e04e45975d0767` | Acuan public terbaik yang dilaporkan |
| `011_Constantine_candidate.csv` | `3b3f560d466b103f22b0dfddad50e15a84c39725d38b5fcb4bb60f429812e907` | Kandidat eksperimen, belum memiliki skor Kaggle |

Keduanya berisi 228.629 baris, ID unik dan kelas integer 1–5, header `building_id,damage_grade`. Kandidat berikutnya berbeda pada **685 baris** dari submission 10. File ke-10 berbeda pada 439 baris dari replay native model v7; penyebabnya belum terverifikasi.

## Dua putaran yang benar-benar dijalankan

Putaran pertama menguji **45 konfigurasi**: sembilan kelompok koreksi dengan shrinkage 0,05 / 0,1 / 0,2 / 0,35 / 0,5. Kelompoknya adalah material, fondasi, atap, lantai dasar, lantai lain, jumlah lantai, kelompok umur, distrik × material, dan pasangan dua skor kelas tertinggi. Offset di-fit pada CAL 42.677 baris untuk memaksimalkan Macro F1 global; kelompok berukuran kurang dari 100 tidak dikoreksi. SEL berisi 37.343 baris.

Tiga jenis kelompok teratas pada SEL dibekukan sebelum diperiksa pada tiga fold historis. Offset di-fit ulang pada CAL setiap fold, kemudian diuji pada 20.000 baris VAL per fold. Perakitan probabilitas VAL diverifikasi terhadap cache asli. Total confirmation 60.000 baris.

| Resep | SEL Macro F1 | Pooled confirmation | Delta confirmation |
|:--|--:|--:|--:|
| Submission 10 | 0.6001154955 | 0.6011875286 | 0 |
| Kelompok umur × 0,5 | 0.6003598472 | 0.6010336757 | −0.0001538529 |
| Pasangan kelas × 0,1 | 0.6002702723 | 0.6012434481 | +0.0000559194 |
| **Lantai dasar × 0,35** | **0.6002676456** | **0.6012892715** | **+0.0001017429** |

Kelompok umur menang pada SEL tetapi menurun pada confirmation. Kandidat lantai dasar memberi confirmation tertinggi dari tiga finalis, namun dua fold menurun. Paired group bootstrap 1.000 replikasi, seed 20261013, berdasarkan prefix enam digit memberi interval delta 95% **[−0,000265138; +0,000481916]**. Interval mencakup nol. Ini bukti yang lemah untuk peningkatan generalisasi.

Putaran kedua melatih **empat model tambahan** pada CAL: LogisticRegression C=0,01 dan 0,1 serta LightGBM 250 pohon, leaves 7 dan 15. Fitur berupa atribut sebelum gempa serta probabilitas dan log-probabilitas ensemble arsip; ID geografis eksplisit dihapus. Pipeline + ColumnTransformer melakukan imputasi, scaling numerik dan one-hot kategori pada CAL. Setiap model diblend dengan baseline pada bobot 0,05 / 0,1 / 0,2, total **12 konfigurasi**. Hasil terbaik SEL **0,6000755900**, masih di bawah baseline **0,6001154955**. Semua ditolak sebelum confirmation lanjutan.

## Interpretasi dan keputusan

**Submission 10 tetap acuan public terbaik.** Kandidat `011_Constantine_candidate.csv` menggunakan koreksi jenis lantai dasar × 0,35 di atas resep submission 10. Bukti unggahan Kaggle yang mengikat hash kandidat 011 ke skor public belum tersedia. Uji lokal tidak cukup untuk menyimpulkan bahwa kandidat mengungguli 0,60627.

Pemilihan akhir juga melihat confirmation sehingga bersifat adaptif. Historical metadata sudah menyatakan data ini bukan test independen yang baru. Bootstrap bersyarat pada resep terpilih dan tidak mengoreksi seluruh pencarian. Jangan menyebut angka 0,601289 sebagai skor Kaggle.

**Angka 0,601380 dalam laporan terdahulu bukan skor submission 10.** Angka itu berasal dari varian offset historis yang rekonstruksinya tidak identik. Target 0,65 belum tercapai.

## Reproduksi dan keterbatasan artefak

Replay berbasis cache probabilitas historis pernah menghasilkan CSV identik. Ini verifikasi preservasi hasil dari cache, bukan bukti pelatihan ulang dari data mentah. File 010, 011, dan ringkasan numerik eksperimen tersimpan dalam repository.

Sebagian offset terlatih, checkpoint model tambahan, dan prediksi validasi per baris dari eksperimen historis tidak tersedia dalam repository. Karena itu repository ini menyimpan **CSV final dan bukti hasil**, belum checkpoint lengkap yang mereproduksi pelatihan putaran ini. Notebook 05 memeriksa kedua CSV dan merangkum JSON hasil; ia tidak melatih ulang model. Pipeline `scripts/train.py` yang sudah ada adalah workflow terpisah dan tidak boleh diberi klaim public score 0,60627.

Arsip ZIP kompetisi (~17 MB) berada di `data/structural-damage-classification.zip`; CSV mentah yang diekstrak tidak dilacak Git. Redistribusi arsip tetap perlu mengikuti ketentuan kompetisi. File 010 dan 011 tersedia di `submissions/`. Hasil pemeriksaan otomatis dapat dilihat pada [run CI](https://github.com/Constantine1507/Structural-Damage-Classification/actions/workflows/reproducibility.yml) dan [indeks bukti](EVIDENCE_INDEX.md).
