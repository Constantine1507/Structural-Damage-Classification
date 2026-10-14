# Evaluasi submission dan iterasi model

## Keputusan

**Submission lama belum layak diganti berdasarkan eksperimen ini.** Target Macro F1 0,65 belum tercapai. Nilai local confirmation terbaik 0,601380 masih berjarak 0,048620 dari target; ini bukan estimasi selisih public leaderboard. Bagian ini mencatat eksperimen historis sebelum 012. Public Macro F1 012 kemudian dilaporkan **0,60449** pada 14 Oktober 2026, di bawah skor terbaik 010 **0,60627**. Skor private belum diketahui, dan bukti unggahan yang mengaitkan skor 012 dengan hash CSV belum disimpan.

## Sumber dan batas bukti

Ringkasan hasil historis di bawah didokumentasikan di `experiment_history.json` dari catatan output eksperimen. Sejumlah checkpoint model dan prediksi per baris tidak tersimpan, sehingga reproduksi penuh hasil lama belum dapat diverifikasi dari repository ini. Karena itu tabel ini merupakan **catatan hasil**, bukan checkpoint yang dapat langsung dipakai untuk inference. Notebook 04 menghitung ulang ringkasan dan metrik confusion matrix dari catatan tersebut, bukan mengulang pelatihan. Jangan mengatribusikan skor ini ke `scripts/train.py`, yang menggunakan desain pipeline berbeda.

Data asli: train 533.465 baris, test 228.629 baris, lima kelas. Hash input dan incumbent tercatat di JSON. CSV incumbent memiliki 228.629 ID unik, dua kolom, kelas integer 1–5, tanpa missing. Replay notebook lama dari cache probabilitas menghasilkan CSV dengan hash identik; **itu replay cache, bukan fresh model training**. Skor public sekitar 0,605 berasal dari catatan leaderboard historis dan belum terhubung ke hash CSV tertentu melalui bukti unggahan yang tersimpan.

## Protokol delapan fit

FIT 373.425 baris, CAL 42.677, SEL 37.343 mengikuti split checkpoint lama. Preprocessing melalui Pipeline dan ColumnTransformer: imputasi median, fitur rasio fisik, interaksi material/fondasi/atap, prefix ID sebagai proxy lokasi, serta multiclass TargetEncoder dengan internal cross-fitting tiga fold (seed 20261013). Label SEL tidak digunakan untuk fit encoder atau model. CAL memilih pengali keputusan tiap kelas; SEL memilih kandidat, sehingga SEL adalah skor seleksi, bukan final test.

Bobot observasi adalah `(rata_rata_jumlah_kelas / jumlah_kelas)**power`, dihitung pada FIT. LightGBM memakai 4 thread dan seed 20261013. Kandidat dibandingkan sebagai model tunggal dan blend terhadap frozen v7 dengan bobot model baru 1, 0,1, 0,25, dan 0,5 (32 perbandingan total). Konfigurasi default: learning rate 0,05, min child 120, lambda L2 15, feature fraction 0,9, bagging 0,9/frequency 1, target smoothing 20.

| Model | Leaves / trees | Power | Standalone SEL | Blend SEL terbaik |
|:--|:--|--:|--:|--:|
| te_lgb_63 | 63 / 650 | 0 | 0.584729 | 0.599438 |
| te_lgb_63_balanced | 63 / 650 | 0.4 | 0.584294 | 0.599429 |
| te_lgb_127 | 127 / 750 | 0.25 | 0.582991 | 0.599071 |
| te_lgb_31_ova | 31 / 850 | 0.25 | 0.581765 | 0.599380 |
| te_lgb_coarse | 63 / 850 | 0.3 | 0.573951 | 0.599190 |
| te_lgb_127_smooth | 127 / 1000 | 0.4 | 0.581998 | 0.598819 |
| te_lgb_31_long | 31 / 1300 | 0.25 | 0.582419 | 0.598863 |
| te_lgb_127_ova | 127 / 750 | 0.35 | 0.582816 | 0.599390 |

Incumbent SEL: **0.600083**. Model 127 memakai min child 80/L2 20; coarse menghapus prefix halus dan smoothing 50; smooth memakai smoothing 80/L2 30; long memakai learning rate 0,035/smoothing 10; 127_ova memakai min child 160/L2 40. OVA memakai objective multiclassova. Fitur prefix halus lebih baik daripada coarse dalam percobaan ini, tetapi tetap kalah dari incumbent. Penambahan pohon dan class weighting tidak menghasilkan perbaikan yang konsisten.

## Offset keputusan per distrik

Offset lima kelas ditentukan pada CAL terhadap skor frozen v7. Optimasi memaksimalkan Macro F1 global dengan pembaruan confusion matrix per distrik, bukan rata-rata metrik distrik. Besaran langkah 0,02; 0,01; 0,005; 0,002, masing-masing dua putaran. Shrinkage dipilih dari 0; 0,1; 0,25; 0,5; 0,75; 1 pada SEL. Bobot 0,1 memberi SEL 0,600279, dibanding incumbent 0,600083. Offset besar justru menurun, konsisten dengan risiko overfit kalibrasi lokal.

| Fold historis | Incumbent | Offset distrik |
|:--|--:|--:|
| 0 | 0.603368 | 0.603377 |
| 1 | 0.602887 | 0.602860 |
| 2 | 0.597165 | 0.597733 |
| Pooled 60.000 baris | **0.601198** | **0.601380** |

Delta pooled **+0,000181**. Paired group bootstrap, 1.000 replikasi, seed 20261013, kelompok prefix enam digit, memberi interval 95% **[−0,000167; +0,000538]**. Interval mencakup nol dan satu fold menurun. Interval bersyarat pada resep terpilih; ia tidak mengoreksi adaptive search dan bukan interval untuk skor Kaggle. Historical metadata juga menyatakan `selection_is_adaptive=True` dan `pristine_independent_test=False`. Mengganti seed tidak menjadikan data historis sebagai test yang benar-benar baru.

## Analisis kelas dan batas prediktor

F1 incumbent per kelas: 1 = 0,691830; 2 = 0,473323; 3 = 0,491380; 4 = 0,588373; 5 = 0,761085. Kelas 2 dan 3 menjadi kelemahan utama. Pada confirmation, 1.688 bangunan kelas 2 diprediksi sebagai 3, dan 1.780 bangunan kelas 3 diprediksi sebagai 2. Kelas 5 juga sering bergeser ke 4 (4.183 kasus). Kesalahan bertetangga ini mendukung pengujian ordinal classification; belum membuktikan model ordinal akan unggul.

Sekitar 5,46% baris train berada dalam kelompok dengan fitur sama tetapi label berbeda. Hal tersebut menunjukkan informasi teramati tidak sepenuhnya membedakan tingkat kerusakan, bukan bukti batas skor maksimum 0,65. Kerusakan juga mungkin dipengaruhi intensitas guncangan dan detail konstruksi yang tidak tersedia. Menambah kapasitas model saja dapat memperkuat memorisasi lokasi.

## Langkah eksperimen berikutnya yang dapat dipertanggungjawabkan

1. Jalankan pipeline modular pada direktori output terpisah. Catat hash data, seed, konfigurasi, skor per kelas dan model bundle; jangan menyamakan skornya dengan eksperimen lama.
2. Gunakan CAL untuk pemilihan dan simpan AUDIT sampai resep dibekukan. Split baru tetap memiliki keterbatasan karena riwayat eksperimen dataset ini.
3. Prioritaskan ordinal threshold models dan OOF ensemble yang kesalahannya berbeda, serta geographic holdout untuk stress test. Bandingkan pada baris validasi yang sama; hentikan kandidat yang tidak meningkatkan bukti.
4. Uji satu kandidat terbaik di Kaggle sesuai kuota, dengan nama versi dan hash CSV. Tidak ada jaminan public maupun private 0,65. Jangan berulang kali memilih konfigurasi berdasarkan public leaderboard saja.

Iterasi yang sudah dijalankan tidak membenarkan penggantian incumbent. Repository dan format notebook dapat diperbaiki sekarang, tanpa mengklaim peningkatan model yang belum terbukti.
