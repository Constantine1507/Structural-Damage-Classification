# Hasil akhir dan keputusan submission — 14 Oktober 2026

## Submission yang dipertahankan

Saya tetap menggunakan **[`010_Constantine(10).csv`](../submissions/010_Constantine(10).csv)** sebagai submission terbaik yang tercatat pada public leaderboard. Macro F1 publiknya **0,60627**; tangkapan layar 13 Oktober 2026 menunjukkan posisi sementara **1 dari 25 peserta**. Ini adalah hasil yang dilaporkan dari Kaggle, bukan hasil menghitung label test tersembunyi. Bukti unggahan yang memasangkan skor dengan SHA-256 file belum disimpan.

SHA-256 CSV 010: `2764f5f59a3924b982a72584edb4facdd8aeabde16519a8ce0e04e45975d0767`. Seluruh **228.629 baris** sudah melalui pemeriksaan format, kelas, dan urutan ID; pemeriksaan ini berbeda dari melatih ulang model yang membuat CSV.

## Mengapa 012 disimpan sebagai eksperimen lanjutan?

File [`010_Constantine(12).csv`](../submissions/010_Constantine(12).csv) dibuat melalui eksperimen baru setelah audit fitur dan risiko leakage. Saya memisahkan **426.772 FIT**, **37.342 CAL**, **37.342 SEL**, serta **32.009 AUDIT**. Model terpilih memakai bobot **0,25 LightGBM + 0,75 CatBoost**, dengan offset kelas dari kalibrasi.

| Ukuran | Hasil | Catatan |
|---|---:|---|
| Macro F1 baseline LightGBM pada audit 012 | 0,590189 | Dibandingkan pada baris audit yang sama |
| Macro F1 kandidat 012 pada audit tersebut | **0,602359** | Evaluasi sesudah pemilihan kandidat dikunci |
| Selisih terhadap baseline lokal | **+0,012170** | Bukan peningkatan terhadap skor Kaggle 010 |
| Prediksi 012 yang berbeda dari 010 | **20.831** | Perubahan label, belum tentu perbaikan |
| Public Kaggle untuk 012 | **0,60449** | Skor unggahan dilaporkan pada 14 Oktober 2026; belum ada bukti unggahan yang terhubung dengan hash CSV |
| Private Kaggle untuk 012 | **Belum diketahui** | Hasil private belum tersedia |

Interval bootstrap berpasangan untuk selisih kandidat terhadap baseline pada audit 012 adalah sekitar **[+0,008584; +0,015711]**. Interval ini tidak menghitung dampak seluruh iterasi historis yang pernah dilakukan. Skor validasi 012 juga tidak dapat dibandingkan langsung dengan public score 010, karena evaluasinya memakai data dan prosedur berbeda.

Detail tabel per kelas, confusion matrix, pengujian wilayah, dan catatan keterbatasan ada di [`fixed_run/ANALYSIS.md`](fixed_run/ANALYSIS.md). Saya mempertahankan hasil yang gagal atau belum meyakinkan dalam laporan eksperimen agar alasan memilih 010 bisa ditelusuri.

## Bukti yang tersedia

- **010:** file final, checksum, pemeriksaan ID, audit prediksi historis, serta GitHub Actions yang berhasil menjalankan verifikasi dasar.
- **012:** log training, konfigurasi yang dipilih dan dibekukan, metrik audit, notebook analisis dengan output, dan catatan replay dari bundle model tersimpan. CSV replay dilaporkan identik byte demi byte dengan 012.
- **Yang belum dapat dibuktikan:** training ulang historis yang mereproduksi 010 persis dari data mentah; full training 012 dari clone bersih pada CI; serta skor private Kaggle.

Pada [GitHub Actions run #5](https://github.com/Constantine1507/Structural-Damage-Classification/actions/runs/38513775622), **19 pengujian lulus**. Run tersebut terutama memeriksa 010 dan menjalankan smoke notebook 09, bukan melatih penuh 012. Checkpoint terlatih 012 tidak disertakan dalam repository publik, sehingga replay dari bundle memerlukan artefak privat tersebut.

## Catatan penutup

Pada versi ini saya membedakan **submission terbaik berdasarkan leaderboard (010)** dari **eksperimen lanjutan yang lebih lengkap catatan validasinya (012)**. Public Macro F1 012 yang dilaporkan pada 14 Oktober 2026 adalah **0,60449**, lebih rendah **0,00178** dari 010 (**0,60627**). Karena itu saya tetap menggunakan 010 sebagai hasil terbaik dan menyimpan 012 sebagai catatan eksperimen; angka public tersebut belum diverifikasi secara independen melalui arsip unggahan yang terhubung dengan hash CSV.
