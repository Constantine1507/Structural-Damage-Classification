# Hasil akhir sementara — 13 Oktober 2026

## Keputusan submission

**File terbaik yang dipertahankan:** `submissions/010_Constantine(10).csv`.

**Public Macro F1: 0,60627**. Pada tangkapan layar leaderboard 13 Oktober, posisi `010_Constantine` adalah **peringkat 1 dari 25 mahasiswa**. Skor submission terbaru pada waktu itu **0,60621**, sehingga best tetap **0,60627**. Angka leaderboard di sini berasal dari bukti tangkapan layar, bukan panggilan langsung ke API Kaggle. **Private Macro F1 dan peringkat akhir belum diketahui.**

| File | Jumlah prediksi | Status |
|---|---:|---|
| `010_Constantine(10).csv` | 228.629 | Acuan terbaik dari public leaderboard (berdasarkan laporan/screenshot) |
| `011_Constantine_candidate.csv` | 228.629 | Kandidat lokal; belum ada asosiasi yang terverifikasi dengan skor Kaggle tertentu |

Dua file tersebut punya skema `building_id,damage_grade` dengan integer kelas 1–5. Rekap sebelumnya mencatat hash SHA-256 untuk keduanya dan beda **685 prediksi**. Pemeriksaan otomatis untuk skema, hash, dan perbedaan bisa dijalankan lewat `python -m scripts.inspect_results`.

## Apa yang bisa saya simpulkan?

Model secara kompetitif berada pada posisi paling tinggi dari 25 peserta **pada screenshot public leaderboard**. Namun nilai Macro F1 tersebut tidak sama dengan accuracy dan tidak menyatakan model sudah cukup untuk penilaian keselamatan bangunan nyata.

Eksperimen setelah submission ke-10 belum menunjukkan peningkatan yang meyakinkan. Koreksi jenis lantai dasar memperbaiki pooled confirmation sekitar **+0,000102**, tetapi interval bootstrap 95% masih meliputi nol; dua fold menurun. Saya memilih tidak melebih-lebihkan selisih ini.

Kelemahan terbesar tetap pada kelas kerusakan menengah, terutama kelas 2 dan 3, yang sering tertukar pada confirmation historis. Pekerjaan berikutnya yang paling masuk akal adalah mengevaluasi kesalahan antarkelas bertetangga secara lebih disiplin, bukan menganggap model dengan pohon lebih banyak otomatis lebih baik.

## Batasan pelaporan

- **Skor publik**: diketahui dari screenshot/laporan, tidak dihitung menggunakan label test tersembunyi.
- **Skor lokal**: berasal dari skema FIT/CAL/SEL dan confirmation historis; sebagian data ini sudah dipakai dalam banyak iterasi sehingga sifatnya adaptif.
- **Skor private**: belum diketahui, jangan diisi dengan skor lokal.
- **Reproduksi**: file CSV final tersimpan; eksperimen modular dapat dijalankan dari raw data. Namun ensemble dan cache historis yang melahirkan CSV ke-10 belum tersedia seluruhnya sebagai checkpoint dalam repo.
- **Target 0,65**: belum tercapai, baik sebagai skor public yang terbukti maupun skor private.
- **Aspek tugas**: struktur repository, Pipeline + ColumnTransformer, docstring, README, dan interpretasi disediakan; keputusan penilaian milik dosen.

Kesimpulan saya sederhana: untuk sementara **submission ke-10 tetap hasil yang layak dipertahankan**. Eksperimen selanjutnya perlu menunjukkan peningkatan yang konsisten pada validasi yang layak sebelum mengubah file utama.
