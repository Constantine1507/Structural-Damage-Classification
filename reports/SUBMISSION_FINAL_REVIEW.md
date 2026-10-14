# Pemeriksaan akhir submission 010 dan 012

14 Oktober 2026. Catatan ini menyatukan keputusan submission dengan batas bukti yang benar-benar tersedia. Notebook pemeriksaan: [03_submission.ipynb](../notebooks/03_submission.ipynb) dan [06_hasil_akhir.ipynb](../notebooks/06_hasil_akhir.ipynb).

## Hasil yang dipertahankan

| Submission | Public Macro F1 yang dilaporkan | Peran |
| --- | ---: | --- |
| `010_Constantine(10).csv` | **0,60627** | Hasil utama |
| `010_Constantine(12).csv` | **0,60449** | Eksperimen lanjutan, opsi kedua jika kompetisi mengizinkan |

Perbedaan public 012 terhadap 010 adalah **−0,00178**. Skor berasal dari laporan unggahan Kaggle 13–14 Oktober 2026; screenshot unggahan yang mengaitkan skor dengan SHA-256 CSV belum disimpan. Private score keduanya belum diketahui.

Checksum yang seharusnya dimiliki berkas final:

- 010: `2764f5f59a3924b982a72584edb4facdd8aeabde16519a8ce0e04e45975d0767`
- 012: `b10072cca29ffbfa36d07228d1000139054cfc17193f2f61aedc7a8d3c73a396`

Keduanya harus memiliki 228.629 baris dan mempertahankan urutan `building_id` dari `sample_submission.csv`. Notebook 03 akan menggagalkan eksekusi bila file berbeda dari checksum yang diharapkan, urutan ID salah, atau kelas di luar 1–5. Notebook ini **memeriksa artefak yang ada**, bukan membangun model terbaik 010 dari awal.

## Hasil validasi dan keterbatasan

| Evaluasi | Macro F1 | Cara membacanya |
| --- | ---: | --- |
| Confirmation historis 010 | 0,601188 | Pernah dipakai dalam pemilihan adaptif; bukan holdout independen murni |
| AUDIT 012 | 0,602359 | Pemilihan model dibekukan dahulu pada run baru, tetapi dataset telah digunakan berkali-kali selama sejarah proyek |
| Baseline LightGBM dalam AUDIT 012 | 0,590189 | Perbandingan langsung yang sah dengan AUDIT 012 di baris yang sama |
| Public Kaggle 010 | 0,60627 | Skor dilaporkan peserta |
| Public Kaggle 012 | 0,60449 | Skor dilaporkan peserta |

Peningkatan AUDIT 012 terhadap baseline lokal adalah **+0,012170**. Peningkatan ini tidak membuktikan bahwa 012 mengalahkan 010 karena keduanya belum dibandingkan pada label dan fold yang sama.

Sebanyak **20.831 prediksi** berbeda antara 010 dan 012 menurut catatan audit model 012. Notebook 03 menghitung ulang perubahan ini dan mencetak tabel transisi prediksi. Tabel transisi bukan confusion matrix terhadap label asli. Notebook 06 menampilkan F1 per kelas dari 32.009 baris AUDIT 012, bukan dari hidden test Kaggle.

Prosedur 012 menggunakan FIT/CAL/SEL/AUDIT, cross-fitting target encoding, dan fitur tetangga. Masih ada ketergantungan pada pola wilayah. Stress test 60.000 baris menghasilkan F1 0,518197 pada random split dan 0,441427 pada group-prefix split, tetapi distribusi kedua holdout berbeda dan skor itu **bukan proyeksi** penurunan private leaderboard. Untuk 010, checkpoint lengkap ensemble akhir tidak tersedia sehingga training ulang hingga menghasilkan CSV identik belum terbukti. Pengujian GitHub Actions memeriksa integritas 010 dan smoke baseline, bukan training penuh 010.

## Penggunaan dan keputusan pengumpulan

Jalankan kedua notebook dari root repository atau folder `notebooks/`. Data asli tersedia pada `data/structural-damage-classification.zip`. Notebook 03 dan 06 dapat dijalankan tanpa mengunduh checkpoint privat, tidak menimpa CSV final, dan menyimpan laporan di `artifacts/submission_review/`. GitHub Actions juga menjalankan keduanya dan menyimpan notebook yang dieksekusi sebagai workflow artifact.

Pilihan utama untuk pengumpulan adalah **010**. Pertahankan **012** sebagai catatan eksperimen dan pilihan final kedua **hanya bila** aturan kompetisi memungkinkan. Tidak ada angka private yang dapat disimpulkan dari skor public. Semua klaim yang belum memiliki bukti unggahan atau reproduksi end-to-end dinyatakan sebagai batasan, bukan seolah sudah diverifikasi.
