# Indeks bukti dan batas reproduksibilitas

Dokumen ini mencatat apa yang benar-benar tersedia di repository. Skor Kaggle, validasi lokal, dan pemeriksaan integritas file bukan jenis bukti yang sama.

## Submission utama 010

| Klaim | Bukti | Batasnya |
|---|---|---|
| File final berisi 228.629 prediksi dan urutan ID tepat | [`010_Constantine(10).csv`](../submissions/010_Constantine(10).csv), `scripts/audit_submission10.py` | Memastikan integritas CSV, bukan melatih ulang ensemble historis |
| SHA-256 final 010: `2764f5f59a3924b982a72584edb4facdd8aeabde16519a8ce0e04e45975d0767` | Script audit dan catatan verifikasi | Identitas file, bukan bukti public score |
| Public Macro F1 terbaik yang dilaporkan: **0,60627** | README dan catatan/tangkapan layar leaderboard 13 Oktober 2026 | Bukti unggahan yang menghubungkan skor ke hash CSV belum diarsipkan |
| Validasi historis v6/v7: **0,600795670 / 0,601198136** pada 60.000 baris | `historical_counts_20261014.json` dan `verified_evidence_20261014.json` | Penghitungan ulang memerlukan checkpoint eksternal dan fold pernah dipakai secara adaptif |
| Prediksi replay native v7 berbeda **439 baris** dari CSV 010 | [`PROVENANCE_010.md`](PROVENANCE_010.md) | Model historis final belum bisa direproduksi persis dari data mentah |

## Eksperimen lanjutan 012

| Klaim | Bukti | Batasnya |
|---|---|---|
| Kandidat 012 dibuat melalui run baru dan bukan salinan 010 | [`012_Constantine_leakage_fixed.csv`](../submissions/012_Constantine_leakage_fixed.csv), [`fixed_run/refit.log`](fixed_run/refit.log), [`fixed_run/metrics.json`](fixed_run/metrics.json) | Skor public 012 dilaporkan 0,60449; private belum diketahui |
| Split FIT/CAL/SEL/AUDIT dan konfigurasi dibekukan | [`fixed_run/protocol.json`](fixed_run/protocol.json), [`fixed_run/frozen_selection.json`](fixed_run/frozen_selection.json) | Holdout audit tidak digunakan untuk seleksi pada run itu, tetapi bukan test historis yang benar-benar baru |
| Macro F1 audit **0,602359** versus baseline **0,590189** | [`fixed_run/metrics.json`](fixed_run/metrics.json), [`fixed_run/ANALYSIS.md`](fixed_run/ANALYSIS.md) | Perbandingan sesama eksperimen 012, bukan perbandingan skor publik dengan 010 |
| Public Kaggle 012 dilaporkan **0,60449** pada 14 Oktober 2026 | [`fixed_run/metrics.json`](fixed_run/metrics.json), [`fixed_run/ANALYSIS.md`](fixed_run/ANALYSIS.md) | Berdasarkan laporan unggahan peserta, belum diverifikasi melalui Kaggle API atau arsip unggahan yang terikat hash CSV; masih di bawah 010 (0,60627) |
| Full refit dan replay bundle menghasilkan CSV identik | [`fixed_run/refit.log`](fixed_run/refit.log), [`fixed_run/bundle_verification.json`](fixed_run/bundle_verification.json) | Bundle disimpan privat; pelatihan ulang pada clone bersih belum diverifikasi independen |
| Notebook 10 menampilkan output analisis | [`10_leakage_fixed_experiment.ipynb`](../notebooks/10_leakage_fixed_experiment.ipynb) | Training di notebook dinonaktifkan secara default; log training berasal dari proses CLI terpisah |
| Diagnostik leakage, ketergantungan wilayah, dan feature importance | [`fixed_run/ANALYSIS.md`](fixed_run/ANALYSIS.md), [`fixed_run/geographic_stress.json`](fixed_run/geographic_stress.json) | Diagnostik bukan jaminan bebas leakage atau prediksi untuk skor private |

SHA-256 012: `b10072cca29ffbfa36d07228d1000139054cfc17193f2f61aedc7a8d3c73a396`. Berkas ini berbeda pada **20.831 prediksi** dari 010. Selisih jumlah label tidak menunjukkan kandidat lebih baik.

## GitHub Actions dan output notebook

[Run #5, 14 Oktober 2026](https://github.com/Constantine1507/Structural-Damage-Classification/actions/runs/38513775622) berhasil menjalankan 19 pengujian, audit CSV 010, eksekusi notebook 09, serta pengunggahan artifact `submission010-reproduction-evidence`. Notebook 09 di branch `main` masih tanpa output sel; output eksekusinya tersimpan pada workflow artifact.

Notebook 01–03 dan 05–09 sebagian besar disimpan sebagai kode sumber tanpa output. Notebook 04 menyimpan output ringkasan historis. **Notebook 10 menyimpan output analisis 012 yang sudah dijalankan**, namun tidak melatih ulang keseluruhan model dari sel notebook secara default. GitHub Actions yang saat ini aktif **belum melakukan full refit kandidat 012**.

## Menjalankan pemeriksaan dari clone baru

```bash
python -m pip install -r requirements.txt nbclient nbconvert ipykernel
python -m pytest -q
python -m scripts.audit_submission10 --data data/structural-damage-classification.zip --submission 'submissions/010_Constantine(10).csv'
mkdir -p artifacts
python -m jupyter nbconvert --execute --to notebook notebooks/09_end_to_end_verified.ipynb --output 09_executed.ipynb --output-dir artifacts
```

Perintah di atas memverifikasi integritas submission dan smoke test, **bukan** mereproduksi training final 010. Pemeriksaan v6/v7 memerlukan checkpoint historis eksternal sekitar 454 MB.

Untuk mengulang 012, ekstrak tiga CSV kompetisi ke `data/` dan gunakan:

```bash
python -m scripts.run_fixed_experiment --data data --output artifacts/fixed_run
```

Jika full refit terputus setelah pemilihan resep tersimpan, gunakan `python -m scripts.resume_fixed_refit --data data --output artifacts/fixed_run --threads 8`. Jalur ini membutuhkan CPU, memori, dan waktu lebih besar daripada smoke test. Paket yang dipakai saat eksperimen tercatat di [`fixed_run/requirements-run.txt`](fixed_run/requirements-run.txt).

**Public Kaggle 012 dilaporkan 0,60449, sedangkan skor private, confusion matrix hidden test, dan bukti unggahan yang mengikat skor 012 ke hash CSV masih belum tersedia.** Jangan mengganti kekosongan tersebut dengan nilai dari validasi lokal.
