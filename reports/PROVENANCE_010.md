# Provenance dan reproduksibilitas submission 010

Tanggal audit: 14 Oktober 2026. Objek final **tetap** `submissions/010_Constantine(10).csv`, bukan kandidat 011 atau model baseline baru.

## Matriks bukti

| Klaim | Status | Bukti |
| --- | --- | --- |
| SHA-256 file final, format, kelas 1–5, 228.629 ID dan urutan sesuai sampel | **LULUS** | `notebooks/09_end_to_end_verified.ipynb` dan `reports/verified_evidence_20261014.json` |
| Metrik v6/v7 dihitung kembali dengan 60.000 label train asli | **LULUS dengan arsip eksternal** | Checkpoint `Structural_Damage_NonNeural_Rebuild_Checkpoint.zip`; v6=0,600795670, v7=0,601198136 |
| Model non-neural baru fit–save–reload–predict | **LULUS (smoke)** | 9.600 baris fit, 2.400 validasi; Macro F1=0,40473982 dan prediksi replay identik |
| Native replay v7 menghasilkan persis submission 010 | **TIDAK** | 439/228.629 label berbeda (0,1920%) |
| Training ulang ensemble lengkap dari raw data menghasilkan CSV 010 byte-identical | **BELUM DIBUKTIKAN** | Beberapa state final/koreksi historis tidak tersedia lengkap |
| Confusion matrix pada Kaggle hidden test | **TIDAK TERSEDIA** | Hidden test tidak memiliki ground truth publik |
| Private Kaggle score | **BELUM DIKETAHUI** | Skor private belum diumumkan/diverifikasi |

## Rantai asal-usul

```text
train.csv + test.csv + sample_submission.csv (ZIP kompetisi)
  ├── research_v6/confirmation/pooled.npz [label 0..4, 60.000 baris]
  ├── research_v7/confirmation/pooled.npz [label 0..4, 60.000 baris]
  └── research_v7/full_refit/native_replay_submission.csv
          | 439 prediksi berbeda, sebab tepatnya belum terverifikasi
          v
      submissions/010_Constantine(10).csv [final yang dibekukan]
```

**Penting:** Panah ke CSV 010 adalah hubungan audit/perbandingan, **bukan** klaim bahwa native v7 saja membuat file akhir. Notebook mengonversi label arsip 0–4 menjadi 1–5 sebelum dibandingkan dengan train asli. Public **0,60627** adalah skor Kaggle yang tercatat dari laporan/tangkapan layar leaderboard, bukan hasil perhitungan hidden label atau uji smoke. Tautan unggahan yang menghubungkan skor ke hash file belum diarsipkan.

## Identitas input

- `train.csv` SHA-256: `231c9bf0430faa9c1d2281b4edbcd9ce0d7a863c6ff67042a8fbc992469d37bb`
- `test.csv` SHA-256: `d732c3070b878ebc1346907d50e6a6a9db0eadf0790631b37c03d3ba6c5c`
- `010_Constantine(10).csv` SHA-256: `2764f5f59a3924b982a72584edb4facdd8aeabde16519a8ce0e04e45975d0767`

## Cara mengulang

Dari root repository, siapkan Python >=3.10, `python -m pip install -r requirements.txt nbclient nbconvert ipykernel` lalu jalankan:

```bash
python -m pytest -q
jupyter nbconvert --to notebook --execute notebooks/09_end_to_end_verified.ipynb \
  --ExecutePreprocessor.timeout=600 --output 09_reexecuted.ipynb \
  --output-dir artifacts/
```

Dataset kompetisi dipakai dari `data/structural-damage-classification.zip`, dan submission asli dari `submissions/010_Constantine(10).csv`. Arsip 454 MB dapat diletakkan di `data/Structural_Damage_NonNeural_Rebuild_Checkpoint.zip`, atau menggunakan environment variable `DAMAGE_CHECKPOINT_ZIP`. Jika arsip tidak tersedia, hanya audit v6/v7 yang **SKIPPED**; verifikasi submission dan smoke run tetap dilakukan.

[GitHub Actions run #3](https://github.com/Constantine1507/Structural-Damage-Classification/actions/runs/38499691437) berhasil menjalankan unit test, audit CSV, dan notebook 09 tanpa checkpoint historis eksternal. **Hasil eksekusi** tersimpan dalam workflow artifact; notebook sumber di branch `main` tidak menyimpan output sel. Run CI membuktikan smoke Macro F1 **0,40473982** dan replay model baru, **bukan** reproduksi skor public **0,60627**. Angka historis v6/v7 di atas memerlukan checkpoint eksternal dan berasal dari audit berbeda.

Bukti smoke bukan reproduksi model kompetisi terbaik. Interpretasi confusion matrix ada di `reports/HISTORICAL_VALIDATION_20261014.md`.


## Catatan perbaikan verifikasi otomatis, 14 Oktober 2026

Pemeriksaan tambahan mengunci SHA-256 dataset dan submission di pengujian notebook serta menghilangkan penggunaan direktori CSV yang belum diekstrak pada notebook 01–03. Semua output eksperimen modular baru disimpan di bawah `artifacts/`, sehingga file terbaik `010_Constantine(10).csv` tetap utuh. Notebook verifikasi menghasilkan laporan JSON dari eksekusi saat ini dan GitHub Actions menyimpan notebook yang dieksekusi sebagai artifact. Riwayat lokal dan hasil workflow tidak dicampur: audit historis memerlukan checkpoint eksternal, sedangkan pemeriksaan integritas submission dan pelatihan baseline smoke dijalankan pada clone bersih.
