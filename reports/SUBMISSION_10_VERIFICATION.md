# Verifikasi akhir submission 010 dan eksperimen historis

**Objek final**: `submissions/010_Constantine(10).csv`. File ini tidak diganti.

## Integritas submission (dijalankan ulang 14 Oktober 2026)

- SHA-256: `2764f5f59a3924b982a72584edb4facdd8aeabde16519a8ce0e04e45975d0767`.
- Git blob SHA `1447e0678a8758f3a7a8dc8d9e8050cebbf0b4d2`, **sama** dengan file asli yang dibandingkan saat audit.
- Jumlah prediksi 228.629; urutan dan seluruh `building_id` sama dengan `sample_submission.csv`. Tidak ada ID duplikat atau kelas di luar 1–5.
- Prediksi menurut kelas: 1=23.729, 2=28.850, 3=42.163, 4=61.866, 5=72.021.
- Salinan file dapat direplay byte-identical. Hal ini **membuktikan preservasi file hasil**, bukan bahwa bobot/model original berhasil dilatih ulang.
- Public Kaggle **0,60627 (dilaporkan sebelumnya)**; skor private tidak diketahui.

## Validasi historis

Dari arsip eksperimen `Structural_Damage_NonNeural_Rebuild_Checkpoint.zip` (~454MB), label 60.000 baris cocok dengan data train asli. Tiga fold 20.000 baris cocok dengan pooled arrays; metrik dihitung menggunakan label sebenarnya.

| Model historis | Banyak validasi | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| v6 | 60.000 | 0,624517 | 0,600796 |
| v7 | 60.000 | 0,625900 | 0,601198 |

Pada 60.000 baris sama, v7 mengganti keputusan 985 baris relatif terhadap v6. Perkiraan bootstrap berpasangan 2.000 resample untuk selisih F1 v7-v6 menunjukkan interval 95% sekitar **[-0,000609; 0,001454]** (mencakup nol). Karena desain eksperimen historis bersifat adaptif, interval tersebut bersifat kondisional, bukan penilaian final yang murni.

| Actual / Pred | G1 | G2 | G3 | G4 | G5 |
|---|---:|---:|---:|---:|---:|
| G1 | 4.289 | 1.267 | 350 | 182 | 117 |
| G2 | 1.021 | 3.411 | 1.688 | 526 | 224 |
| G3 | 485 | 1.780 | 5.373 | 2.367 | 735 |
| G4 | 192 | 573 | 2.332 | 9.058 | 2.319 |
| G5 | 207 | 512 | 1.386 | 4.183 | 15.423 |

**Perbedaan penting:** arsip prediksi native v7 `research_v7/full_refit/native_replay_submission.csv` berbeda dari submission 010 pada **439/228.629 baris (0,1920%)**. Koreksi akhir atau state model persis yang menjelaskan seluruh perubahan belum tervalidasi. Karena itu, klaim bahwa training v7 mentah secara langsung dapat menghasilkan file submission 010 tidak dibuat.

## Uji eksekusi

- Notebook audit dan baseline mandiri lokal (`Run All`) telah berhasil dieksekusi tanpa error pada **mode smoke** menggunakan data kompetisi asli.
- Uji baseline lokal historis yang tercatat di laporan ini memakai LightGBM `Pipeline`/`ColumnTransformer` pada 9.600 baris fit dan 2.400 validasi: **Macro-F1 0,450334**, dengan replay `joblib` identik. **Run CI notebook 09 yang terpisah** memperoleh **0,404740** memakai konfigurasi baseline berbeda. Keduanya adalah smoke test, bukan skor final submission 010.
- Percobaan mode full terhenti saat proses kernel pada lingkungan eksekusi saat ini. Hasil full baseline tidak diklaim berhasil; status tersebut terpisah dari hasil validasi historis.
- Dataset sudah dibatasi hash train `231c9bf0430faa9c1d2281b4edbcd9ce0d7a863c6ff67042a8fbc992469d37bb` dan test `d732c3070b878ebc1346907d50e6a6c1a6a9db0eadf0790631b37c03d3ba6c5c`.

## Jalankan ulang

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q

# Pemeriksaan SHA, ID, dan label submission terbaik
python -m scripts.audit_submission10 \
  --data data/structural-damage-classification.zip \
  --submission "submissions/010_Constantine(10).csv"

# Audit v6/v7 bila arsip eksternal tersedia
python -m scripts.audit_submission10 \
  --data data/structural-damage-classification.zip \
  --submission "submissions/010_Constantine(10).csv" \
  --checkpoint data/Structural_Damage_NonNeural_Rebuild_Checkpoint.zip
```

Notebook sumber [`notebooks/08_submission10_verified.ipynb`](../notebooks/08_submission10_verified.ipynb) tidak menyimpan output eksekusi pada branch `main`. File checkpoint historis tidak tersedia di repository karena ukurannya; jalankan notebook dengan checkpoint asli untuk mengulang audit historis. Tanpa file ini, semua pemeriksaan submission tetap berjalan dan evaluasi historis dengan label asli dilewati, tidak direkayasa.

**Catatan interpretasi:** confusion matrix di atas mengukur validasi historis pada data train, **bukan** label test Kaggle. Tidak ada neural network yang digunakan.
