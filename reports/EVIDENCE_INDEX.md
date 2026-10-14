# Indeks bukti eksperimen dan reproduksibilitas

Dokumen ini membedakan hasil yang tersimpan, pemeriksaan yang berhasil dijalankan, dan skor kompetisi yang hanya tercatat dari leaderboard.

| Objek | Status dan angka | Bukti tersedia | Batas kesimpulan |
|---|---|---|---|
| Submission 010 | 228.629 prediksi; SHA-256 `2764f5f59a3924b982a72584edb4facdd8aeabde16519a8ce0e04e45975d0767` | `submissions/010_Constantine(10).csv`; `scripts/audit_submission10.py` | Integritas CSV, bukan pelatihan ulang ensemble historis |
| Best public leaderboard | **0,60627**, dicatat dari laporan/tangkapan layar | README dan catatan skor historis | Bukti unggahan Kaggle yang mengaitkan skor ke hash CSV belum disimpan |
| Kandidat 011 | 685 prediksi berbeda dari 010 | `submissions/011_Constantine_candidate.csv` | Skor **0,60621** pernah dilaporkan untuk pengiriman berikutnya; kaitannya dengan hash 011 belum terbukti |
| Validasi historis v6/v7 | Macro F1 **0,600795670 / 0,601198136**; 60.000 baris | `reports/historical_counts_20261014.json`, `reports/verified_evidence_20261014.json` | Audit menggunakan arsip checkpoint eksternal sekitar 454 MB; fold historis bersifat adaptif |
| Native v7 versus submission 010 | **439** label berbeda | `reports/PROVENANCE_010.md` | Full training identik submission 010 belum terbukti |
| Baseline non-neural CI | Fit 9.600, validasi 2.400; Macro F1 **0,40473982**; replay cocok | [GitHub Actions run #3](https://github.com/Constantine1507/Structural-Damage-Classification/actions/runs/38499691437) | Smoke test pipeline baru; bukan model leaderboard |
| Private leaderboard | **Belum diketahui** | Label hidden test tidak tersedia | Tidak dapat dihitung atau diperkirakan sebagai skor resmi |

## Notebook dan output eksekusi

Notebook 01–03 dan 05–09 pada branch `main` tersimpan sebagai source code tanpa output sel. Notebook 04 menyimpan output perhitungan ulang ringkasan historis, bukan training model baru. Notebook 09 berhasil dieksekusi dalam GitHub Actions dan file **notebook yang sudah dieksekusi** tersedia sebagai artifact bernama `submission010-reproduction-evidence` pada [run #3](https://github.com/Constantine1507/Structural-Damage-Classification/actions/runs/38499691437). Artifact juga memuat JSON audit dan hasil smoke yang dihasilkan selama CI. CI tidak memuat arsip checkpoint historis; bagian tersebut berstatus `SKIPPED`.

## Pengujian ulang dari clone bersih

```bash
python -m pip install -r requirements.txt nbclient nbconvert ipykernel
python -m pytest -q
python -m scripts.audit_submission10 --data data/structural-damage-classification.zip --submission "submissions/010_Constantine(10).csv"
mkdir -p artifacts
python -m jupyter nbconvert --execute --to notebook notebooks/09_end_to_end_verified.ipynb --output 09_executed.ipynb --output-dir artifacts
```

Perintah terakhir membuat `artifacts/09_executed.ipynb` dan `artifacts/reproduction_evidence/verified_evidence.json`. Audit v6/v7 memerlukan checkpoint eksternal sesuai [PROVENANCE_010.md](PROVENANCE_010.md). Hasil smoke bukan bukti skor Kaggle dan replay cache bukan training ulang penuh.

Arsip ZIP kompetisi (~17 MB) memang disertakan dalam repository. Kelayakan redistribusi arsip tetap harus mengikuti aturan kompetisi.
