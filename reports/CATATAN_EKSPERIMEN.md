# Catatan eksperimen

Dokumen ini adalah rangkuman percobaan yang benar-benar tercatat sepanjang pengerjaan, termasuk yang tidak memperbaiki skor. Saya sengaja tidak menuliskan hasil lokal seolah-olah skor Kaggle.

## Perkembangan skor publik

Dari catatan percakapan selama proses ini, public score yang pernah saya laporkan bergerak melalui **0,58970**, **0,60625**, **0,60503**, **0,60606**, **0,60616**, hingga **0,60627**. Urutan tersebut adalah *riwayat angka yang dilaporkan*, bukan rekonstruksi pasti nomor submission per angka. Terakhir, leaderboard memperlihatkan submission terbaru **0,60621**, sementara **best tetap 0,60627**.

Alasan skor tidak selalu naik: model baru atau koreksi keputusan bisa memperbaiki sebagian kelas tetapi merusak kelas lainnya. Macro F1 menghitung rata-rata lima kelas, bukan jumlah prediksi benar keseluruhan.

## Delapan percobaan boosting + target encoding

Data eksperimen historis ada pada `reports/experiment_history.json`. Model yang dibandingkan meliputi variasi LightGBM daun **31/63/127**, pohon **650–1.300**, `class_weight_power` **0–0,4**, smoothing encoding, objective multiclass/OVA, dan geografi lebih kasar. Ada **8 fit** dan total **32 pengujian kandidat/blend** yang tercatat.

| Percobaan | Standalone SEL Macro F1 | Blend SEL terbaik |
|---|---:|---:|
| te_lgb_63 | 0,584729 | 0,599438 |
| te_lgb_63_balanced | 0,584294 | 0,599429 |
| te_lgb_127 | 0,582991 | 0,599071 |
| te_lgb_31_ova | 0,581765 | 0,599380 |
| te_lgb_coarse | 0,573951 | 0,599190 |
| te_lgb_127_smooth | 0,581998 | 0,598819 |
| te_lgb_31_long | 0,582419 | 0,598863 |
| te_lgb_127_ova | 0,582816 | 0,599390 |

Baseline pembanding pada SEL = **0,600083**. Tidak ada model atau blend dari delapan fit ini yang mengalahkan baseline. Ini membuat saya berhati-hati dengan asumsi bahwa menambah kapasitas pohon otomatis menaikkan skor.

## Koreksi keputusan berbasis distrik

Perubahan offset kelas yang dihitung per distrik diuji dengan shrinkage; skala **0,1** menjadi pilihan pada SEL. Pada pooled confirmation historis 60.000 baris:
- baseline **0,601198**
- koreksi distrik **0,601380**
- selisih **+0,000181**
- bootstrap 95% **[-0,000167; +0,000538]**

Nilai positifnya kecil dan interval memuat nol; satu dari tiga fold justru menurun. Skor confirmation ini **bukan skor Kaggle**, dan resep koreksi tersebut **bukan rekonstruksi identik submission ke-10**.

## Eksperimen setelah submission ke-10 (0,60627)

Putaran pertama mengevaluasi **45 konfigurasi**: sembilan kelompok (material, fondasi, atap, lantai dasar, lantai lainnya, jumlah lantai, kelompok umur, distrik × material, pasangan dua kelas dengan probabilitas tertinggi) dikombinasikan dengan shrinkage **0,05 / 0,1 / 0,2 / 0,35 / 0,5**. Koreksi kelas di-fit pada CAL 42.677 baris dan diseleksi pada SEL 37.343 baris. Tiga finalis dicek pada tiga fold konfirmasi, total 60.000 baris.

| Resep | SEL | Pooled confirmation | Delta dari resep submission 10 |
|---|---:|---:|---:|
| Submission 10 | 0,600115 | 0,601188 | 0 |
| Usia bangunan × 0,5 | 0,600360 | 0,601034 | −0,000154 |
| Dua kelas teratas × 0,1 | 0,600270 | 0,601243 | +0,000056 |
| **Jenis lantai dasar × 0,35** | **0,600268** | **0,601289** | **+0,000102** |

Kandidat usia sempat lebih bagus pada seleksi, tetapi turun pada confirmation. Koreksi lantai dasar memberikan nilai confirmation tertinggi di antara finalis, tetapi dua dari tiga fold juga turun. Selisihnya tidak meyakinkan: interval bootstrap 95% **[-0,000265; +0,000482]** masih memuat nol.

Pada putaran kedua, **4 model tambahan** (LogisticRegression C=0,01/0,1 dan LightGBM kecil dengan 7/15 daun) dibandingkan melalui **12 blend**. Hasil SEL terbaik hanya **0,600076**, di bawah acuan submission 10 **0,600115**. Tidak ada alasan berbasis data untuk mengganti incumbent dengan model-model itu.

## Kandidat ke-11 dan keputusan

`011_Constantine_candidate.csv` berbeda pada **685 baris** dari `010_Constantine(10).csv`. Dari pemeriksaan lokal, kandidat itu masuk akal untuk diuji, tetapi **tidak terbukti mengalahkan 0,60627**. Skor submission terakhir yang tampak pada tangkapan layar, 0,60621, belum dapat dipasangkan secara pasti dengan hash CSV kandidat ini; karena itu saya tidak mengklaim file 011 menghasilkan skor tersebut.

Saya tetap menyimpan submission ke-10 sebagai hasil terbaik yang terkonfirmasi di public leaderboard. Private leaderboard belum tersedia.

## Kekeliruan yang paling sering muncul

Pada konfirmasi historis, F1 kelas 2 **0,4733** dan kelas 3 **0,4914** jauh di bawah kelas 5 **0,7611**. Kelas 2 kerap diprediksi 3 (**1.688 kasus**), dan kelas 3 diprediksi 2 (**1.780 kasus**). Ini menjadi alasan saya ingin menguji klasifikasi ordinal lebih jauh; **ordinal belum menghasilkan eksperimen terbaik yang bisa diklaim dalam repository sekarang**.

Referensi angka mentah: `reports/experiment_history.json`, `reports/submission10_followup_results.json`. File-model sementara dan probabilitas validasi setiap baris tidak seluruhnya tersimpan. Hasil yang tercatat tersebut tidak boleh dinisbatkan ke `scripts/train.py` yang menggunakan pipeline lain.
