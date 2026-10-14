# Metode yang digunakan

## 1. Tujuan dan data

Saya mengerjakan klasifikasi kerusakan bangunan lima kelas (Grade 1–5) untuk data gempa Nepal 2015. Dataset yang digunakan pada eksperimen historis mempunyai **533.465 baris train** dan **228.629 baris test**. Metrik kompetisi adalah **Macro F1**: F1 dihitung pada tiap kelas, kemudian kelimanya dirata-ratakan dengan bobot sama. Dengan metrik ini, peningkatan kelas yang jumlahnya kecil tetap berpengaruh.

Kolom identitas bangunan diperlakukan sebagai *string* supaya angka nol awal tidak hilang. Sebelum modeling, saya memeriksa ID ganda, irisan train–test, missing value, perbedaan kolom, serta format target. Pada arsip yang pernah diperiksa, tidak ditemukan missing value dan tidak ada ID bangunan yang muncul pada train sekaligus test. **Contoh target di sample_submission bukan label test.** Distribusinya nyaris rata sehingga tidak boleh digunakan sebagai ground truth.

## 2. Eksplorasi dan interpretasi fitur

Saya mulai dari karakteristik bangunan sebelum gempa: umur, luas, jumlah lantai, tinggi, fondasi, bentuk lantai, atap, dan material superstructure. Beberapa turunan yang digunakan/diuji adalah log umur dan luas, rasio tinggi per lantai, luas per lantai, serta interaksi usia bangunan dengan jumlah lantai. Rasio dibangun dengan penanganan pembagi nol dan imputasi pada pipeline.

Kode distrik/ward atau potongan depan `building_id` bisa memuat informasi wilayah. Namun saya memperlakukannya hanya sebagai **proxy**, bukan koordinat geografis dan bukan variabel sebab-akibat. Eksperimen historis yang menggunakan target encoding geografis harus melakukan cross-fitting (OOF) pada train agar label validasi tidak bocor ke fitur. Pipeline modular sekarang hanya memakai hitungan prefix tanpa label, yang di-fit dari data training pada masing-masing split.

Data berisi satu kategori `superstructure` dengan 11 kemungkinan material, bukan 11 kolom `has_superstructure_...`. Saya menjaga kedua format tersebut agar tidak dicampur dalam penafsiran.

## 3. Model yang saya eksplorasi

Model dasar berasal dari keluarga **gradient boosting untuk data tabular**, terutama LightGBM. Saya juga membandingkan variasi kapasitas model (jumlah daun dan pohon), pemberian bobot kelas, `multiclass` versus `multiclassova`, fitur wilayah lebih kasar, serta smoothing pada encoding. CatBoost dengan kategori asli ikut dimasukkan sebagai kandidat dalam workflow modular.

Pada eksperimen historis saya mencoba model yang dilatih sendiri dan *blend* probabilitasnya dengan baseline yang dibekukan. Ada eksperimen koreksi keputusan kelas menurut distrik, kemudian menurut kelompok karakteristik bangunan (usia, jenis lantai, material, fondasi, atap, dan sebagainya). Tujuan koreksi adalah menekan kekeliruan pada kelas yang sering tertukar, bukan mengubah label asli.

Tidak ada neural network yang digunakan. Daftar angka eksperimen dan konfigurasi yang tersimpan ada di [CATATAN_EKSPERIMEN.md](CATATAN_EKSPERIMEN.md), [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md), serta [SUBMISSION_10_AND_NEXT.md](SUBMISSION_10_AND_NEXT.md).

## 4. Pemisahan data dan evaluasi

Dalam eksperimen historis, terdapat beberapa tahap data yang disebut **FIT**, **CAL**, **SEL**, dan confirmation fold:
- **FIT** untuk mempelajari parameter model/encoder; statistik dari target tidak boleh dihitung dari bagian validasi.
- **CAL** untuk menentukan koreksi atau bobot yang memerlukan label.
- **SEL** untuk memilih kandidat. Skornya adalah skor pemilihan, bukan test independen.
- **Confirmation** historis untuk memeriksa perubahan pada fold lain. Karena riwayat tuning sebelumnya sudah melihat data yang berkaitan, fold ini *bukan* bukti independen yang bersih.

Pipeline modular `scripts/train.py` menggunakan desain lain: 70% FIT, 15% CAL, 15% AUDIT. Pembagian dilakukan secara stratified; AUDIT digunakan sesudah pemilihan kandidat. Saya tidak menyamakan skor pipeline baru dengan hasil eksperimen historis.

Untuk eksperimen yang melibatkan koreksi keputusan per kelas, saya juga menyimpan kontrol tanpa koreksi. Kenaikan kecil perlu dilihat pada data yang sama, bukan sekadar dibandingkan dua skor dari pembagian berbeda.

## 5. Pemeriksaan kebocoran data

Beberapa risiko yang saya temukan/perhatikan:
1. **Target encoding sebelum split** dapat mengirim label baris validasi kembali sebagai prediktor. Solusi: cross-fitting pada training dan mapping hanya dari training untuk validasi/test.
2. **Menggunakan label post-earthquake** dapat memberikan petunjuk langsung mengenai akibat gempa. Pada dataset yang diperiksa tidak ada fitur `_post_eq`; transformasi tetap membuangnya bila ditemukan.
3. **Prefix ID dan kode lokasi** berisiko menghafal pola wilayah. Uji tanpa geografi adalah kontrol yang perlu dipertahankan.
4. **Mengulang tuning pada fold yang sudah pernah dilihat** menyebabkan overfitting evaluasi. Mengganti seed saja bukan jaminan set validasi baru yang independen.
5. **Menggunakan public leaderboard sebagai alat tuning terus-menerus** dapat membuat keputusan terlalu cocok pada sebagian test yang terbuka.

Dari audit fitur identik (selain ID dan label), **29.104 baris training atau 5,46%** berada dalam kelompok yang memiliki fitur sama tetapi grade berbeda. Saya tidak langsung membuang baris ini; kemungkinan variabel penjelas yang tersedia memang belum cukup mengidentifikasi kerusakan dengan tepat.

## 6. Menentukan hasil yang layak dipakai

Saya mempertahankan submission terbaik ketika kandidat baru hanya meningkat tipis pada confirmation dan interval bootstrap selisihnya masih memuat nol. Untuk kandidat submission 11, perbedaan lokal terhadap submission 10 adalah sekitar **+0,000102** Macro F1 pada pooled confirmation. Interval 95% yang tercatat **[-0,000265; +0,000482]**, sehingga peningkatan belum meyakinkan. Selain itu pemilihan kandidat bersifat adaptif, yang tidak seluruhnya ditangani oleh bootstrap.

**Batas reproduksi:** source code modular tersedia dan dapat dieksekusi dengan data kompetisi. Namun checkpoint historis yang membuat skor publik 0,60627 tidak tersedia lengkap, sehingga notebook baru tidak mengklaim membangun ulang model terbaik itu dari nol.

## 7. Keputusan akhir: submission 010 dan eksperimen 012

Pada tahap pengumpulan, saya memisahkan hasil terbaik menurut leaderboard dari eksperimen perbaikan yang baru selesai:

- **010** (`submissions/010_Constantine(10).csv`) tetap menjadi submission utama. Public Macro F1 **0,60627** berasal dari catatan leaderboard; training ensemble historisnya belum berhasil direproduksi persis dari data mentah.
- **012** (`submissions/012_Constantine_leakage_fixed.csv`) adalah percobaan lanjutan menggunakan `SafeFeatures`, target encoding dan label tetangga yang di-cross-fit, serta partisi FIT/CAL/SEL/AUDIT yang terpisah. Model pemenang 25% LightGBM + 75% CatBoost memperoleh Macro F1 lokal **0,602359** pada AUDIT. Skor Kaggle 012 belum diketahui.

Audit 012 memakai 32.009 baris, dengan baseline LightGBM **0,590189** pada baris yang sama. Selisih lokal **+0,012170** tidak dapat digunakan untuk menyimpulkan 012 mengalahkan public score 010. Prediksi final 012 dilaporkan cocok byte demi byte dengan replay dari bundle model yang tersimpan pada lingkungan eksperimen; bundle tersebut tidak dipublikasikan di repository.

Untuk 012, kondisi model dan lingkungan eksekusi tercatat dalam [`fixed_run/ANALYSIS.md`](fixed_run/ANALYSIS.md), [`fixed_run/protocol.json`](fixed_run/protocol.json), [`fixed_run/metrics.json`](fixed_run/metrics.json), dan [`fixed_run/refit.log`](fixed_run/refit.log). Riwayat tuning pada dataset yang sama tetap menjadi batas dalam membaca skor audit. Sampai ada skor Kaggle baru yang terverifikasi, saya mempertahankan 010 sebagai hasil utama.
