# Audit lanjutan setelah public score 0,60621

## Keputusan saat ini

Submission 10 (0,60627) tetap menjadi acuan public terbaik yang dilaporkan. Kandidat 011 memperoleh **0,60621**, selisih **−0,00006** atau −0,006 poin persentase. Kedua skor adalah laporan pengguna; private score tidak tersedia. Hasil baru dikaitkan dengan CSV kandidat yang terakhir dikirim, dengan asumsi file diunggah tanpa perubahan isi.

Penurunan tersebut **tidak membuktikan kebocoran data**, tetapi juga tidak membuktikan pipeline bersih. Temuan utama audit ini adalah penggunaan validation berulang untuk pemilihan, konteks transduktif pada notebook asli, dan ketergantungan pada identifier lokasi. Belum ditemukan bukti penggunaan label test dalam pembentukan fitur yang diperiksa. Ada bagian provenance yang tidak dapat diaudit ulang karena checkpoint eksperimen baru tidak tersimpan.

## Pemeriksaan baru yang benar-benar dijalankan

Audit memakai notebook asli `Structural_Damage_NonNeural_Complete(1).ipynb`, ZIP kompetisi asli dan kedua CSV final dari repository. SHA-256 data sama dengan arsip eksperimen:

- train: `231c9bf0430faa9c1d2281b4edbcd9ce0d7a863c6ff67042a8fbc992469d37bb`;
- test: `d732c3070b878ebc1346907d50e6a6c1a6a9db0eadf0790631b37c03d3ba6c5c`.

Train **533.465 baris**, test **228.629 baris**, nol missing, nol ID duplikat per file, dan nol irisan ID train/test. Tidak ada kolom dengan nama `post_eq` atau `damage` pada matriks fitur setelah target dikeluarkan. Tidak adanya nama tersebut hanya pemeriksaan schema, bukan pembuktian seluruh provenance semantik setiap kolom.

### Uji manipulasi label held-out

Diambil 2.000 baris deterministik, dibagi menjadi FIT 1.500 dan held-out 500. Seluruh label held-out digeser ke kelas lain, lalu fitur dihitung ulang. Hasil keenam fungsi berikut **identik sebelum/sesudah perubahan label held-out**:

| Fungsi asli | Jumlah fitur | Label held-out memengaruhi hasil? |
|:--|--:|:--|
| `compact_frame` | 76 | Tidak pada uji ini |
| `kernel_features` | 136 | Tidak pada uji ini |
| `physical_kernels` | 198 | Tidak pada uji ini |
| `pair_features` | 210 | Tidak pada uji ini |
| `hierarchical_profiles` | 168 | Tidak pada uji ini |
| `numeric_neighbours` | 56 | Tidak pada uji ini |

Kode pembentuk referensi memisahkan OOF training dan menggunakan FIT untuk baris di luar FIT. Prior kelas dalam fungsi yang diperiksa dihitung dari label referensi. `read_data` mengeluarkan damage_grade sebelum membuat fitur. `fit_crossfitted_meta` juga membentuk prediksi OOF pada CAL untuk meta-model.

Ini bukti terbatas yang mendukung tidak adanya label held-out langsung pada jalur tersebut, **bukan sertifikat bebas leakage untuk semua model, seluruh dataset, cache lama, dan proses seleksi**. Uji ini tidak mengulang training ensemble atau menguji skor model dengan label acak.

## Temuan 1 — selection bias sudah nyata dalam prosedur

Kandidat dipilih setelah 45 konfigurasi koreksi; tiga finalis kemudian dibandingkan lagi pada confirmation. Nama “confirmation” tidak membuatnya tetap independen jika hasilnya dipakai untuk menentukan pemenang. Metadata historis sudah menandai `selection_is_adaptive=True` dan `pristine_independent_test=False`.

| Evaluasi | Submission 10 | Kandidat 011 | Delta |
|:--|--:|--:|--:|
| SEL | 0.6001154955 | 0.6002676456 | +0.0001521502 |
| Confirmation gabungan | 0.6011875286 | 0.6012892715 | +0.0001017429 |
| Public, laporan pengguna | 0.60627 | 0.60621 | −0.00006 |

Delta confirmation per fold: **+0,000525646; −0,000029369; −0,000158652**. Perbaikan terutama ditopang satu fold. Interval bootstrap conditional 95% [−0,000265138; +0,000481916] mencakup nol dan tidak mengoreksi pencarian banyak kandidat. Hasil Kaggle yang berbalik arah konsisten dengan keuntungan lokal yang lemah; penyebab spesifik belum dapat ditetapkan dari dua angka agregat.

Koreksi terhadap keputusan sebelumnya: kandidat 011 layak disebut eksperimen eksploratif, tetapi bukti saat itu belum cukup untuk menyatakannya lebih baik atau merekomendasikan penggantian submission 10.

## Temuan 2 — konteks notebook asli bersifat transduktif

`confirm_v7_recipe` membentuk `reference = pd.concat([d, t])`. `contextual_covariates` menghitung jumlah, mean, standar deviasi dan proporsi kategori dari seluruh reference. `make_v7_frame`/`build_member_frame` memasukkan hasil tersebut ke fitur. Dengan demikian, atribut validasi/test dapat mengubah representasi baris training.

Uji tambahan: menambahkan satu baris tanpa label dengan atribut diubah mengubah fitur konteks pada **empat dari 2.000 baris lama**. Jalur dependensinya terbukti, walaupun dampak terhadap skor belum diukur.

Ini **bukan kebocoran label test**. Untuk protokol inductive, ini kontaminasi preprocessing: statistik seharusnya dipelajari dari training fold saja. Untuk kompetisi transduktif, penggunaan atribut test tanpa label mungkin diizinkan, tetapi bergantung aturan kompetisi; audit ini tidak menyimpulkan pelanggaran. Klaim “semua statistik hanya dari training” tidak berlaku untuk notebook asli.

Antisipasi: buat jalur inductive terpisah. Setiap outer fold mem-fit tabel konteks hanya dari outer-training. Validation/test hanya melakukan lookup dengan fallback statistik training. Ukur ablasi konteks train-only versus transduktif pada split dan model yang sama. Jangan mengubah klaim skor submission 10 menjadi skor dari pipeline train-only yang belum dilatih.

## Temuan 3 — ID lengkap masih terdapat pada metode asli

`engineer` membuat `id_numeric`, `id_house_order`, dan `id_unit_order`; `COMPACT` memasukkan `id_numeric`. Jadi README sebelumnya terlalu luas ketika menyatakan ID lengkap tidak pernah masuk model. Pernyataan tersebut benar untuk `scripts/feature_engineering.py` yang baru, bukan keseluruhan ensemble asli.

ID tidak otomatis menjadi target leakage. Akan tetapi urutan pencatatan, kelompok bangunan, geografi, dan kedekatan survey dapat termuat dalam ID. Random split dapat mempertahankan struktur yang sama pada kedua sisi, sehingga performa terhadap wilayah baru tampak lebih baik daripada kenyataannya.

Semua district, municipality dan ward pada test terdapat pada train. Sebanyak **19.670 baris test (8,603%)** memiliki prefix 11 digit yang juga muncul pada train. Prefix ini hanya proxy kelompok; tanpa kamus data, jangan menyebutnya pasti satu rumah atau satu objek.

Antisipasi: bandingkan empat ablasi yang ditetapkan sebelumnya: (a) atribut fisik tanpa ID/geografi, (b) geografi eksplisit, (c) prefix kasar, (d) ID/order halus. Tambahkan geographic group holdout sebagai stress test. Group holdout mengukur transfer ke wilayah tidak terlihat dan tidak identik dengan tugas random split kompetisi; jangan membandingkan dua jenis split seolah mengukur sasaran yang sama.

## Temuan 4 — overlap profil dan label tidak konsisten

**23.781 baris test (10,402%)** memiliki profil fitur identik dengan setidaknya satu baris train, setelah building_id dikeluarkan. **56.682 baris train** berada dalam kelompok profil berulang. Dari sana, **10.695 kelompok / 29.104 baris (5,46% train)** mengandung lebih dari satu damage_grade.

Kesamaan profil tidak membuktikan duplikat objek: bangunan berbeda dapat memiliki atribut survey yang sama. Label berbeda tidak otomatis salah label. Tidak boleh menghapus semua konflik atau menyalin label dari profil yang cocok dengan keyakinan mutlak.

Antisipasi: lakukan split berbasis hash profil dan, terpisah, prefix kelompok sebagai stress test; ukur seen-profile versus unseen-profile. Untuk agregasi target, gunakan smoothing dan OOF di dalam fold. Jika benar-benar ada data duplikat satu objek, kelompokkan sebelum splitting berdasarkan bukti identitas objek, bukan hanya kesamaan fitur.

## Mengapa koreksi lantai dasar belum membantu?

685 baris berubah, **0,2996%** dari test. Perubahan terbesar adalah kelas 2→1 (183), 5→4 (157), 3→4 (147), dan 3→2 (80). Akibat bersihnya:

| Kelas | Perubahan jumlah prediksi |
|:--|--:|
| 1 | +205 |
| 2 | −145 |
| 3 | −194 |
| 4 | +280 |
| 5 | −146 |

Koreksi menggeser keputusan batas kelas; tidak menambah informasi fisik baru. Kelas 2 dan 3 memang lemah dalam evaluasi historis, tetapi pengurangan prediksi keduanya dapat menaikkan precision sambil menurunkan recall. Tanpa label test, kita tidak tahu perubahan mana yang benar atau kelas mana yang membuat public turun.

| Lantai dasar | Baris test | Berubah | Proporsi berubah |
|:--|--:|--:|--:|
| Mud | 185.399 | 271 | 0,146% |
| Brick/Stone | 19.829 | 203 | 1,024% |
| RC | 22.003 | 202 | 0,918% |
| Timber | 1.087 | 9 | 0,828% |

Dampak relatif jauh lebih besar pada Brick/Stone dan RC daripada Mud. Ini membuat hasil sensitif terhadap komposisi subset public/private dan kualitas kalibrasi per kelompok. Perbedaan komposisi itu masih hipotesis, bukan temuan tentang label atau pembagian Kaggle.

## Pengamanan untuk eksperimen berikutnya

1. **Bekukan submission 10.** Simpan hash file, skor public beserta sumbernya, dan jangan menimpanya. Kandidat 011 dicatat tidak mengungguli acuan public.
2. **Kunci protokol sebelum mencoba model baru.** Batasi daftar hipotesis, seed, metrik dan aturan pemilihan. Outer fold untuk penilaian, inner fold untuk pemilihan fitur/model/encoding/blend/threshold. Jika outer score dipakai lagi untuk memilih, beri label adaptif. Mengganti seed tidak menghapus riwayat penggunaan dataset.
3. **Rebuild fitur di dalam setiap outer fold.** Bukan membuat fitur target encoding OOF sekali untuk seluruh dataset lalu melakukan CV model. OOF global masih dapat memakai label outer-validation sebagai referensi fitur outer-training. Encoder, neighbor label bank, feature selection, class weights, resampling dan early stopping harus mengikuti batas fold.
4. **Uji invariant sebelum pelatihan besar.** FIT/CAL/VAL harus disjoint pada ID; referensi label setiap query harus mengecualikan query dan outer-validation. Pada jalur inductive, menambah/mengubah X_test tidak boleh mengubah state transformer yang sudah di-fit. Jalankan label-sentinel dan label-permutation controls; jangan mengharapkan batas Macro F1 acak yang selalu persis 0,2.
5. **Pisahkan evaluasi random dan group.** Gunakan random stratified CV untuk sasaran kompetisi jika itu sesuai split penyelenggara, lalu group-profile/ward untuk robustness. Laporkan keduanya, jangan mengambil skor tertinggi dari berbagai split.
6. **Sederhanakan keputusan.** Bandingkan base ensemble, bobot kelas global dan offset ringan. Hindari koreksi kategori berlapis yang dipilih dari perubahan Macro F1 sekitar 0,0001. Nilai stabilitas per kelas, per fold dan per kelompok, bukan hanya pooled score.
7. **Simpan artefak sebelum berganti sesi.** ID setiap split, hash kode/data, daftar fitur, konfigurasi, model, probabilitas OOF, encoder reference IDs, rule kalibrasi dan CSV. Hash CSV memverifikasi identitas file; tidak membuktikan pipeline bebas leakage.
8. **Batasi umpan balik leaderboard.** Public bukan sumber label untuk tuning. Prioritaskan satu kandidat dengan bukti lokal yang lebih kuat, sesuai kuota kompetisi. Skor private belum dapat dipastikan dari public.

Untuk peningkatan model yang lebih besar, prioritaskan fitur/model yang menghasilkan pola kesalahan berbeda dan diagnosis kelas 2–3, bukan melanjutkan pencarian offset serupa tanpa batas. Target 0,65 masih selisih 0,04373 dari acuan public dan tidak dapat dijamin dengan tuning tambahan. Audit ini tidak melatih atau mengklaim menghasilkan model baru dengan skor lebih tinggi.

## Cara menjalankan audit

Dari root repository, sediakan notebook asli dan CSV kompetisi yang sudah diekstrak:

```bash
python reports/deep_audit/run_checks.py --notebook "path/to/Structural_Damage_NonNeural_Complete(1).ipynb" --data path/to/data
```

Script membaca definisi fungsi dari notebook tanpa mengeksekusi notebook secara keseluruhan. Hanya jalankan pada notebook asli tepercaya. Hasil terukur tersimpan dalam `check_results.json`. File model/checkpoint eksperimen baru yang hilang tidak direkonstruksi dalam audit ini.

## Rujukan metodologi

- scikit-learn, Common pitfalls and recommended practices: https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn, Nested versus non-nested cross-validation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- scikit-learn, Target Encoder's Internal Cross fitting: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html
