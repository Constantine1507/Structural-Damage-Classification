"""Turn actual experiment artifacts into an Indonesian report and executed reading notebook."""
import contextlib, io, json, shutil
from pathlib import Path
import nbformat


def main():
    """Read completed results; never invent metrics or rerun model selection on the audit."""
    root=Path.cwd();src=root/'artifacts/fixed_run';dest=root/'reports/fixed_run';dest.mkdir(parents=True,exist_ok=True)
    metrics=json.loads((src/'metrics.json').read_text());protocol=json.loads((src/'protocol.json').read_text());selection=json.loads((src/'frozen_selection.json').read_text());control=json.loads((src/'permutation_control.json').read_text())
    for name in ['metrics.json','protocol.json','frozen_selection.json','invariants.json','permutation_control.json','geographic_stress.json','id_proxy_diagnostic.json','bundle_verification.json','feature_importance.csv','run.log']:
        shutil.copy2(src/name,dest/name)
    shutil.copy2(src/'012_Constantine_leakage_fixed.csv',root/'submissions/012_Constantine_leakage_fixed.csv')
    geo=json.loads((src/'geographic_stress.json').read_text())
    identity=json.loads((src/'id_proxy_diagnostic.json').read_text())
    winner=metrics['winner'];report=metrics['audit_classification']
    lines=['# Eksperimen ulang setelah audit kebocoran data','',
      '## Hasil dan keputusan','',
      f"Kandidat baru `012_Constantine_leakage_fixed.csv` dihasilkan dari pelatihan ulang, bukan dari penggantian nama CSV lama. Macro F1 seleksi adalah **{winner['selection_f1']:.6f}**, sedangkan Macro F1 evaluasi akhir adalah **{metrics['audit_f1']:.6f}**. Skor Kaggle kandidat ini belum diketahui. Submission ke-10 dengan skor publik 0,60627 tetap menjadi acuan terbaik yang telah dilaporkan.",'',
      'Angka validasi pada split baru tidak dapat langsung dibandingkan dengan angka validasi eksperimen historis atau skor public leaderboard. Data historis sudah sering dipakai dalam eksperimen. Karena itu bagian audit ini hanya disebut tidak dipakai untuk seleksi dalam run ini, bukan data yang belum pernah dilihat dalam seluruh riwayat proyek.','',
      '## Apa yang diperbaiki','',
      '- Statistik jumlah dan rata-rata kelompok dihitung pada bagian fit. Baris test dan validasi tidak ikut membentuk statistik tersebut.',
      '- Fitur ID numerik utuh dihapus. Prefix dan urutan parsial ID masih digunakan sebagai hipotesis kedekatan survei, bukan sebagai koordinat geografis yang terverifikasi.',
      '- Label tetangga dibentuk secara cross-fitting tiga lipatan. Untuk setiap baris latih, seluruh lipatannya dikeluarkan dari referensi. TargetEncoder multiclass juga menggunakan cross-fitting internal.',
      '- Kalibrasi offset kelas, pemilihan model, dan evaluasi akhir mempunyai bagian data yang berbeda. Tidak ada offset yang disesuaikan memakai skor Kaggle kandidat ini.',
      '- Pipeline dan ColumnTransformer tersedia dalam implementasi. Prediksi disimpan bersama konfigurasi, indeks split, probabilitas, dan model terlatih agar hasil dapat ditelusuri.',
      '', '## Pembagian data','', '| Bagian | Baris | Fungsi |','|---|---:|---|']
    for key,meaning in [('fit','Pelatihan dan cross-fitting'),('calibration','Early stopping dan kalibrasi kelas'),('selection','Memilih model, bobot, dan kekuatan kalibrasi'),('audit','Evaluasi satu kali setelah keputusan dikunci')]:
        lines.append(f"| {key} | {protocol['rows'][key]:,} | {meaning} |")
    lines+=['','Tiga konfigurasi LightGBM dan satu CatBoost numerik dilatih. CatBoost di sini menerima keluaran numerik dari ColumnTransformer, termasuk target encoding; bukan konfigurasi categorical-native. Ada 23 resep probabilitas (model tunggal, pasangan, dan rata-rata semua model), masing-masing diuji dengan tiga kekuatan kalibrasi, sehingga total 69 kandidat keputusan pada bagian seleksi.','',
      '## Kandidat teratas pada bagian seleksi','', '| Bobot model | Macro F1 seleksi | Macro F1 kalibrasi |','|---|---:|---:|']
    for item in selection['all_candidates'][:8]:lines.append(f"| {json.dumps(item['weights'])} | {item['selection_f1']:.6f} | {item['cal_f1']:.6f} |")
    lines+=['',f"Offset log-probabilitas pemenang: `{winner['offsets']}`.",'',
      '## Evaluasi akhir','', '| Kelas | Precision | Recall | F1 | Support |','|---|---:|---:|---:|---:|']
    for k in range(1,6):
        z=report[str(k)];lines.append(f"| {k} | {z['precision']:.4f} | {z['recall']:.4f} | {z['f1-score']:.4f} | {int(z['support'])} |")
    lines+=['',f"Baseline LightGBM 47 daun tanpa kalibrasi pada audit yang sama: **{metrics['audit_raw_lgb47_f1']:.6f}**. Selisih kandidat: **{metrics['audit_f1']-metrics['audit_raw_lgb47_f1']:+.6f}**. Interval bootstrap berpasangan berdasarkan prefix enam digit, 500 pengulangan: `{metrics['paired_group_bootstrap_delta_ci95']}`. Perbandingan ini terhadap baseline run baru, bukan terhadap submission ke-10.",'',
      'Bootstrap tersebut bersyarat pada resep terpilih. Ia tidak mengoreksi keseluruhan pencarian historis. Prefix digunakan sebagai proxy kelompok, bukan bukti bahwa setiap kelompok benar-benar satu unit geografis.','',
      'Confusion matrix: baris adalah kelas aktual, kolom adalah prediksi kelas 1–5.','', '| Aktual | 1 | 2 | 3 | 4 | 5 |','|---|---:|---:|---:|---:|---:|']
    for k,row in enumerate(metrics['audit_confusion_matrix'],1):lines.append('| '+str(k)+' | '+' | '.join(map(str,row))+' |')
    weak=sorted(range(1,6),key=lambda k:report[str(k)]['f1-score'])[:2]
    lines+=['',f"Dua kelas dengan F1 terendah dalam run ini adalah kelas {weak[0]} dan {weak[1]}. Kenaikan selanjutnya perlu dinilai dari kesalahan kelas ini, bukan hanya accuracy atau perubahan jumlah label test.",'',
      '## Pemeriksaan kebocoran dan batasnya','',
      '- Uji perubahan komposisi batch prediksi lolos: menambahkan baris lain tidak mengubah fitur baris lama.',
      '- Kolom label palsu pada query tidak mengubah fitur. Kolom full building_id dan id_numeric tidak masuk ke classifier.',
      f"- Kontrol label acak pada {control['rows']:,} baris menghasilkan Macro F1 {control['macro_f1']:.6f}; kontrol mayoritas {control['majority_macro_f1']:.6f}. Hasil rendah ini konsisten dengan tidak adanya jalan pintas target yang jelas pada kontrol tersebut. Ini bukan sertifikat bebas kebocoran.",
      f"- Diagnostik geografis pada sampel {geo['rows']:,} baris, dengan jumlah fit dan holdout yang sama, menghasilkan Macro F1 {geo['splits']['random']['macro_f1']:.6f} pada split acak dan {geo['splits']['group_prefix6']['macro_f1']:.6f} pada split prefix6 yang terpisah seluruhnya. Distribusi kelas kedua holdout berbeda. Penurunan ini menunjukkan ketergantungan pada pola lokal, bukan ukuran langsung kebocoran atau estimasi penurunan skor Kaggle. Diagnostik ini tidak dipakai memilih kandidat.",
      f"- Sebanyak {identity['test_prefix6_seen_fraction']:.1%} baris test memiliki prefix6 yang sudah ada pada train. Ada {identity['unique_prefix6']:,} prefix6 pada train, dan masing-masing berada dalam satu ward_id. Karena cakupan wilayah test tumpang tindih dengan train, stress test wilayah baru menguji skenario yang lebih sulit dan berbeda, bukan pengganti langsung validasi untuk kompetisi ini.",
      '- Referensi tetangga saat cross-fitting lebih kecil daripada saat inferensi. Perubahan kepadatan referensi dapat menimbulkan pergeseran fitur walaupun label tidak bocor.',
      '- Early stopping dan offset memakai bagian kalibrasi yang sama. Bagian seleksi terpisah mengurangi risiko memilih kalibrasi yang hanya bagus di sana; audit tetap diperlukan.',
      '- Menambah iterasi sampai satu angka validasi mencapai 0,65 berisiko menyesuaikan model terhadap noise. Protokol run ini dikunci sebelum hasil audit dilihat. Tidak ada klaim target 0,65 sudah tercapai.','',
      '## Integritas CSV','',
      f"CSV berisi 228.629 baris dengan header `building_id,damage_grade`, ID unik sesuai urutan sample submission, dan label integer 1–5. Ada {metrics['changed_vs_submission10']:,} prediksi yang berbeda dari submission ke-10.",'',
      f"SHA256: `{metrics['submission_sha256']}`.",'',
      f"Jumlah prediksi per kelas: `{metrics['test_class_counts']}`.",'',
      '## Menjalankan ulang','',
      '```bash','python -m scripts.run_fixed_experiment --data data --output artifacts/fixed_run','python -m scripts.check_fixed_control --data data','python -m scripts.check_geographic_stress --data data','python -m scripts.verify_fixed_delivery --data data','python -m scripts.summarize_fixed_run','python -m pytest -q tests/test_leakage_fixed.py tests/test_preprocess.py','```','',
      'Letakkan CSV asli pada folder data. Untuk kontrol label acak, sesuaikan parameter --data jika memakai lokasi berbeda. Notebook 07 membaca hasil yang benar-benar dihasilkan eksperimen dan menyediakan perintah pelatihan ulang. Bundle model memuat referensi latih sehingga tidak dipublikasikan ke GitHub.','',
      'Rekomendasi: uji kandidat CSV ini di Kaggle. Pertahankan submission ke-10 sebagai pilihan terbukti sampai hasil kandidat baru tersedia.']
    (dest/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    cells=[nbformat.v4.new_markdown_cell('# Perbaikan preprocessing dan validasi\n\nNotebook ini membaca keluaran eksperimen aktual setelah pelatihan ulang. Angka Kaggle untuk kandidat baru belum tersedia. Kode lengkap ada di `scripts/leakage_fixed.py` dan `scripts/run_fixed_experiment.py`.'),
      nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nROOT = Path.cwd()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nfolder = ROOT / 'reports/fixed_run'\nmetrics = json.loads((folder / 'metrics.json').read_text())\nprotocol = json.loads((folder / 'protocol.json').read_text())\nselection = json.loads((folder / 'frozen_selection.json').read_text())\nprint(json.dumps(protocol['rows'], indent=2))"),
      nbformat.v4.new_markdown_cell('## Protokol\n\nStatistik konteks hanya dipelajari dari fit. Neighbor target dan TargetEncoder menggunakan cross-fitting. Kalibrasi, seleksi, dan audit terpisah. ID utuh tidak menjadi fitur; fragmen ID masih menjadi proxy lokasi. Audit ini tidak digunakan untuk pemilihan pada run baru, tetapi bukan holdout yang dapat dijamin belum pernah terekspos selama seluruh riwayat eksperimen.'),
      nbformat.v4.new_code_cell("print(pd.DataFrame(selection['all_candidates']).head(10).to_string(index=False))"),
      nbformat.v4.new_code_cell("print('Audit Macro F1:', metrics['audit_f1'])\nprint('Baseline audit:', metrics['audit_raw_lgb47_f1'])\nprint(pd.DataFrame({k:v for k,v in metrics['audit_classification'].items() if isinstance(v, dict)}).T.to_string())\nprint('Confusion matrix:')\nprint(pd.DataFrame(metrics['audit_confusion_matrix'], index=range(1,6), columns=range(1,6)).to_string())"),
      nbformat.v4.new_code_cell("print((folder / 'invariants.json').read_text())\nprint((folder / 'permutation_control.json').read_text())"),
      nbformat.v4.new_code_cell("submission = pd.read_csv(ROOT / 'submissions/012_Constantine_leakage_fixed.csv', dtype={'building_id':str})\nassert list(submission.columns) == ['building_id','damage_grade']\nassert len(submission) == 228629 and submission.building_id.is_unique\nassert submission.damage_grade.isin([1,2,3,4,5]).all()\nprint(submission.head().to_string(index=False))\nprint(submission.damage_grade.value_counts().sort_index().to_string())"),
      nbformat.v4.new_markdown_cell('## Pelatihan ulang\n\nJalankan `python -m scripts.run_fixed_experiment --data data --output artifacts/fixed_run` dari root repository. Pelatihan berat dijalankan melalui CLI, bukan diulang ketika notebook analisis ini dibuka. `scripts/predict_fixed.py` dapat memprediksi dari bundle yang telah disimpan. Rincian interpretasi dan keterbatasan berada di `reports/fixed_run/ANALYSIS.md`.')]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    scope={};count=0
    for cell in nb.cells:
        if cell.cell_type=='code':
            count+=1;stream=io.StringIO()
            with contextlib.redirect_stdout(stream):exec(compile(cell.source,'<executed analysis notebook>','exec'),scope)
            cell.execution_count=count;cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue())]
    nbformat.validate(nb);nbformat.write(nb,root/'notebooks/07_leakage_fixed_experiment.ipynb')
    print('REPORT_AND_EXECUTED_NOTEBOOK_WRITTEN')

if __name__=='__main__':main()
