
# 📘 Judul Proyek
Perbandingan Model Machine Learning dan Deep Learning dalam Klasifikasi Varietas Beras Cammeo dan Osmancik

## 👤 Informasi
- **Nama:** Hanifah Alya Nuraini
- **Repo:** (https://github.com/selena2355/Data-Science.git)  
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan sesuai domain  
- Melakukan data preparation  
- Membangun 3 model: **Baseline**, **Advanced**, **Deep Learning**  
- Melakukan evaluasi dan menentukan model terbaik  

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
1. Proses identifikasi varietas beras Cammeo dan Osmancik secara manual sulit dilakukan secara konsisten karena kemiripan karakteristik fisik antar varietas.
2. Dataset beras memiliki banyak fitur numerik hasil ekstraksi citra yang sulit dianalisis tanpa pendekatan berbasis machine learning.
3. Diperlukan model klasifikasi yang mampu mempelajari pola non-linear antar fitur untuk meningkatkan akurasi identifikasi varietas beras.
4. Dibutuhkan perbandingan antara model Machine Learning dan Deep Learning untuk menentukan pendekatan yang paling efektif dalam mengklasifikasikan varietas beras.

**Goals:**  
1. Membangun model klasifikasi untuk membedakan varietas beras Cammeo dan Osmancik menggunakan data fitur morfologi dengan tingkat akurasi yang baik.
2. Mengimplementasikan dan membandingkan tiga pendekatan pemodelan, yaitu model baseline (Logistic Regression), model Machine Learning lanjutan (Random Forest), dan model Deep Learning (Multilayer Perceptron).
3. Mengevaluasi performa masing-masing model menggunakan metrik evaluasi yang sesuai untuk klasifikasi, seperti Accuracy dan F1-Score. 

---
## 📁 Struktur Folder
```
project/
│
├── data/                               # Dataset (tidak di-commit, download manual)
│
├── images/                             # Visualizations
│   ├── class_distribution.png
│   ├── confusion_matrix_model1.png
│   ├── confusion_matrix_model2.png
│   ├── confusion_matrix_model3.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── histograms.png
│   ├── history_accuracy.png
│   ├── history_loss.png
│   ├── visualisasi_accuracy.png
│   └── visualisasi_waktu.png
│
├── models/                             # Saved models
│   ├── model_baseline.pkl
│   ├── model_mlp.h5
│   └── model_rf.pkl
│
├── notebooks/                          # Jupyter notebooks
│   └── ML_Project.ipynb
│
├── src/                                # Source code
│   ├── eda_class_distribution.py
│   ├── eda_heatmap.py
│   ├── eda_histogram.py
│   ├── evaluate.py                     # disediakan sebagai modul evaluasi opsional dan tidak digunakan langsung dalam notebook eksperimen.
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_advanced.py
│   ├── train_baseline.py
│   └── train_deep_learning.py
│   
├── .gitignore
├── Checklist Submit.md                  # Checklist
├── Laporan Proyek Machine Learning.md   # Laporan
├── README.md                     
└── requirements.txt                     # Dependencies
```
---

# 3. 📊 Dataset
- **Sumber:** (https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) 
- **Jumlah Data:** 3810 
- **Tipe:** Tabular  

### Fitur Utama
| Nama Fitur        | Deskripsi                                                    |
| ----------------- | ------------------------------------------------------------ |
| Area              | Luas area butir beras hasil segmentasi citra                 |
| Perimeter         | Panjang keliling (boundary) butir beras                      |
| Major_Axis_Length | Panjang sumbu utama elips yang memodelkan bentuk butir beras |
| Minor_Axis_Length | Panjang sumbu minor elips yang memodelkan bentuk butir beras |
| Eccentricity      | Tingkat kelonjongan bentuk elips (0–1)                       |
| Convex_Area       | Luas area convex hull dari butir beras                       |
| Extent            | Rasio antara area objek dengan bounding box                  |
| Class             | Label varietas beras (target klasifikasi)                    |

---

# 4. 🔧 Data Preparation
Transformasi:
- Encoding
- Scaling

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Logistic Regression  
- **Model 2 – Advanced ML:** Random Forest  
- **Model 3 – Deep Learning:** Multilayer Perceptron  

---

# 6. 🧪 Evaluation
**Metrik:**
- Accuracy
- F1-Score 

### Hasil Singkat
| Model | Score (Accuracy) | Catatan |
|-------|------------------|---------|
| Logistic Regression | 0.916 | Cocok sebagai pembanding awal, cepat dan efisien |
| Random Forest | 0.919 | Memberikan performa terbaik secara keseluruhan |
| MLP | 0.915 | Tidak memberikan peningkatan signifikan dibanding model klasik |


---

# 7. 🏁 Kesimpulan
- Model terbaik: Random Forest 
- Alasan: 
    - Memberikan performa terbaik  
    - Menghasilkan jumlah kesalahan paling rendah
    - Memberikan keseimbangan yang baik antara performa dan kompleksitas.
- Insight penting: 
    - Model machine learning tradisional seperti Random Forest dapat mengungguli deep learning pada data tabular dengan ukuran kecil hingga menengah.
    - Deep learning (MLP) tidak selalu memberikan peningkatan performa yang signifikan, terutama jika kompleksitas data tidak terlalu tinggi.

---

# 8. 🔮 Future Work
✅ Feature engineering lebih lanjut

✅ Hyperparameter tuning lebih ekstensif

✅ Ensemble methods (combining models)

✅ Membuat API (Flask/FastAPI)

✅ Membuat web application (Streamlit/Gradio)

✅ Improving inference speed

✅ Reducing model size

---

# 9. 🔁 Reproducibility

**Python Version:** 3.12.5

**Main Libraries & Versions:**
numpy==2.3.5  
pandas==2.3.3  
scikit-learn==1.8.0  
matplotlib==3.10.8  
seaborn==0.13.2  
joblib==1.5.2  

# Deep Learning Framework (pilih salah satu)
tensorflow_cpu==2.20.0 

**Additional Libraries:**
ucimlrepo – digunakan untuk mengunduh dataset dari UCI Machine Learning Repository secara langsung
