import os
import matplotlib.pyplot as plt

def plot_histograms(X, save_path="../images/histograms.png"):
    # Membuat direktori penyimpanan jika belum ada
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 10))
    
    # Menampilkan distribusi setiap fitur untuk melihat pola sebaran data
    # Digunakan untuk mendeteksi skewness dan potensi outlier
    X.hist(bins=20)
    
    # Judul umum untuk seluruh subplot histogram
    plt.suptitle("Feature Distributions (Histogram)", fontsize=14)

    plt.tight_layout()
    
    # Menyimpan hasil visualisasi sebagai bagian dari EDA
    plt.savefig(save_path)
    plt.close()
