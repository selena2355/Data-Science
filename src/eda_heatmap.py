import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(X, save_path="../images/correlation_heatmap.png"):
    # Membuat folder output jika belum tersedia agar file hasil bisa disimpan
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    
    # Menghitung korelasi antar fitur untuk melihat hubungan linear dan potensi multikolinearitas
    sns.heatmap(X.corr(), cmap="coolwarm", linewidths=0.5, annot=True)

    # Visualisasi ini digunakan sebagai dasar analisis feature selection
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    
    # Menyimpan heatmap sebagai dokumentasi analisis eksploratif data
    plt.savefig(save_path)
    plt.close()
