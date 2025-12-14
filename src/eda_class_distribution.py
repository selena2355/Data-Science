import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(y, save_path="../images/class_distribution.png"):
    # Membuat direktori output jika belum ada agar proses penyimpanan tidak error
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)

    # Visualisasi distribusi kelas untuk mendeteksi potensi class imbalance
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    # Menyimpan hasil visualisasi untuk dokumentasi laporan
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()