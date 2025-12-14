import pandas as pd
from scipy.io import arff

def load_rice(path="../data/Rice_Cammeo_Osmancik.arff"):
    # Memuat dataset ARFF dan metadata-nya
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Label kelas dari file ARFF sering bertipe bytes
    # Perlu didecode agar konsisten sebagai string untuk preprocessing dan modeling
    if df['Class'].dtype == object:
        df['Class'] = df['Class'].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

    return df

if __name__ == "__main__":
    # Digunakan untuk pengujian cepat saat file dijalankan langsung
    # Memastikan data berhasil dimuat dan distribusi kelas sesuai
    df = load_rice()
    print(df.head())
    print(df['Class'].value_counts())
