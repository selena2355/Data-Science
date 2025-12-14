from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(X, y):
    # Encode label kelas dari bentuk kategorikal ke numerik
    # Wajib untuk model ML dan DL yang hanya menerima angka
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Standardisasi fitur agar memiliki mean 0 dan varians 1
    # Penting untuk model berbasis jarak dan neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Membagi data menjadi training dan testing
    # Stratify digunakan agar distribusi kelas tetap seimbang
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    # Konversi tipe data ke float32
    # Lebih efisien secara memori dan kompatibel dengan TensorFlow
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Mengembalikan data hasil preprocessing
    # Scaler dan LabelEncoder ikut dikembalikan untuk konsistensi saat inference
    return X_train, X_test, y_train, y_test, scaler, le
