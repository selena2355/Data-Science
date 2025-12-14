import os
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train):
    # Inisialisasi Random Forest dengan parameter utama
    # n_estimators menentukan jumlah pohon
    # max_depth membatasi kompleksitas model agar tidak overfitting
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    # Proses training model menggunakan data training
    model.fit(X_train, y_train)

    # Menentukan direktori penyimpanan model
    # Path dibuat relatif agar kode bisa dijalankan di environment lain
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    # Menyimpan model hasil training untuk digunakan kembali (inference/evaluasi)
    model_path = os.path.join(model_dir, "model_rf.pkl")
    joblib.dump(model, model_path)

    # Mengembalikan model yang sudah dilatih
    return model
