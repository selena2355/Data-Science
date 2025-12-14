import os
import joblib
from sklearn.linear_model import LogisticRegression

def train_baseline(X_train, y_train):
    # Inisialisasi Logistic Regression sebagai baseline model
    # max_iter ditingkatkan untuk memastikan proses optimasi konvergen
    model = LogisticRegression(max_iter=200)

    # Melatih model menggunakan data training
    model.fit(X_train, y_train)

    # Menentukan direktori penyimpanan model
    # Menggunakan path relatif agar kode portable dan tidak hardcoded
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    # Menyimpan model baseline untuk evaluasi dan perbandingan
    model_path = os.path.join(model_dir, "model_baseline.pkl")
    joblib.dump(model, model_path)

    # Mengembalikan model yang sudah dilatih
    return model
