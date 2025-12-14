import os
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Menentukan base directory agar path model konsisten
# dan tidak bergantung pada lokasi eksekusi script
BASE_DIR = os.path.dirname(__file__)

def evaluate_sklearn(model_name, X_test, y_test):
    # Memuat model sklearn yang sudah disimpan
    model_path = os.path.join(BASE_DIR, "..", "models", model_name)
    model = joblib.load(model_path)

    # Melakukan prediksi pada data uji
    pred = model.predict(X_test)

    # Menampilkan hasil evaluasi klasifikasi
    # Digunakan untuk melihat performa precision, recall, f1-score, dan error antar kelas
    print("Evaluation for:", model_name)
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))

def evaluate_mlp(model_name, X_test, y_test):
    # Memuat model deep learning (MLP) yang disimpan dalam format Keras
    model_path = os.path.join(BASE_DIR, "..", "models", model_name)
    model = tf.keras.models.load_model(model_path)

    # Output model berupa probabilitas, sehingga perlu threshold
    # Threshold 0.5 digunakan untuk konversi ke kelas biner
    pred = (model.predict(X_test) > 0.5).astype("int32")

    # Evaluasi performa model MLP
    print("Evaluation for:", model_name)
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
