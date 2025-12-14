import os
import tensorflow as tf
from tensorflow import keras

def train_mlp(X_train, y_train):
    # Membangun arsitektur MLP untuk data tabular
    # Input layer menyesuaikan jumlah fitur
    model = keras.Sequential([
        keras.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        # Dropout untuk mengurangi risiko overfitting
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        # Output layer dengan sigmoid untuk klasifikasi biner
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Konfigurasi proses training
    # binary_crossentropy digunakan karena target bersifat biner
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Proses training model
    # validation_split digunakan untuk memantau performa pada data validasi
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2
    )

    # Menyimpan model hasil training
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model_mlp.h5")
    model.save(model_path)

    # Mengembalikan model dan history untuk analisis training
    return model, history
