import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import mlflow
import os

# --- Konfigurasi MLflow ---
# Mengaktifkan autologging untuk scikit-learn
# Autologging akan secara otomatis melacak parameter, metrik, dan model
mlflow.sklearn.autolog()

# --- Fungsi Preprocessing Ulang (untuk simulasi, data harus distandarisasi) ---
def load_and_preprocess_data(file_path: str):
    """Memuat data yang sudah bersih dan melakukan standarisasi ulang
       seperti langkah di Kriteria 1."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {file_path}. Pastikan data sudah disalin.")
        return None, None, None, None
        
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Standarisasi Ulang (WAJIB, karena model harus dilatih dengan data terstandar)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Pembagian Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# --- Fungsi Pelatihan Model ---
def train_model():
    # Jalur file hasil preprocessing dari Kriteria 1
    # Ganti jalur jika diperlukan!
    PREPROCESSED_DATA_PATH = './namadataset_preprocessing/cleaned_dataset.csv' 
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data(PREPROCESSED_DATA_PATH)
    
    if X_train is None:
        return

    # Inisialisasi Eksperimen MLflow
    # Menggunakan nama eksperimen 'Diabetes_Prediction_Basic'
    mlflow.set_experiment("Diabetes_Prediction_Basic")
    
    # Memulai MLflow Run untuk mencatat semua proses
    with mlflow.start_run() as run:
        print("\n--- MLflow Run Dimulai ---")
        
        # 1. Definisikan Model (Model Klasifikasi Sederhana)
        model = LogisticRegression(solver='liblinear', random_state=42)
        
        # 2. Latih Model
        model.fit(X_train, y_train)
        print("Model Logistic Regression berhasil dilatih.")
        
        # 3. Evaluasi (Autologging sudah mencatat metrik dasar seperti 'accuracy')
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Akurasi Model pada Testing Set: {accuracy:.4f}")
        
        # Autologging akan mencatat:
        # - Parameter model (misalnya solver, random_state)
        # - Metrik evaluasi (accuracy, f1-score, precision, recall)
        # - Model terlatih sebagai artefak
        
        print(f"MLflow Run ID: {run.info.run_id}")
        print("Semua artefak dan metrik telah dicatat secara otomatis oleh MLflow Autolog.")

if __name__ == '__main__':
    train_model()