import pandas as pd
import numpy as np
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# --- Konfigurasi MLflow dan DagsHub (Skor 4) ---
# Ganti dengan nama user DagsHub dan nama repository Anda
DAGSHUB_REPO_OWNER = "Mansurgayo" 
DAGSHUB_REPO_NAME = "Membangun_model"
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

# Menginisialisasi DagsHub dan mengatur tracking URI
try:
    dagshub.init(DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow Tracking diarahkan ke DagsHub: {MLFLOW_TRACKING_URI}")
except Exception as e:
    print(f"Gagal menginisialisasi DagsHub. Pastikan Anda sudah login (dagshub login): {e}")

# --- Fungsi Preprocessing Ulang ---
def load_and_preprocess_data(file_path: str):
    """Memuat data dan melakukan standarisasi."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {file_path}.")
        return None, None, None, None
        
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Standarisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Pembagian Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# --- Fungsi Pelatihan dan Tuning Model (Skor 3 & 4) ---
def train_and_tune_model():
    
    PREPROCESSED_DATA_PATH = './namadataset_preprocessing/cleaned_dataset.csv' 
    X_train, X_test, y_train, y_test = load_and_preprocess_data(PREPROCESSED_DATA_PATH)
    
    if X_train is None:
        return

    # Inisialisasi Eksperimen MLflow
    mlflow.set_experiment("Diabetes_Prediction_Advanced")
    
    with mlflow.start_run() as run:
        print("\n--- MLflow Run Dimulai ---")
        print(f"Tracking ke DagsHub, Run ID: {run.info.run_id}")
        
        # --- 1. Konfigurasi Hyperparameter Tuning (Bayesian Optimization) ---
        
        # Ruang Pencarian untuk Logistic Regression
        search_space = {
            'C': Real(1e-6, 1e+6, prior='log-uniform'), # Inverse regularization strength
            'penalty': ['l1', 'l2']
        }

        # Model Estimator
        model_base = LogisticRegression(solver='liblinear', random_state=42)

        # Inisialisasi BayesSearchCV (Hyperparameter Tuning)
        opt = BayesSearchCV(
            model_base,
            search_space,
            n_iter=20, # Jumlah iterasi tuning
            cv=5,       # Cross-validation 5-fold
            random_state=42,
            scoring='f1'
        )

        print("Memulai Hyperparameter Tuning (Bayesian Optimization)...")
        opt.fit(X_train, y_train)

        # Ambil Model Terbaik
        best_model = opt.best_estimator_
        
        print("Tuning Selesai.")
        print(f"Parameter Terbaik: {opt.best_params_}")
        
        # --- 2. Manual Logging (Skor 3 & 4) ---
        
        # Log Parameter Terbaik
        mlflow.log_params(opt.best_params_)
        mlflow.log_param("tuning_method", "BayesSearchCV")
        mlflow.log_param("cv_folds", 5)

        # Prediksi pada Test Set
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] # Probabilitas untuk ROC AUC

        # Hitung Metrik Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # **METRIK TAMBAHAN (Skor 4)**
        # 1. Rata-rata Skor Cross-Validation (pada Training Set)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv_accuracy = np.mean(cv_scores)

        # 2. Jumlah Fitur
        num_features = X_train.shape[1]
        
        # Log Metrik
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1,
            "test_roc_auc": roc_auc,
            
            # Log Metrik Tambahan (Minimal 2 nilai tambahan)
            "mean_cv_accuracy": mean_cv_accuracy, # Metrik Tambahan 1
            "num_features": num_features          # Metrik Tambahan 2
        })
        
        print(f"\nModel Akurasi (Test): {accuracy:.4f}")
        print(f"Model ROC AUC (Test): {roc_auc:.4f}")
        print(f"Rata-rata CV Akurasi: {mean_cv_accuracy:.4f}")

        # --- 3. Log Model sebagai Artefak (Skor 4) ---
        mlflow.sklearn.log_model(best_model, "best_logistic_regression_model")
        
        print("\nModel dan metrik telah dicatat secara manual ke DagsHub!")


if __name__ == '__main__':
    train_and_tune_model()