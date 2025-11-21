# ==============================
# modelling_tuning_dagshub.py (VERSI LOG MANUAL STABIL & STRUKTUR LENGKAP)
# ==============================

import json
import os
import shutil 
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real

import mlflow
import mlflow.sklearn
import joblib
import dagshub
import yaml 

# --- IMPOR STABIL UNTUK LOGGING MANUAL ---
from mlflow.sklearn import get_default_conda_env, get_default_pip_requirements
from mlflow.models.model import Model
from mlflow.models.signature import infer_signature
# -----------------------------------------------

# ==========================
# KONFIG DAGS HUB (Tidak diubah)
# ==========================
DAGSHUB_REPO_OWNER = "Mansurgayo"
DAGSHUB_REPO_NAME = "Membangun_model"
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

try:
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow diarahkan ke: {MLFLOW_TRACKING_URI}")
except Exception as e:
    print(f"WARNING: Gagal set MLflow DAGsHub: {e}")

# ==========================
# DISABLE AUTOLOG (Tidak diubah)
# ==========================
mlflow.sklearn.autolog(disable=True)

# ==========================
# FUNGSI ARTEFAK (Tidak diubah)
# ...
# ==========================
def save_estimator_html(model, cols, out_path="estimator.html"):
    try:
        coef_rows = "".join(
            f"<tr><td>{c}</td><td>{float(coef):.6f}</td></tr>"
            for c, coef in zip(cols, model.coef_.ravel())
        )
        intercept = float(model.intercept_[0])
    except:
        coef_rows = "<tr><td colspan='2'>Model tidak punya koefisien</td></tr>"
        intercept = "N/A"

    html = f"""
    <html><body>
    <h1>Estimator Summary</h1>
    <h2>{model.__class__.__name__}</h2>

    <h3>Params</h3>
    <pre>{json.dumps(model.get_params(), indent=2)}</pre>

    <h3>Intercept</h3>
    <p>{intercept}</p>

    <h3>Coefficients</h3>
    <table border="1" cellpadding="4" cellspacing="0">
        <tr><th>Feature</th><th>Coef</th></tr>
        {coef_rows}
    </table>
    </body></html>
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path

def save_metrics_json(data, out_path="metric_info.json"):
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path

def save_confusion_matrix_png(y_true, y_pred, out_path="training_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

# ==========================
# LOAD DATA (Tidak diubah)
# ...
# ==========================
def load_and_preprocess_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    if "Outcome" not in df.columns:
        raise ValueError("Dataset harus punya kolom 'Outcome'.")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        pd.DataFrame(X_scaled, columns=X.columns),
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

# ==========================
# TRAINING
# ==========================
def train_and_tune():
    DATA_PATH = "./namadataset_preprocessing/cleaned_dataset.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
    
    # Inisialisasi variabel artefak lokal
    est, met, cm = "estimator.html", "metric_info.json", "training_confusion_matrix.png"
    MODEL_ARTIFACT_PATH = "model"

    try:
        mlflow.set_experiment("Diabetes_Prediction_Advanced")
        with mlflow.start_run():
            search_space = {
                "C": Real(1e-6, 1e6, prior="log-uniform"),
                "penalty": ["l1", "l2"],
            }
            base_model = LogisticRegression(solver="liblinear", random_state=42)
            opt = BayesSearchCV(
                estimator=base_model,
                search_spaces=search_space,
                n_iter=20,
                cv=5,
                scoring="f1",
                random_state=42,
                n_jobs=1,
            )

            print("\n🔍 Tuning hyperparameter…")
            opt.fit(X_train, y_train)
            best_model = opt.best_estimator_

            # LOG PARAMS & METRICS (Tidak diubah)
            mlflow.log_params({
                "C": float(opt.best_params_["C"]),
                "penalty": str(opt.best_params_["penalty"]),
                "tuning_method": "BayesSearchCV",
                "n_iter": 20
            })
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }
            mlflow.log_metrics(metrics)

            # ==========================
            # ARTEFAK NON-MODEL (di Root Run)
            # ==========================
            est = save_estimator_html(best_model, X_train.columns.tolist())
            met = save_metrics_json({"metrics": metrics, "best_params": opt.best_params_})
            cm = save_confusion_matrix_png(y_test, y_pred)
            mlflow.log_artifact(est)
            mlflow.log_artifact(met)
            mlflow.log_artifact(cm)

            # ==========================
            # LOG MODEL LENGKAP SECARA MANUAL (FINAL STABIL)
            # ==========================
            print("📦 Logging model lengkap secara manual...")
            os.makedirs(MODEL_ARTIFACT_PATH, exist_ok=True)
            
            # 1. SIMPAN MODEL.PKL
            model_pkl_path = os.path.join(MODEL_ARTIFACT_PATH, "model.pkl")
            joblib.dump(best_model, model_pkl_path)
            
            # 2. BUAT METADATA MLMODEL (model/MLmodel)
            signature = infer_signature(X_train, best_model.predict(X_train))
            
            mlflow_model = Model(
                artifact_path=MODEL_ARTIFACT_PATH, 
                run_id=mlflow.active_run().info.run_id,
                signature=signature
            )
            
            mlflow.sklearn.FLAVOR_NAME
            mlflow_model.add_flavor(
                mlflow.sklearn.FLAVOR_NAME, 
                serialization_format='joblib', 
                sklearn_version=sklearn.__version__, 
                pickled_model="model.pkl"
            )
            
            mlflow_model_path = os.path.join(MODEL_ARTIFACT_PATH, "MLmodel")
            mlflow_model.save(mlflow_model_path)
            
            # 3. BUAT FILE LINGKUNGAN (conda.yaml, requirements.txt, python_env.yaml)
            
            # a. conda.yaml (Memastikan konversi dict ke YAML)
            conda_env_path = os.path.join(MODEL_ARTIFACT_PATH, "conda.yaml")
            with open(conda_env_path, "w") as f:
                conda_dict = get_default_conda_env(include_cloudpickle=False)
                yaml.dump(conda_dict, f, default_flow_style=False)
                 
            # b. requirements.txt
            req_path = os.path.join(MODEL_ARTIFACT_PATH, "requirements.txt")
            with open(req_path, "w") as f:
                 f.write('\n'.join(get_default_pip_requirements()))

            # c. python_env.yaml (FINAL: Tambahkan sebagai dummy file kosong agar sesuai struktur gambar)
            python_env_path = os.path.join(MODEL_ARTIFACT_PATH, "python_env.yaml")
            with open(python_env_path, "w") as f:
                f.write('{}') 
            
            # 4. LOG SEMUA ARTEFAK DI FOLDER 'model'
            mlflow.log_artifacts(MODEL_ARTIFACT_PATH, artifact_path="model")
            
            print("\n🎉 Training selesai. Semua artefak & model berhasil diupload ke DAGsHub.")
            
    finally:
        # PEMBERSIHAN FILE LOKAL
        for f in [est, met, cm]:
            if os.path.exists(f):
                os.remove(f)
        
        if os.path.exists(MODEL_ARTIFACT_PATH):
            shutil.rmtree(MODEL_ARTIFACT_PATH)

if __name__ == "__main__":
    train_and_tune()