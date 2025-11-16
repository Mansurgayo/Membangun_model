import pandas as pd
import numpy as np
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real

# ======================
# KONFIGURASI DAGSHUB
# ======================
DAGSHUB_REPO_OWNER = "Mansurgayo"
DAGSHUB_REPO_NAME = "Membangun_model"

MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

try:
    # Tidak membuat repo baru
    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow diarahkan ke: {MLFLOW_TRACKING_URI}")

except Exception as e:
    print(f"⚠️ ERROR DAGSHUB: {e}")
    print("Pastikan kamu sudah login dengan: dagshub login")


# ======================
# PREPROCESSING DATA
# ======================
def load_and_preprocess_data(file_path: str):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan → {file_path}")
        return None, None, None, None

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ======================
# TRAINING + HYPERTUNING
# ======================
def train_and_tune_model():

    DATA_PATH = "./namadataset_preprocessing/cleaned_dataset.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    if X_train is None:
        return

    mlflow.set_experiment("Diabetes_Prediction_Advanced")

    with mlflow.start_run():
        print("\n===== MLflow Run Dimulai =====")

        # ----------------------------
        # BAYESIAN OPTIMIZATION
        # ----------------------------
        search_space = {
            "C": Real(1e-6, 1e+6, prior="log-uniform"),
            "penalty": ["l1", "l2"]
        }

        model = LogisticRegression(
            solver="liblinear",
            random_state=42
        )

        opt = BayesSearchCV(
            model,
            search_space,
            n_iter=20,
            cv=5,
            scoring="f1",
            random_state=42
        )

        print("🔍 Hyperparameter tuning berjalan...")
        opt.fit(X_train, y_train)

        best_model = opt.best_estimator_
        print(f"🎯 Best Params: {opt.best_params_}")

        mlflow.log_params(opt.best_params_)
        mlflow.log_param("tuning_method", "BayesSearchCV")

        # ----------------------------
        # METRIC
        # ----------------------------
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # METRIK TAMBAHAN
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")
        mean_cv_accuracy = np.mean(cv_scores)

        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_roc_auc": roc_auc,
            "mean_cv_accuracy": mean_cv_accuracy,
            "num_features": X_train.shape[1]
        })

        # ----------------------------
        # LOG MODEL
        # ----------------------------
        mlflow.sklearn.log_model(best_model, "best_logistic_regression_model")

        print("\n===== LOGGING SELESAI =====")
        print("Akurasi:", accuracy)
        print("ROC AUC:", roc_auc)
        print("CV Mean Accuracy:", mean_cv_accuracy)
        print("Model berhasil dikirim ke DagsHub! 🚀")


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    train_and_tune_model()
