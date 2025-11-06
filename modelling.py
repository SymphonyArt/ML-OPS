import mlflow
import mlflow.lightgbm
import pandas as pd
import lightgbm as lgb
import sklearn
import os
import json
import matplotlib.pyplot as plt 

from sklearn.utils import estimator_html_repr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay  
)
from mlflow.models import infer_signature
from sklearn.model_selection import ParameterGrid, cross_val_score

# Variabel global untuk data
X_train, y_train, X_test, y_test = None, None, None, None

def load_data():
    """Memuat data ke variabel global."""
    global X_train, y_train, X_test, y_test
    print("Loading data from CSV files...")
    
    # Ganti nama file jika perlu
    X_TRAIN_FILE = "X_train_res.csv"
    Y_TRAIN_FILE = "y_train_res.csv"
    X_TEST_FILE = "X_test.csv"
    Y_TEST_FILE = "y_test.csv"

    try:
        X_train = pd.read_csv(X_TRAIN_FILE)
        y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()
        X_test = pd.read_csv(X_TEST_FILE)
        y_test = pd.read_csv(Y_TEST_FILE).squeeze()
        print("Data loaded successfully.")
        return True
    except FileNotFoundError as e:
        print(f"Error: File not found. Pastikan file CSV ada di direktori yang sama.")
        print(e)
        return False

def run_training():
    
    # 1a. Mengatur MLflow (Selalu melacak ke folder 'mlruns' lokal)
    mlruns_path = os.path.abspath("mlruns")
    os.makedirs(mlruns_path, exist_ok=True)
    mlflow.set_tracking_uri("file:///" + mlruns_path.replace("\\", "/"))
    
    experiment_name = "LGBM_Manual" 
    mlflow.set_experiment(experiment_name)

    # 1b. Muat data
    if not load_data():
        return 
    
    # 1c. Tentukan Hyperparameter Grid
    param_grid_dict = {
       'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1,0.2],
        'max_depth': [10, 20, -1, None],
        'min_child_samples': [10, 20],
        'feature_fraction': [0.8, 1.0]
    }
    
    grid = ParameterGrid(param_grid_dict)
    print(f"Total parameter combinations to test: {len(grid)}")

    # 1d. Lakukan Grid Search Manual (in-memory, tanpa MLflow)
    print("Starting Manual Grid Search (in-memory)...")
    
    best_score = -1
    best_params = None
    
    for i, params in enumerate(grid):
        print(f"\nStarting Trial {i+1}/{len(grid)} - Params: {params}")
        
        model = lgb.LGBMClassifier(
            objective='binary', 
            random_state=42,
            **params
        )
        
        try:
            score = cross_val_score(
                model,
                X_train,
                y_train,
                n_jobs=-1,
                cv=3,
                scoring='f1_weighted'
            ).mean()
            
            print(f"Trial {i+1} finished. Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = params
                
        except Exception as e:
            print(f"Trial {i+1} failed with params {params}. Error: {e}")
            
    print("\nManual Grid Search finished.")
    print(f"Best CV F1-Score: {best_score:.4f}")
    print(f"Best parameters found: {best_params}")

    # 2. Log Model TERBAIK (Satu Run Saja)
    if best_params is None:
        print("No successful trials. Exiting before logging best model.")
        return

    print("\nLogging best model and final artifacts to MLflow...")
    
    with mlflow.start_run(run_name="Best_Model") as best_run:
        
        # 2a. Log Parameter & Metrik Terbaik
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1_weighted", best_score)
        
        # 2b. Latih ulang model terbaik
        best_model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            **best_params
        )
        best_model.fit(X_train, y_train)
        
        # 2c. Evaluasi pada data Tes
        y_preds = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds, average='weighted')
        report_str = classification_report(y_test, y_preds)
        cm = confusion_matrix(y_test, y_preds)

        # 2d. Log Metrik Tes
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_weighted", f1)
        
        # 2e. Log Artefak (Laporan Teks)
        mlflow.log_text(report_str, "classification_report.txt")
        
        # 2f. Log Artefak (Confusion Matrix)
        print("Logging confusion matrix...")
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                          display_labels=best_model.classes_)
            fig, ax = plt.subplots()
            disp.plot(ax=ax, cmap="Blues")
            plt.title("Confusion Matrix")
            plt.savefig("confusion_matrix.png")
            plt.close(fig)
            mlflow.log_artifact("confusion_matrix.png")
            os.remove("confusion_matrix.png")
            print("Confusion matrix logged.")
        except Exception as e:
            print(f"Error logging confusion matrix: {e}")

        # 2g. Log Model (PENTING UNTUK DOCKER BUILD)
        print("Logging model for Docker build...")
        signature = infer_signature(X_train, y_preds)
        input_example = X_train.head(5)
    
        mlflow.lightgbm.log_model(
            lgb_model=best_model,
            artifact_path="lgbm_tuning", # <-- Nama ini dicari oleh file YAML
            signature=signature,
            input_example=input_example,
        )
        print("Model logged successfully.")

    print(f"\nFinished. Cek eksperimen '{experiment_name}' di folder 'mlruns'.")

if __name__ == "__main__":
    run_training()