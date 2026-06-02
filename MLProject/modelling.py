"""
Modelling.py - Model Machine Learning Dasar untuk Workflow CI
Dataset: Insurance Premium Prediction
"""

import pandas as pd
import os
import sys
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings('ignore')

# Memaksa UTF-8 output untuk mengatasi error emoji di Windows (saat dijalankan lokal)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==================== KONFIGURASI MLFLOW LOKAL ====================
# Mengatur MLflow agar menyimpan log eksperimen secara lokal ke folder 'mlruns'
# Ini mencegah masalah otorisasi DagsHub di GitHub Actions (mengikuti cara rekan Anda)
mlflow.set_tracking_uri("mlruns")

# ==================== KONFIGURASI DATA ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'insurance_preprocessing', 'insurance_preprocessed.csv')
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ==================== LOAD DATA ====================
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data shape: {df.shape}")
    return df

# ==================== PREPARE DATA ====================
def prepare_data(df):
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test

# ==================== EVALUATE MODEL ====================
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

# ==================== TRAIN AND LOG RANDOM FOREST ====================
def train_random_forest(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name='RandomForest_Baseline'):
        print("\nMemulai pelatihan model RandomForest...")
        
        n_estimators = 100
        max_depth = 15
        min_samples_split = 5
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_metrics = evaluate_model(y_train, y_train_pred)
        test_metrics = evaluate_model(y_test, y_test_pred)
        
        print(f"\nPelatihan selesai! R2 Score (Test): {test_metrics['r2']:.4f}")
        return model, y_test_pred

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Set experiment
    mlflow.set_experiment("insurance_prediction_baseline")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    # Load data
    df = load_data(PREPROCESSED_DATA_PATH)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train model
    rf_model, rf_predictions = train_random_forest(X_train, X_test, y_train, y_test)