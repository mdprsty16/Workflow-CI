"""
Modelling.py - Model Machine Learning Dasar dengan MLflow Manual Logging
Dataset: Insurance Premium Prediction
Target: Prediksi charges (biaya asuransi)
"""

import pandas as pd
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
import mlflow
import mlflow.sklearn
import dagshub
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI DAGSUB ====================
# Initialize DagsHub (satu baris setup MLflow + authentication)
dagshub.init(repo_owner="mdprsty16", repo_name="Eksperimen_SML_M-Deco-Prasetyo", mlflow=True)

# ==================== KONFIGURASI DATA ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'insurance_preprocessing', 'insurance_preprocessed.csv')
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ==================== LOAD DATA ====================
def load_data(filepath):
    """Load preprocessed dataset"""
    df = pd.read_csv(filepath)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

# ==================== PREPARE DATA ====================
def prepare_data(df):
    """Prepare features and target"""
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# ==================== EVALUATE MODEL ====================
def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }



# ==================== TRAIN AND LOG RANDOM FOREST ====================
def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with MLflow logging"""
    
    with mlflow.start_run(run_name='RandomForest_Baseline'):
        print("\n" + "="*50)
        print("Training Random Forest Model")
        print("="*50)
        
        # Model parameters
        n_estimators = 100
        max_depth = 15
        min_samples_split = 5
        
        # Create and train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate
        train_metrics = evaluate_model(y_train, y_train_pred)
        test_metrics = evaluate_model(y_test, y_test_pred)
        
        print(f"\nTrain Metrics:")
        print(f"  RMSE: {train_metrics['rmse']:.4f}")
        print(f"  MAE: {train_metrics['mae']:.4f}")
        print(f"  R2 Score: {train_metrics['r2']:.4f}")
        
        print(f"\nTest Metrics:")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  R2 Score: {test_metrics['r2']:.4f}")
        
        return model, y_test_pred

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("="*70)
    print("INSURANCE PREMIUM PREDICTION - MODEL TRAINING (BASELINE)")
    print("="*70)
    
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
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("Check MLflow at DagsHub")
    print("="*70)
