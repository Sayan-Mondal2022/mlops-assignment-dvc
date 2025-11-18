import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import yaml
import joblib
import json
import time
import os

def load_config():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_processed_data():
    processed_data = joblib.load('data/processed/processed_data.pkl')
    return processed_data

def get_model(model_name, config):
    """Factory function to create model instances"""
    
    model_config = config['model'][model_name]
    
    if model_name == "random_forest":
        return RandomForestRegressor(**model_config)
    
    elif model_name == "xgboost":
        return xgb.XGBRegressor(**model_config)
    
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor(**model_config)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model():
    config = load_config()
    processed_data = load_processed_data()
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    model_name = config['model']['name']
    
    # Get model instance
    print(f"🚀 Training {model_name}...")
    model = get_model(model_name, config)
    
    # Train model with timing
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    
    # Handle MAPE calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
    mape = float(mape)
    
    # Save metrics - ONLY for the current model (as per dvc.yaml)
    metrics = {
        'model': model_name,
        'training_time_seconds': float(training_time),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'parameters': config['model'][model_name]
    }
    
    # Save model-specific metrics (as per dvc.yaml: metrics/training/${model.name}_metrics.json)
    os.makedirs('metrics/training', exist_ok=True)
    metrics_file = f'metrics/training/{model_name}_metrics.json'
    
    # Load existing metrics if file exists, otherwise create new
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            try:
                existing_metrics = json.load(f)
                # If it's a list, append; if single dict, convert to list
                if isinstance(existing_metrics, list):
                    existing_metrics.append(metrics)
                    all_metrics = existing_metrics
                else:
                    all_metrics = [existing_metrics, metrics]
            except json.JSONDecodeError:
                all_metrics = [metrics]
    else:
        all_metrics = [metrics]
    
    # Keep only last 5 runs to avoid file getting too large
    if len(all_metrics) > 5:
        all_metrics = all_metrics[-5:]
    
    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save model - ONLY the current model (as per dvc.yaml: models/${model.name}_model.pkl)
    os.makedirs('models', exist_ok=True)
    model_filename = f'models/{model_name}_model.pkl'
    joblib.dump(model, model_filename)
    
    # Feature importance - save with model name
    if hasattr(model, 'feature_importances_'):
        feature_importance = {}
        for feature, importance in zip(X_train.columns, model.feature_importances_):
            feature_importance[feature] = float(importance)
        
        os.makedirs('metrics/feature_importance', exist_ok=True)
        with open(f'metrics/feature_importance/{model_name}_feature_importance.json', 'w') as f:
            json.dump(feature_importance, f, indent=2)
        
        print("📊 Feature Importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"   {feature}: {importance:.4f}")
    
    print(f"✅ {model_name.upper()} trained successfully!")
    print(f"📊 Metrics: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    print(f"💾 Model saved: {model_filename}")
    print(f"📈 Metrics saved: {metrics_file}")
    
    return model, metrics

if __name__ == "__main__":
    train_model()