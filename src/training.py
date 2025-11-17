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

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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
    
    # Calculate metrics - convert to Python native types
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    
    # Handle MAPE calculation carefully to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
    mape = float(mape)
    
    # Save metrics
    metrics = {
        'model': model_name,
        'training_time_seconds': float(training_time),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    # Ensure metrics directory exists
    import os
    os.makedirs('metrics', exist_ok=True)
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{model_name}_model.pkl')
    
    # Feature importance with type conversion
    if hasattr(model, 'feature_importances_'):
        feature_importance = {}
        for feature, importance in zip(X_train.columns, model.feature_importances_):
            feature_importance[feature] = float(importance)  # Convert to Python float
        
        with open('metrics/feature_importance.json', 'w') as f:
            json.dump(feature_importance, f, indent=2)
        
        print("📊 Feature Importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"   {feature}: {importance:.4f}")
    
    print(f"✅ {model_name.upper()} trained successfully!")
    print(f"📊 Metrics: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    
    return model, metrics

if __name__ == "__main__":
    train_model()