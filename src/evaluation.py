import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import yaml
import os

def load_config():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate_model():
    config = load_config()
    
    # Load model and data
    model_name = config['model']['name']
    model = joblib.load(f'models/{model_name}_model.pkl')
    processed_data = joblib.load('data/processed/processed_data.pkl')
    
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'mse': float(mean_squared_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': float(r2_score(y_test, y_pred))
    }
    
    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create plots directory
    os.makedirs('metrics/plots', exist_ok=True)
    
    # Create visualization plots
    create_plots(y_test, y_pred, X_test, config)
    
    return metrics

def create_plots(y_test, y_pred, X_test, config):
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual CPU Usage')
    plt.ylabel('Predicted CPU Usage')
    plt.title('Actual vs Predicted CPU Usage')
    plt.savefig('metrics/plots/actual_vs_predicted.png')
    plt.close()
    
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('metrics/plots/residual_plot.png')
    plt.close()
    
    # Feature importance plot (if available)
    try:
        model = joblib.load(f"models/{config['model']['name']}_model.pkl")
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance Plot')
            plt.tight_layout()
            plt.savefig('metrics/plots/feature_importance.png')
            plt.close()
    except:
        pass

if __name__ == "__main__":
    metrics = evaluate_model()
    print("Evaluation completed!")
    print(f"Final Metrics: {metrics}")