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

def evaluate_single_model():
    """Evaluate the currently selected model"""
    config = load_config()
    
    model_name = config['model']['name']
    model_path = f'models/{model_name}_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Load model and data
    model = joblib.load(model_path)
    processed_data = joblib.load('data/processed/processed_data.pkl')
    
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'model': model_name,
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'mse': float(mean_squared_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': float(r2_score(y_test, y_pred))
    }
    
    # Ensure the evaluation directory exists
    os.makedirs('metrics/evaluation', exist_ok=True)
    
    # Save evaluation metrics
    evaluation_file = f'metrics/evaluation/{model_name}_evaluation.json'
    with open(evaluation_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create plots directory
    os.makedirs(f'metrics/plots/{model_name}', exist_ok=True)
    
    # Create all plots
    create_plots(y_test, y_pred, X_test, model_name, model)
    
    print(f"✅ {model_name.upper()} evaluation completed!")
    print(f"📊 Metrics saved to: {evaluation_file}")
    
    return metrics

def create_plots(y_test, y_pred, X_test, model_name, model):
    """Create evaluation plots including feature importance"""
    
    # 1. Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual CPU Usage')
    plt.ylabel('Predicted CPU Usage')
    plt.title(f'{model_name.upper()} - Actual vs Predicted CPU Usage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'metrics/plots/{model_name}/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='green')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name.upper()} - Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'metrics/plots/{model_name}/residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance plot
    create_feature_importance_plot(model, X_test, model_name)
    
    # 4. Prediction Distribution plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_test, bins=30, alpha=0.7, color='blue', label='Actual')
    plt.xlabel('CPU Usage')
    plt.ylabel('Frequency')
    plt.title('Actual CPU Usage Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(y_pred, bins=30, alpha=0.7, color='red', label='Predicted')
    plt.xlabel('CPU Usage')
    plt.ylabel('Frequency')
    plt.title('Predicted CPU Usage Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'metrics/plots/{model_name}/distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def create_feature_importance_plot(model, X_test, model_name):
    """Create feature importance plot for the model"""
    
    # For Linear Regression, show coefficients with signs
    if model_name == 'linear_regression' and hasattr(model, 'coef_'):
        coefficients = model.coef_
        feature_names = X_test.columns
        
        # Create DataFrame for sorting by absolute value but keeping sign
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_value': np.abs(coefficients)
        }).sort_values('abs_value', ascending=True)
        
        # Plot coefficients with colors based on sign
        plt.figure(figsize=(12, 8))
        
        colors = ['red' if x < 0 else 'green' for x in coef_df['coefficient']]
        bars = plt.barh(coef_df['feature'], coef_df['coefficient'], 
                color=colors, edgecolor='black', alpha=0.7)
        
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'{model_name.upper()} - Feature Coefficients\n(Green=Positive, Red=Negative)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        
        # Add value labels on bars
        for i, v in enumerate(coef_df['coefficient']):
            plt.text(v + (0.001 if v >= 0 else -0.01), i, f'{v:.4f}', 
                    va='center', fontsize=9, 
                    color='white' if abs(v) > np.max(np.abs(coef_df['coefficient'])) * 0.3 else 'black')
        
        plt.tight_layout()
        plt.savefig(f'metrics/plots/{model_name}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature coefficients plot created for {model_name}")
        
        # Save coefficients data to JSON
        coef_dict = coef_df.set_index('feature')['coefficient'].to_dict()
        os.makedirs('metrics/feature_importance', exist_ok=True)
        with open(f'metrics/feature_importance/{model_name}_feature_importance.json', 'w') as f:
            json.dump(coef_dict, f, indent=2)
    
    # For tree-based models with feature_importances_
    elif hasattr(model, 'feature_importances_'):
        # Get feature importance
        feature_importance = model.feature_importances_
        feature_names = X_test.columns
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar plot
        plt.barh(importance_df['feature'], importance_df['importance'], 
                color='skyblue', edgecolor='black', alpha=0.7)
        
        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'{model_name.upper()} - Feature Importance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, v in enumerate(importance_df['importance']):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'metrics/plots/{model_name}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature importance plot created for {model_name}")
        
        # Save feature importance data to JSON
        importance_dict = importance_df.set_index('feature')['importance'].to_dict()
        os.makedirs('metrics/feature_importance', exist_ok=True)
        with open(f'metrics/feature_importance/{model_name}_feature_importance.json', 'w') as f:
            json.dump(importance_dict, f, indent=2)
            
    else:
        print(f"⚠️  {model_name} doesn't have feature_importances_ or coefficients attribute")
        
        # For models without feature_importance, create a placeholder message
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'{model_name.upper()}\nNo feature importance available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title(f'{model_name.upper()} - Feature Importance Not Available')
        plt.axis('off')
        plt.savefig(f'metrics/plots/{model_name}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

def evaluate_all_models():
    """Evaluate all available models and create comparison"""
    models_to_evaluate = ['linear_regression', 'xgboost', 'lightgbm', 'random_forest']
    all_metrics = {}
    
    print("🧪 Evaluating all models...")
    print("="*60)
    
    for model_name in models_to_evaluate:
        model_path = f'models/{model_name}_model.pkl'
        if os.path.exists(model_path):
            print(f"\nEvaluating {model_name}...")
            
            # Temporarily change the model in params for evaluation
            with open('params.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            original_model = config['model']['name']
            config['model']['name'] = model_name
            
            with open('params.yaml', 'w') as f:
                yaml.dump(config, f)
            
            # Evaluate this model
            metrics = evaluate_single_model()
            if metrics:
                all_metrics[model_name] = metrics
            
            # Restore original model
            config['model']['name'] = original_model
            with open('params.yaml', 'w') as f:
                yaml.dump(config, f)
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    # Create comparison report if we have multiple models
    if len(all_metrics) > 1:
        create_comparison_report(all_metrics)
    
    return all_metrics

    
def create_comparison_report(all_metrics):
    """Create a comparison report for all models"""
    metrics_df = pd.DataFrame(all_metrics).T
    
    # Save comparison data
    os.makedirs('metrics/comparison', exist_ok=True)
    with open('metrics/comparison/models_comparison.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    metrics_to_plot = ['r2', 'rmse', 'mae']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 2, i)
        if metric == 'r2':
            # Higher R² is better
            bars = plt.bar(metrics_df.index, metrics_df[metric], color=colors, alpha=0.7, edgecolor='black')
            plt.title(f'{metric.upper()} Comparison\n(Higher is better)', fontweight='bold')
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
        else:
            # Lower RMSE/MAE is better
            bars = plt.bar(metrics_df.index, metrics_df[metric], color=colors, alpha=0.7, edgecolor='black')
            plt.title(f'{metric.upper()} Comparison\n(Lower is better)', fontweight='bold')
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
        
        plt.ylabel(metric.upper(), fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics/comparison/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n📊 MODEL COMPARISON SUMMARY:")
    print("="*60)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name.upper():12} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

if __name__ == "__main__":
    # Evaluate current model
    metrics = evaluate_single_model()
    
    if metrics:
        print("✅ Evaluation completed!")
        print(f"📊 Final Metrics: {metrics}")
        
        # Also evaluate all models for comparison
        print("\n" + "="*60)
        evaluate_all_models()
    else:
        print("❌ Evaluation failed!")