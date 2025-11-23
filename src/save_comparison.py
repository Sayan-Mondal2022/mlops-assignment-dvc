import json
import pandas as pd
from datetime import datetime

def save_final_comparison():
    """Save the final comparison results"""
    
    # Load all individual model metrics
    models = ['xgboost', 'linear_regression', 'random_forest']
    all_metrics = {}
    
    for model in models:
        try:
            with open(f'metrics/evaluation/{model}_evaluation.json', 'r') as f:
                all_metrics[model] = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  {model} metrics not found")
    
    if all_metrics:
        # Save detailed comparison
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'models': all_metrics,
            'summary': {
                'best_r2': max(all_metrics.items(), key=lambda x: x[1]['r2']),
                'best_rmse': min(all_metrics.items(), key=lambda x: x[1]['rmse']),
                'best_mae': min(all_metrics.items(), key=lambda x: x[1]['mae'])
            }
        }
        
        # Ensure directory exists
        import os
        os.makedirs('metrics/comparison', exist_ok=True)
        
        # Save JSON
        with open('metrics/comparison/final_model_comparison.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Save CSV for easy viewing
        df = pd.DataFrame(all_metrics).T
        df.to_csv('metrics/comparison/final_model_comparison.csv')
        
        print("✅ Final comparison saved!")
        print("📁 metrics/comparison/final_model_comparison.json")
        print("📁 metrics/comparison/final_model_comparison.csv")
        
        # Print summary
        print("\n🎯 FINAL RANKINGS:")
        print("="*60)
        print(f"🏆 Best R²:    {comparison_data['summary']['best_r2'][0]} ({comparison_data['summary']['best_r2'][1]['r2']:.4f})")
        print(f"🏆 Best RMSE:  {comparison_data['summary']['best_rmse'][0]} ({comparison_data['summary']['best_rmse'][1]['rmse']:.4f})")
        print(f"🏆 Best MAE:   {comparison_data['summary']['best_mae'][0]} ({comparison_data['summary']['best_mae'][1]['mae']:.4f})")
        
        return comparison_data
    else:
        print("❌ No metrics found to compare")
        return None

if __name__ == "__main__":
    save_final_comparison()