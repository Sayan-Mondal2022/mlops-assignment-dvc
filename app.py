import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set page configuration
st.set_page_config(
    page_title="CPU Usage Predictor",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for blue theme styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        padding: 0.5rem;
        border-radius: 10px;
        border: None;
        margin: 1rem 0;
        background: linear-gradient(135deg, #1f77b4 0%, #2c90c9 100%);
        color: white;
        text-align: center;
    }
    .model-result {
        background-color: white;
        color: #333;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .range-indicator {
        background: linear-gradient(135deg, #2c90c9 0%, #4ca8e0 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton button {
        background: linear-gradient(135deg, #1f77b4 0%, #2c90c9 100%);
        color: white;
        border: none;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #1668a1 0%, #1f77b4 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class CPUPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
        
    def load_models(self):
        """Load all trained models and preprocessing objects"""
        try:
            # Load preprocessing objects
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            
            # Load models
            model_files = {
                'XGBoost': 'models/xgboost_model.pkl',
                'LightGBM': 'models/lightgbm_model.pkl', 
                'Random Forest': 'models/random_forest_model.pkl'
            }
            
            for name, path in model_files.items():
                if os.path.exists(path):
                    self.models[name] = joblib.load(path)
            
            return len(self.models) > 0
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess user input for prediction"""
        try:
            # Create DataFrame
            df = pd.DataFrame([input_data])
            
            # Encode controller_kind
            if 'controller_kind' in df.columns:
                df['controller_kind'] = self.label_encoder.transform(df['controller_kind'])
            
            # Scale numerical features
            numerical_cols = [col for col in self.feature_names if col != 'controller_kind']
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error in preprocessing: {str(e)}")
            return None
    
    def predict(self, input_data):
        """Make predictions using all loaded models"""
        predictions = {}
        
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        if processed_data is None:
            return None
        
        # Make predictions with each model
        for model_name, model in self.models.items():
            try:
                pred = model.predict(processed_data)[0]
                predictions[model_name] = max(0, float(pred))  # Ensure non-negative
            except Exception as e:
                predictions[model_name] = None
        
        return predictions
    
    def load_model_metrics(self):
        """Load evaluation metrics for each model"""
        metrics = {}
        metric_files = {
            'XGBoost': 'metrics/evaluation/xgboost_evaluation.json',
            'LightGBM': 'metrics/evaluation/lightgbm_evaluation.json',
            'Random Forest': 'metrics/evaluation/random_forest_evaluation.json'
        }
        
        for name, path in metric_files.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        metrics[name] = json.load(f)
                except:
                    metrics[name] = None
            else:
                metrics[name] = None
        
        return metrics

def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ CPU Usage Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Quick CPU Usage Estimation")
    
    # Initialize predictor
    predictor = CPUPredictor()
    
    # Load models
    with st.spinner("Loading AI models..."):
        models_loaded = predictor.load_models()
    
    if not models_loaded:
        st.error("❌ Models not loaded. Please train models first.")
        return
    
    # Input form
    st.header("🔧 Enter Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_request = st.number_input("CPU Request (cores)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
        mem_request = st.number_input("Memory Request (GB)", min_value=0.1, max_value=1000.0, value=4.0, step=0.1)
        cpu_limit = st.number_input("CPU Limit (cores)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    
    with col2:
        mem_limit = st.number_input("Memory Limit (GB)", min_value=0.1, max_value=1000.0, value=8.0, step=0.1)
        runtime_minutes = st.number_input("Runtime (minutes)", min_value=1, max_value=10080, value=60, step=1)
        controller_kind = st.selectbox("Controller Kind", 
                                     options=predictor.label_encoder.classes_.tolist())

    # Prediction button
    if st.button("🚀 Predict CPU Usage", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'cpu_request': cpu_request,
            'mem_request': mem_request,
            'cpu_limit': cpu_limit,
            'mem_limit': mem_limit,
            'runtime_minutes': runtime_minutes,
            'controller_kind': controller_kind
        }
        
        # Make predictions
        with st.spinner("Analyzing with AI models..."):
            predictions = predictor.predict(input_data)
        
        if predictions:
            # Filter out None predictions
            valid_predictions = {k: v for k, v in predictions.items() if v is not None}
            
            if valid_predictions:
                # Calculate statistics
                min_pred = min(valid_predictions.values())
                max_pred = max(valid_predictions.values())
                avg_pred = np.mean(list(valid_predictions.values()))
                
                # Display main prediction
                st.markdown("---")
                st.header("🎯 Prediction Results")
                
                # Average prediction card
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>Estimated CPU Usage</h2>
                    <h1>{avg_pred:.4f}</h1>
                    <p>Based on {len(valid_predictions)} AI models</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Prediction range
                st.markdown(f"""
                <div class="range-indicator">
                    <h3>📊 Prediction Range</h3>
                    <p><strong>Minimum:</strong> {min_pred:.4f} | <strong>Maximum:</strong> {max_pred:.4f}</p>
                    <p>All models agree on a range between {min_pred:.4f} and {max_pred:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual model results
                st.subheader("🤖 Individual Model Predictions")
                for model_name, prediction in valid_predictions.items():
                    st.markdown(f"""
                    <div class="model-result">
                        <strong>{model_name}:</strong> {prediction:.4f}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show detailed analysis button
                st.subheader("📈 Show Detailed Analysis & Graphs")
                show_detailed_analysis(valid_predictions, predictor)
                
                # Input summary (collapsible)
                with st.expander("📋 View Input Summary"):
                    st.json(input_data)

def show_detailed_analysis(predictions, predictor):
    """Show detailed analysis and graphs"""
    st.markdown("---")
    st.header("📊 Detailed Analysis")
    
    # Model comparison chart
    st.subheader("Model Predictions Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(predictions.keys())
    values = list(predictions.values())
    
    # Blue color palette
    blue_colors = ['#1f77b4', '#2c90c9', '#4ca8e0']
    
    bars = ax.bar(models, values, color=blue_colors, alpha=0.7)
    ax.set_ylabel('Predicted CPU Usage')
    ax.set_title('Model Predictions Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    
    # Performance metrics
    st.subheader("🎯 Model Performance Metrics")
    metrics = predictor.load_model_metrics()
    
    if any(metrics.values()):
        # Create performance table
        perf_data = []
        for model_name, metric_data in metrics.items():
            if metric_data and model_name in predictions:
                perf_data.append({
                    'Model': model_name,
                    'R² Score': metric_data.get('r2', 0),
                    'RMSE': metric_data.get('rmse', 0),
                    'MAE': metric_data.get('mae', 0),
                    'Current Prediction': predictions[model_name]
                })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df.style.format({
                'R² Score': '{:.4f}',
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}',
                'Current Prediction': '{:.4f}'
            }).background_gradient(subset=['R² Score'], cmap='Blues'), 
            use_container_width=True)
    
    # Feature importance (if available)
    st.subheader("🔍 Feature Importance")
    
    # Try to load feature importance for the best model
    best_model = max(predictions.items(), key=lambda x: x[1])[0]
    feature_importance_file = f'metrics/feature_importance/{best_model.lower().replace(" ", "_")}_feature_importance.json'
    
    if os.path.exists(feature_importance_file):
        try:
            with open(feature_importance_file, 'r') as f:
                feature_importance = json.load(f)
            
            # Create feature importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())
            
            # Sort by importance
            sorted_idx = np.argsort(importance)
            features = [features[i] for i in sorted_idx]
            importance = [importance[i] for i in sorted_idx]
            
            bars = ax.barh(features, importance, color='#1f77b4', alpha=0.7)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Feature Importance - {best_model}')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, imp in zip(bars, importance):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{imp:.4f}', va='center', fontsize=9)
            
            st.pyplot(fig)
            
        except Exception as e:
            st.info("Feature importance data not available for detailed analysis")
    else:
        st.info("Feature importance analysis not available")

if __name__ == "__main__":
    main()