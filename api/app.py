import streamlit as st
import pandas as pd
import json
import os
import sys

# Add the api directory to Python path so we can import predictor
sys.path.append('/app/api')
from predictor import CPUPredictor

# Helper function to get color from theme
def get_color(color_name):
    """Helper function to get color from theme"""
    colors = {
        "green": "#98c379",
        "blue": "#61afef", 
        "purple": "#c678dd",
        "orange": "#d19a66",
        "red": "#e06c75",
        "cyan": "#56b6c2",
        "bg-secondary": "#282c34",
        "text-primary": "#abb2bf"
    }
    return colors.get(color_name, "#61afef")

# Page configuration
st.set_page_config(
    page_title="CPU Usage Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for One Atom Dark Theme with Fira Code
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap');

    :root {{
        --bg-primary: #282c34;
        --bg-secondary: #3b4048;
        --bg-tertiary: #4a505c;
        --text-primary: #abb2bf;
        --text-secondary: #7f848e;
        --accent-green: {get_color("green")};
        --accent-blue: {get_color("blue")};
        --accent-purple: {get_color("purple")};
        --accent-orange: {get_color("orange")};
        --accent-red: {get_color("red")};
        --accent-cyan: {get_color("cyan")};
        --font-main: 'Fira Code', 'monospace';
        --font-code: 'Fira Code', monospace;
    }}
    
    .main, .stApp {{
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: var(--font-main);
    }}
    
    pre, code, .stCode, .stCodeBlock, 
    .stNumberInput div[data-baseweb="input"] input, 
    .stTextInput div[data-baseweb="input"] input,
    .stTextArea textarea {{
        font-family: var(--font-code) !important;
        font-weight: 500;
    }}

    .css-1d391kg, .css-1lcbmhc, .stSelectbox > div[role="button"], .st-bw {{
        background-color: var(--bg-secondary) !important;
        border-color: var(--bg-tertiary);
    }}
    
    .stNumberInput div[data-baseweb="input"], .stTextInput div[data-baseweb="input"] {{
        background-color: var(--bg-tertiary);
        color: var(--text-primary);
    }}

    h1, h2, h3, .css-10trblm {{ 
        color: var(--accent-blue);
    }}

    .stAlert {{
        border-left: 5px solid var(--accent-blue);
    }}

    .metric-card {{
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }}
    
    .best-model {{
        border: 3px solid var(--accent-green);
        box-shadow: 0 0 20px rgba(152, 195, 121, 0.5);
    }}
    
    .plot-container {{
        background: var(--bg-secondary);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# CORRECTED PATHS FOR DOCKER CONTAINER
MODELS_DIR = "/app/models"
METRICS_DIR = "/app/metrics"

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load the predictor with cached resource"""
    predictor = CPUPredictor()
    try:
        # Update predictor to use correct paths in container
        predictor.models_dir = MODELS_DIR
        models_loaded = predictor.load_models()
        if models_loaded:
            st.success("✅ Predictor initialized successfully")
            return predictor
        else:
            st.error("❌ Predictor models failed to load")
            return None
    except Exception as e:
        st.error(f"❌ Failed to load predictor: {e}")
        return None

# Load metrics helper functions with CORRECTED PATHS
def load_model_metrics():
    """Load metrics for all models"""
    # UPDATED: Replaced lightgbm with linear_regression
    metrics_paths = {
        "linear_regression": f"{METRICS_DIR}/evaluation/linear_regression_evaluation.json",
        "random_forest": f"{METRICS_DIR}/evaluation/random_forest_evaluation.json", 
        "xgboost": f"{METRICS_DIR}/evaluation/xgboost_evaluation.json"
    }
    
    metrics = {}
    for model_name, path in metrics_paths.items():
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    metrics[model_name] = json.load(f)
                st.success(f"✅ Loaded {model_name} metrics")
            else:
                st.warning(f"⚠️ Metrics file not found: {path}")
                metrics[model_name] = {}
        except Exception as e:
            st.error(f"❌ Failed to load metrics for {model_name}: {e}")
            metrics[model_name] = {}
    
    return metrics

def load_comparison_data():
    """Load model comparison data"""
    try:
        # Load CSV comparison
        csv_path = f"{METRICS_DIR}/comparison/final_model_comparison.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            comparison_data = df.to_dict('records')
            st.success("✅ Loaded comparison CSV data")
        else:
            comparison_data = []
            st.warning(f"⚠️ Comparison CSV not found: {csv_path}")
        
        # Load JSON comparison
        json_path = f"{METRICS_DIR}/comparison/final_model_comparison.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_comparison = json.load(f)
            st.success("✅ Loaded comparison JSON data")
        else:
            json_comparison = {}
            st.warning(f"⚠️ Comparison JSON not found: {json_path}")
            
        return {
            "csv_data": comparison_data,
            "json_data": json_comparison
        }
    except Exception as e:
        st.error(f"❌ Failed to load comparison data: {e}")
        return {"csv_data": [], "json_data": {}}

def get_best_model(metrics):
    """Determine the best model based on metrics"""
    if not metrics:
        return None
    
    best_model = None
    best_score = float('-inf')
    
    for model_name, model_metrics in metrics.items():
        r2_score = model_metrics.get('r2', -1)
        if r2_score > best_score:
            best_score = r2_score
            best_model = model_name
    
    return best_model

def get_plot_path(model_name, plot_type):
    """Get path for a specific plot"""
    plot_mapping = {
        "actual_vs_predicted": f"{METRICS_DIR}/plots/{model_name}/actual_vs_predicted.png",
        "distribution_comparison": f"{METRICS_DIR}/plots/{model_name}/distribution_comparison.png",
        "feature_importance": f"{METRICS_DIR}/plots/{model_name}/feature_importance.png",
        "residual_plot": f"{METRICS_DIR}/plots/{model_name}/residual_plot.png"
    }
    
    plot_path = plot_mapping.get(plot_type)
    if plot_path and os.path.exists(plot_path):
        return plot_path
    return None

# Debug function to check file structure
def debug_file_structure():
    """Debug function to check what files are available"""
    st.sidebar.subheader("🔍 Debug Info")
    if st.sidebar.checkbox("Show File Structure"):
        st.sidebar.write("**Models Directory:**")
        try:
            models = os.listdir(MODELS_DIR)
            st.sidebar.write(models)
        except:
            st.sidebar.error("Cannot access models directory")
        
        st.sidebar.write("**Metrics Directory:**")
        try:
            metrics_items = os.listdir(METRICS_DIR)
            st.sidebar.write(metrics_items)
        except:
            st.sidebar.error("Cannot access metrics directory")

# Main app functions
def main():
    st.sidebar.title("🚀 CPU Usage Predictor")
    app_mode = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Model Details", "Predict CPU Usage"]
    )
    
    # Debug info in sidebar
    debug_file_structure()
    
    # Load predictor (cached)
    predictor = load_predictor()
    
    if app_mode == "Dashboard":
        show_dashboard()
    elif app_mode == "Model Details":
        show_model_details()
    elif app_mode == "Predict CPU Usage":
        show_prediction(predictor)

def show_dashboard():
    """Display the main dashboard - Focused on model comparisons"""
    st.title("📊 Model Dashboard")
    st.markdown("Comprehensive comparison of all machine learning models")
    
    # Load data
    with st.spinner("Loading metrics data..."):
        metrics = load_model_metrics()
        comparison_data = load_comparison_data()
        best_model = get_best_model(metrics)
    
    # Debug info
    with st.expander("🔍 Debug: Loaded Data"):
        st.write("Available models:", list(metrics.keys()))
        st.write("Best model:", best_model)
    
    # Section 1: Best Model Highlight
    st.subheader("🏆 Best Performing Model")
    
    if best_model:
        best_model_metrics = metrics.get(best_model, {})
        r2_score = best_model_metrics.get('r2', 0)
        mae_score = best_model_metrics.get('mae', 0)
        rmse_score = best_model_metrics.get('rmse', 0)

        st.markdown(
            f'<div style="border: 3px solid {get_color("green")}; border-radius: 12px; padding: 20px; background-color: {get_color("bg-secondary")}; margin-bottom: 20px;">',
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"🎯 {best_model.upper().replace('_', ' ')}")
            st.metric("R² Score", f"{r2_score:.4f}", help="Higher is better")
            
        with col2:
            st.write("**Why this model is the best:**")
            st.write(f"• **Highest R² Score:** {r2_score:.4f} (closest to 1.0)")
            st.write(f"• **Low Error Rates:** MAE: {mae_score:.4f}, RMSE: {rmse_score:.4f}")
            st.write(f"• **Best overall performance** across all evaluation metrics")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Could not determine the best model from available metrics")

    # Section 2: Model Comparison Visualization
    st.subheader("📈 Model Comparison Overview")
    
    comparison_plot_path = f"{METRICS_DIR}/comparison/metrics_comparison.png"
    if os.path.exists(comparison_plot_path):
        st.image(comparison_plot_path, use_container_width=True)
        st.caption("Visual comparison of model performance metrics")
    else:
        st.info("📊 Comparison visualization will appear here once generated")

    # Section 3: Detailed Comparison Table
    st.subheader("📋 Detailed Model Comparison")
    
    if comparison_data['csv_data']:
        df = pd.DataFrame(comparison_data['csv_data'])
        st.dataframe(df, use_container_width=True)
        
        with st.expander("💡 Comparison Insights"):
            if len(df) > 0:
                if 'r2' in df.columns:
                    best_r2_row = df.loc[df['r2'].idxmax()]
                    st.write(f"**Best R² Score:** {best_r2_row['r2']:.4f} ({best_r2_row['model']})")
                
                if 'rmse' in df.columns:
                    best_rmse_row = df.loc[df['rmse'].idxmin()]
                    st.write(f"**Best RMSE:** {best_rmse_row['rmse']:.4f} ({best_rmse_row['model']})")
                
                if 'mae' in df.columns:
                    best_mae_row = df.loc[df['mae'].idxmin()]
                    st.write(f"**Best MAE:** {best_mae_row['mae']:.4f} ({best_mae_row['model']})")
    else:
        st.info("📄 Detailed comparison table will appear here once comparison data is available")

    # Section 4: Individual Model Performance Cards
    st.subheader("🤖 Individual Model Performance")
    
    col1, col2, col3 = st.columns(3)
    # UPDATED: Replaced lightgbm with linear_regression
    models = ["linear_regression", "random_forest", "xgboost"]
    columns = [col1, col2, col3]
    
    for i, model_name in enumerate(models):
        with columns[i]:
            model_metrics = metrics.get(model_name, {})
            is_best = model_name == best_model
            
            card_style = f'style="border: 2px solid {get_color("green") if is_best else get_color("bg-tertiary")}; border-radius: 10px; padding: 15px; background-color: {get_color("bg-secondary")};"'
            
            st.markdown(f'<div {card_style}>', unsafe_allow_html=True)
            
            if is_best:
                st.subheader(f"👑 {model_name.upper().replace('_', ' ')}")
            else:
                st.subheader(f"🤖 {model_name.upper().replace('_', ' ')}")
            
            if model_metrics and all(key in model_metrics for key in ['r2', 'mae', 'mse', 'rmse']):
                st.metric("R² Score", f"{model_metrics.get('r2', 0):.4f}")
                
                with st.expander("Error Metrics"):
                    st.write(f"**MAE:** {model_metrics.get('mae', 0):.4f}")
                    st.write(f"**MSE:** {model_metrics.get('mse', 0):.4f}")
                    st.write(f"**RMSE:** {model_metrics.get('rmse', 0):.4f}")
                
                r2_score = model_metrics.get('r2', 0)
                if r2_score >= 0.9:
                    st.success("Excellent 🎯")
                elif r2_score >= 0.8:
                    st.info("Very Good 👍")
                elif r2_score >= 0.7:
                    st.warning("Good ⚡")
                else:
                    st.error("Needs Improvement 📉")
                    
            else:
                st.error("❌ Metrics not available")
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Section 5: Navigation Guidance
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🔍 Want deeper insights?** 
        Visit the **Model Details** page to explore feature importance, residual plots, and detailed metrics.
        """)
    
    with col2:
        st.info("""
        **🔮 Ready to predict?** 
        Go to the **Predict CPU Usage** page to get real-time predictions from all models.
        """)

def show_model_details():
    """Display individual model details"""
    st.title("🔍 Model Details")
    
    # UPDATED: Replaced lightgbm with linear_regression
    model_name = st.selectbox(
        "Select a model to view details",
        ["linear_regression", "random_forest", "xgboost"]
    )
    
    st.subheader(f"{model_name.upper().replace('_', ' ')} Model")
    
    # CORRECTED PATH for metrics
    metrics_path = f"{METRICS_DIR}/evaluation/{model_name}_evaluation.json"
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            model_metrics = json.load(f)
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        
        if all(key in model_metrics for key in ['r2', 'mae', 'mse', 'rmse']):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R² Score", f"{model_metrics.get('r2', 0):.4f}")
            with col2:
                st.metric("MAE", f"{model_metrics.get('mae', 0):.4f}")
            with col3:
                st.metric("MSE", f"{model_metrics.get('mse', 0):.4f}")
            with col4:
                st.metric("RMSE", f"{model_metrics.get('rmse', 0):.4f}")
            
            # Additional details
            st.subheader("📈 Detailed Metrics")
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.write("**Model Information:**")
                st.write(f"- Model Type: {model_metrics.get('model', model_name)}")
                st.write(f"- R² Score: {model_metrics.get('r2', 0):.6f}")
                st.write(f"- MAE: {model_metrics.get('mae', 0):.6f}")
            
            with details_col2:
                st.write("**Error Metrics:**")
                st.write(f"- MSE: {model_metrics.get('mse', 0):.6f}")
                st.write(f"- RMSE: {model_metrics.get('rmse', 0):.6f}")
                
        else:
            st.error("❌ Metrics not in expected format")
            st.write("Available keys:", list(model_metrics.keys()))
    
    else:
        st.error(f"❌ Metrics file not found: {metrics_path}")
    
    # CORRECTED PATH for feature importance
    feature_importance_path = f"{METRICS_DIR}/feature_importance/{model_name}_feature_importance.json"
    if os.path.exists(feature_importance_path):
        st.subheader("🎯 Feature Importance")
        with open(feature_importance_path, 'r') as f:
            feature_importance = json.load(f)
        
        fi_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        st.dataframe(fi_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Features", len(fi_df))
        with col2:
            if len(fi_df) > 0:
                top_feature = fi_df.iloc[0]
                st.metric("Most Important Feature", f"{top_feature['Feature']} ({top_feature['Importance']:.4f})")
    
    # Model visualizations with CORRECTED PATHS
    st.subheader("📊 Model Visualizations")
    
    plot_types = [
        ("Actual vs Predicted", "actual_vs_predicted"),
        ("Residual Plot", "residual_plot"),
        ("Feature Importance", "feature_importance"),
        ("Distribution Comparison", "distribution_comparison")
    ]
    
    cols = st.columns(2)
    plots_found = 0
    
    for idx, (plot_title, plot_type) in enumerate(plot_types):
        plot_path = get_plot_path(model_name, plot_type)
        if plot_path:
            with cols[idx % 2]:
                st.image(plot_path, caption=plot_title, use_container_width=True)
                plots_found += 1
        else:
            with cols[idx % 2]:
                st.info(f"ℹ️ {plot_title} not available")
    
    if plots_found == 0:
        st.info("ℹ️ No visualization plots available for this model")

def show_prediction(predictor):
    """Display prediction interface"""
    st.title("🔮 Predict CPU Usage")
    
    if not predictor:
        st.error("❌ Predictor is not available. Please check if models are loaded.")
        return
    
    st.markdown("Enter the required parameters to predict CPU usage:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            cpu_request = st.number_input(
                "CPU Request (cores)", min_value=0.1, max_value=100.0, value=1.0, step=0.1
            )
            mem_request = st.number_input(
                "Memory Request (GB)", min_value=0.1, max_value=1000.0, value=4.0, step=0.1
            )
            cpu_limit = st.number_input(
                "CPU Limit (cores)", min_value=0.1, max_value=100.0, value=2.0, step=0.1
            )
        
        with col2:
            mem_limit = st.number_input(
                "Memory Limit (GB)", min_value=0.1, max_value=1000.0, value=8.0, step=0.1
            )
            runtime_minutes = st.number_input(
                "Runtime (minutes)", min_value=1.0, max_value=10080.0, value=30.0, step=1.0
            )
            controller_kind = st.selectbox(
                "Controller Kind",
                ["Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob", "ReplicaSet"]
            )
        
        submitted = st.form_submit_button("🚀 Predict CPU Usage")
    
    if submitted:
        with st.spinner("Making predictions..."):
            try:
                input_data = {
                    'cpu_request': cpu_request, 'mem_request': mem_request,
                    'cpu_limit': cpu_limit, 'mem_limit': mem_limit,
                    'runtime_minutes': runtime_minutes, 'controller_kind': controller_kind
                }
                
                predictions = predictor.predict_all(input_data)
                
                if predictions:
                    valid_predictions = {k: v for k, v in predictions.items() if v is not None}
                    
                    if valid_predictions:
                        avg_prediction = sum(valid_predictions.values()) / len(valid_predictions)
                        st.success("✅ Prediction completed!")
                        
                        st.subheader("📊 Prediction Results")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Average Prediction", f"{avg_prediction:.4f}", "cores")
                        
                        st.subheader("🤖 Individual Model Predictions")
                        pred_cols = st.columns(3)
                        
                        # UPDATED: Replaced LightGBM with Linear Regression
                        model_names = {
                            "Linear Regression": predictions.get("Linear Regression"),
                            "Random Forest": predictions.get("Random Forest"),
                            "XGBoost": predictions.get("XGBoost")
                        }
                        
                        for idx, (model_name, prediction) in enumerate(model_names.items()):
                            with pred_cols[idx]:
                                if prediction is not None:
                                    st.metric(model_name, f"{prediction:.4f}", "cores")
                                else:
                                    st.warning(f"{model_name}: N/A")
                    else:
                        st.error("❌ No valid predictions generated")
                else:
                    st.error("❌ Prediction failed")
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

if __name__ == "__main__":
    main()