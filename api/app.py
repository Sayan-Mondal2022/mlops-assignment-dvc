import streamlit as st
import pandas as pd
import json
import os
from predictor import CPUPredictor # Assuming this class exists and works

# Helper function to get color from theme
def get_color(color_name):
    """Helper function to get color from theme"""
    colors = {
        # One Atom Dark Palette (more vivid colors)
        "green": "#98c379", # Updated: was #b5bd68
        "blue": "#61afef",  # Updated: was #81a2be
        "purple": "#c678dd",# Updated: was #b294bb
        "orange": "#d19a66",# Updated: was #de935f
        "red": "#e06c75",   # Updated: was #cc6666
        "cyan": "#56b6c2",  # Updated: was #8abeb7
        "bg-secondary": "#282c34", # Background secondary
        "text-primary": "#abb2bf"  # Primary text color
    }
    return colors.get(color_name, "#61afef") # Default to blue

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
    /* 1. IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap');

    /* ONE ATOM DARK THEME VARIABLES */
    :root {{
        --bg-primary: #282c34;     /* Darker, Primary Background */
        --bg-secondary: #3b4048;   /* Slightly lighter panels/sidebar */
        --bg-tertiary: #4a505c;    /* Card highlights */
        --text-primary: #abb2bf;   /* Main text color */
        --text-secondary: #7f848e; /* Subtle text */
        --accent-green: {get_color("green")};
        --accent-blue: {get_color("blue")};
        --accent-purple: {get_color("purple")};
        --accent-orange: {get_color("orange")};
        --accent-red: {get_color("red")};
        --accent-cyan: {get_color("cyan")};
        /* Set base fonts */
        --font-main: 'Fira Code', 'monospace';
        --font-code: 'Fira Code', monospace;
    }}
    
    .main, .stApp {{
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: var(--font-main); /* Apply Inter to all main elements */
    }}
    
    /* 2. APPLY FIRA CODE TO CODE-RELATED ELEMENTS */
    pre, code, .stCode, .stCodeBlock, 
    /* Input/Output fields to look like code */
    .stNumberInput div[data-baseweb="input"] input, 
    .stTextInput div[data-baseweb="input"] input,
    .stTextArea textarea {{
        font-family: var(--font-code) !important;
        font-weight: 500;
    }}


    /* Streamlit components */
    .css-1d391kg, .css-1lcbmhc, .stSelectbox > div[role="button"], .st-bw {{
        background-color: var(--bg-secondary) !important;
        border-color: var(--bg-tertiary);
    }}
    
    /* Text input/number input (visual styling) */
    .stNumberInput div[data-baseweb="input"], .stTextInput div[data-baseweb="input"] {{
        background-color: var(--bg-tertiary);
        color: var(--text-primary);
    }}

    /* Headers and titles */
    h1, h2, h3, .css-10trblm {{ 
        color: var(--accent-blue);
    }}

    /* Info/Warning/Success blocks */
    .stAlert {{
        border-left: 5px solid var(--accent-blue);
    }}

    /* Card styling adjustment for the new palette */
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
        box-shadow: 0 0 20px rgba(152, 195, 121, 0.5); /* Use green accent for glow */
    }}
    
    .plot-container {{
        background: var(--bg-secondary);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load the predictor with cached resource"""
    predictor = CPUPredictor()
    try:
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

# Load metrics helper functions
def load_model_metrics():
    """Load metrics for all models"""
    metrics_paths = {
        "lightgbm": "metrics/evaluation/lightgbm_evaluation.json",
        "random_forest": "metrics/evaluation/random_forest_evaluation.json", 
        "xgboost": "metrics/evaluation/xgboost_evaluation.json"
    }
    
    metrics = {}
    for model_name, path in metrics_paths.items():
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    metrics[model_name] = json.load(f)
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
        csv_path = "metrics/comparison/final_model_comparison.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            comparison_data = df.to_dict('records')
        else:
            comparison_data = []
            st.warning(f"⚠️ Comparison CSV not found: {csv_path}")
        
        # Load JSON comparison
        json_path = "metrics/comparison/final_model_comparison.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_comparison = json.load(f)
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
        # Assuming metrics are at the root level of the model_metrics dictionary
        r2_score = model_metrics.get('r2', -1)
        if r2_score > best_score:
            best_score = r2_score
            best_model = model_name
    
    return best_model

def get_plot_path(model_name, plot_type):
    """Get path for a specific plot"""
    plot_mapping = {
        "actual_vs_predicted": f"metrics/plots/{model_name}/actual_vs_predicted.png",
        "distribution_comparison": f"metrics/plots/{model_name}/distribution_comparison.png",
        "feature_importance": f"metrics/plots/{model_name}/feature_importance.png",
        "residual_plot": f"metrics/plots/{model_name}/residual_plot.png"
    }
    
    plot_path = plot_mapping.get(plot_type)
    if plot_path and os.path.exists(plot_path):
        return plot_path
    return None

# Main app functions
def main():
    st.sidebar.title("🚀 CPU Usage Predictor")
    app_mode = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Model Details", "Predict CPU Usage"]
    )
    
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
    metrics = load_model_metrics()
    comparison_data = load_comparison_data()
    best_model = get_best_model(metrics)
    
    # Section 1: Best Model Highlight
    st.subheader("🏆 Best Performing Model")
    
    if best_model:
        best_model_metrics = metrics.get(best_model, {})
        r2_score = best_model_metrics.get('r2', 0)
        mae_score = best_model_metrics.get('mae', 0)
        rmse_score = best_model_metrics.get('rmse', 0)

        # Best model card with highlighted border (using new theme colors)
        st.markdown(
            f'<div style="border: 3px solid {get_color("green")}; border-radius: 12px; padding: 20px; background-color: {get_color("bg-secondary")}; margin-bottom: 20px;">',
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"🎯 {best_model.upper().replace('_', ' ')}")
            st.metric(
                "R² Score", 
                f"{r2_score:.4f}",
                help="Higher is better"
            )
            
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
    
    # Comparison plot from metrics/comparison folder
    comparison_plot_path = "metrics/comparison/metrics_comparison.png"
    if os.path.exists(comparison_plot_path):
        st.image(comparison_plot_path, use_container_width=True)
        st.caption("Visual comparison of model performance metrics")
    else:
        st.info("📊 Comparison visualization will appear here once generated")

    # Section 3: Detailed Comparison Table
    st.subheader("📋 Detailed Model Comparison")
    
    if comparison_data['csv_data']:
        # Display the comparison table
        df = pd.DataFrame(comparison_data['csv_data'])
        st.dataframe(df, use_container_width=True)
        
        # Add some insights from the comparison data
        with st.expander("💡 Comparison Insights"):
            if len(df) > 0:
                # Find best model from comparison data
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
    
    # Create columns for model cards
    col1, col2, col3 = st.columns(3)
    
    models = ["lightgbm", "random_forest", "xgboost"]
    columns = [col1, col2, col3]
    
    for i, model_name in enumerate(models):
        with columns[i]:
            model_metrics = metrics.get(model_name, {})
            is_best = model_name == best_model
            
            # Card container (using new theme colors)
            card_style = ""
            if is_best:
                card_style = f'style="border: 2px solid {get_color("green")}; border-radius: 10px; padding: 15px; background-color: {get_color("bg-secondary")};"'
            else:
                card_style = f'style="border: 1px solid {get_color("bg-tertiary")}; border-radius: 10px; padding: 15px; background-color: {get_color("bg-secondary")};"'
            
            st.markdown(f'<div {card_style}>', unsafe_allow_html=True)
            
            # Model header with icon
            if is_best:
                st.subheader(f"👑 {model_name.upper().replace('_', ' ')}")
            else:
                st.subheader(f"🤖 {model_name.upper().replace('_', ' ')}")
            
            # Display metrics if available
            if model_metrics and all(key in model_metrics for key in ['r2', 'mae', 'mse', 'rmse']):
                st.metric("R² Score", f"{model_metrics.get('r2', 0):.4f}")
                
                # Compact error metrics
                with st.expander("Error Metrics"):
                    st.write(f"**MAE:** {model_metrics.get('mae', 0):.4f}")
                    st.write(f"**MSE:** {model_metrics.get('mse', 0):.4f}")
                    st.write(f"**RMSE:** {model_metrics.get('rmse', 0):.4f}")
                
                # Performance indicator
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
                with st.expander("Debug Info"):
                    st.write("Available keys:", list(model_metrics.keys()))
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Section 5: JSON Comparison Data (if available)
    if comparison_data['json_data']:
        st.subheader("🔧 Technical Comparison Data")
        
        with st.expander("View Raw Comparison Data"):
            st.json(comparison_data['json_data'])

    # Section 6: Navigation Guidance
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **🔍 Want deeper insights?** Visit the **Model Details** page to explore:
        • Feature importance analysis
        • Residual plots
        • Prediction vs actual charts
        • Detailed performance metrics
        """)
    
    with col2:
        st.info(f"""
        **🔮 Ready to predict?** Go to the **Predict CPU Usage** page to:
        • Input your parameters
        • Get real-time predictions
        • Compare model outputs
        • Make informed decisions
        """) 
        
def show_model_details():
    """Display individual model details"""
    st.title("🔍 Model Details")
    
    model_name = st.selectbox(
        "Select a model to view details",
        ["lightgbm", "random_forest", "xgboost"]
    )
    
    st.subheader(f"{model_name.upper().replace('_', ' ')} Model")
    
    # Load specific model metrics
    metrics_path = f"metrics/evaluation/{model_name}_evaluation.json"
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            model_metrics = json.load(f)
        
        # Performance metrics - your structure has metrics at root level
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
            st.json(model_metrics)
    
    else:
        st.error(f"❌ Metrics file not found: {metrics_path}")
    
    # Feature importance
    feature_importance_path = f"metrics/feature_importance/{model_name}_feature_importance.json"
    if os.path.exists(feature_importance_path):
        st.subheader("🎯 Feature Importance")
        with open(feature_importance_path, 'r') as f:
            feature_importance = json.load(f)
        
        # Convert to DataFrame for better display
        fi_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        st.dataframe(fi_df, use_container_width=True)
        
        # Show feature importance stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Features", len(fi_df))
        with col2:
            top_feature = fi_df.iloc[0]
            st.metric("Most Important Feature", f"{top_feature['Feature']} ({top_feature['Importance']:.4f})")
    
    # Model visualizations
    st.subheader("📊 Model Visualizations")
    
    plot_types = [
        ("Actual vs Predicted", "actual_vs_predicted"),
        ("Residual Plot", "residual_plot"),
        ("Feature Importance", "feature_importance"),
        ("Distribution Comparison", "distribution_comparison")
    ]
    
    # Display plots in 2x2 grid
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
    
    if not predictor or not predictor.models_loaded:
        st.error("❌ Predictor is not available. Please check if models are loaded.")
        return
    
    st.markdown("Enter the required parameters to predict CPU usage:")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            cpu_request = st.number_input(
                "CPU Request (cores)",
                min_value=0.1,
                max_value=100.0,
                value=1.0,
                step=0.1,
                help="Requested CPU cores"
            )
            
            mem_request = st.number_input(
                "Memory Request (GB)",
                min_value=0.1,
                max_value=1000.0,
                value=4.0,
                step=0.1,
                help="Requested memory in GB"
            )
            
            cpu_limit = st.number_input(
                "CPU Limit (cores)",
                min_value=0.1,
                max_value=100.0,
                value=2.0,
                step=0.1,
                help="CPU limit in cores"
            )
        
        with col2:
            mem_limit = st.number_input(
                "Memory Limit (GB)",
                min_value=0.1,
                max_value=1000.0,
                value=8.0,
                step=0.1,
                help="Memory limit in GB"
            )
            
            runtime_minutes = st.number_input(
                "Runtime (minutes)",
                min_value=1.0,
                max_value=10080.0,  # 1 week
                value=30.0,
                step=1.0,
                help="Job runtime in minutes"
            )
            
            controller_kind = st.selectbox(
                "Controller Kind",
                ["Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob", "ReplicaSet"],
                help="Type of Kubernetes controller"
            )
        
        submitted = st.form_submit_button("🚀 Predict CPU Usage")
    
    # Handle prediction
    if submitted:
        with st.spinner("Making predictions..."):
            try:
                # Prepare input data
                input_data = {
                    'cpu_request': cpu_request,
                    'mem_request': mem_request,
                    'cpu_limit': cpu_limit,
                    'mem_limit': mem_limit,
                    'runtime_minutes': runtime_minutes,
                    'controller_kind': controller_kind
                }
                
                # Make prediction
                predictions = predictor.predict_all(input_data)
                
                if predictions:
                    # Filter out None predictions
                    valid_predictions = {k: v for k, v in predictions.items() if v is not None}
                    
                    if valid_predictions:
                        # Calculate average
                        avg_prediction = sum(valid_predictions.values()) / len(valid_predictions)
                        
                        # Display results
                        st.success("✅ Prediction completed!")
                        
                        # Average prediction
                        st.subheader("📊 Prediction Results")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Average Prediction", 
                                f"{avg_prediction:.4f}",
                                "cores"
                            )
                        
                        # Individual model predictions
                        st.subheader("🤖 Individual Model Predictions")
                        pred_cols = st.columns(3)
                        
                        model_names = {
                            "LightGBM": predictions.get("LightGBM"),
                            "Random Forest": predictions.get("Random Forest"),
                            "XGBoost": predictions.get("XGBoost")
                        }
                        
                        for idx, (model_name, prediction) in enumerate(model_names.items()):
                            with pred_cols[idx]:
                                if prediction is not None:
                                    st.metric(
                                        model_name,
                                        f"{prediction:.4f}",
                                        "cores"
                                    )
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