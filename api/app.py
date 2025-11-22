import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predictor import CPUPredictor
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ------------------------------------------------------
# Load predictor with local models
# ------------------------------------------------------
predictor = CPUPredictor()
try:
    models_loaded = predictor.load_models()
    if models_loaded:
        print("✅ Predictor initialized successfully.")
        print(f"✅ Loaded models: {list(predictor.models.keys())}")
    else:
        print("❌ Predictor models failed to load.")
except Exception as e:
    print("❌ Failed to load predictor models:", e)

# ------------------------------------------------------
# Helper functions for metrics loading
# ------------------------------------------------------
def load_model_metrics():
    """Load metrics for all models"""
    metrics_paths = {
        "lightgbm": "evaluation/lightgbm_evaluation.json",
        "random_forest": "evaluation/random_forest_evaluation.json", 
        "xgboost": "evaluation/xgboost_evaluation.json"
    }
    
    metrics = {}
    for model_name, path in metrics_paths.items():
        try:
            with open(path, 'r') as f:
                metrics[model_name] = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load metrics for {model_name}: {e}")
            metrics[model_name] = {}
    
    return metrics

def load_comparison_data():
    """Load model comparison data"""
    try:
        # Load CSV comparison
        df = pd.read_csv("comparison/final_model_comparison.csv")
        comparison_data = df.to_dict('records')
        
        # Load JSON comparison
        with open("comparison/final_model_comparison.json", 'r') as f:
            json_comparison = json.load(f)
            
        return {
            "csv_data": comparison_data,
            "json_data": json_comparison
        }
    except Exception as e:
        print(f"❌ Failed to load comparison data: {e}")
        return {"csv_data": [], "json_data": {}}

def get_best_model(metrics):
    """Determine the best model based on metrics"""
    if not metrics:
        return None
    
    best_model = None
    best_score = float('-inf')
    
    for model_name, model_metrics in metrics.items():
        if model_metrics and 'test_metrics' in model_metrics:
            # Use R² score to determine best model (you can change this logic)
            r2_score = model_metrics['test_metrics'].get('r2', -1)
            if r2_score > best_score:
                best_score = r2_score
                best_model = model_name
    
    return best_model

# ------------------------------------------------------
# Routes
# ------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Main dashboard page"""
    try:
        # Load all metrics and comparison data
        metrics = load_model_metrics()
        comparison_data = load_comparison_data()
        best_model = get_best_model(metrics)
        
        return render_template("dashboard.html", 
                             metrics=metrics,
                             comparison_data=comparison_data,
                             best_model=best_model)
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return render_template("dashboard.html", 
                             metrics={}, 
                             comparison_data={"csv_data": [], "json_data": {}},
                             best_model=None)

@app.route("/model/<model_name>", methods=["GET"])
def model_details(model_name):
    """Individual model details page"""
    try:
        # Load specific model metrics
        metrics_paths = {
            "lightgbm": "evaluation/lightgbm_evaluation.json",
            "random_forest": "evaluation/random_forest_evaluation.json",
            "xgboost": "evaluation/xgboost_evaluation.json"
        }
        
        if model_name not in metrics_paths:
            return render_template("error.html", error="Model not found"), 404
        
        with open(metrics_paths[model_name], 'r') as f:
            model_metrics = json.load(f)
        
        # Load feature importance if available
        feature_importance_path = f"feature_importance/{model_name}_feature_importance.json"
        feature_importance = {}
        try:
            with open(feature_importance_path, 'r') as f:
                feature_importance = json.load(f)
        except:
            print(f"⚠️ No feature importance found for {model_name}")
        
        # Get plot paths
        plot_paths = {
            "actual_vs_predicted": f"plots/{model_name}/actual_vs_predicted.png",
            "distribution_comparison": f"plots/{model_name}/distribution_comparison.png", 
            "feature_importance_plot": f"plots/{model_name}/feature_importance.png",
            "residual_plot": f"plots/{model_name}/residual_plot.png"
        }
        
        return render_template("model_details.html",
                             model_name=model_name,
                             metrics=model_metrics,
                             feature_importance=feature_importance,
                             plot_paths=plot_paths)
        
    except Exception as e:
        print(f"❌ Model details error: {e}")
        return render_template("error.html", error=str(e)), 500

@app.route("/predict", methods=["GET"])
def predict_page():
    """Prediction page"""
    return render_template("predict.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    """Prediction API endpoint"""
    # Check if predictor is ready
    if not predictor.models_loaded:
        return jsonify({
            "error": "Service unavailable - model files not loaded",
            "details": "The prediction service is not ready. Please check if all model files are available."
        }), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        print(f"🎯 Received prediction request: {data}")

        # Validate required fields
        required_fields = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}",
                "required_fields": required_fields
            }), 400

        # Use the predictor
        predictions = predictor.predict_all(data)
        
        if predictions is None:
            return jsonify({"error": "Feature preprocessing failed"}), 500

        # Filter out None predictions
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}
        
        print(f"📊 Valid predictions: {valid_predictions}")
        
        if len(valid_predictions) == 0:
            return jsonify({
                "error": "No valid predictions generated",
                "debug": {
                    "received_data": data,
                    "all_predictions": predictions
                }
            }), 500
            
        # Calculate average of valid predictions
        avg_prediction = sum(valid_predictions.values()) / len(valid_predictions)
        
        # Convert to your expected format
        result_predictions = {
            "lightgbm": predictions.get("LightGBM"),
            "random_forest": predictions.get("Random Forest"),
            "xgboost": predictions.get("XGBoost")
        }

        return jsonify({
            "avg_prediction": round(avg_prediction, 4),
            "predictions": result_predictions,
            "status": "success"
        })

    except Exception as e:
        print(f"❌ Route error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """API endpoint to get all metrics"""
    try:
        metrics = load_model_metrics()
        comparison_data = load_comparison_data()
        best_model = get_best_model(metrics)
        
        return jsonify({
            "metrics": metrics,
            "comparison": comparison_data,
            "best_model": best_model,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ready" if predictor.models_loaded else "loading",
        "models_loaded": list(predictor.models.keys()) if predictor.models_loaded else [],
        "models_loaded_count": len(predictor.models) if predictor.models_loaded else 0
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug)