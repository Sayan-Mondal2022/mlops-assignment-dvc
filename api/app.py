import os
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
# Routes
# ------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
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