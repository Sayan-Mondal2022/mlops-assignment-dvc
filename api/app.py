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
# Configuration
# ------------------------------------------------------
MODEL_DIR = "models"

MODEL_FILES = [
    "label_encoder.pkl",
    "lightgbm_model.pkl",
    "random_forest_model.pkl",
    "scaler.pkl",
    "xgboost_model.pkl"
]

# ------------------------------------------------------
# Verify model files exist locally
# ------------------------------------------------------
# def verify_model_files():
#     print("📁 Verifying model files:")
#     all_files_exist = True
    
#     for fname in MODEL_FILES:
#         local_path = f"{MODEL_DIR}/{fname}"
#         exists = os.path.exists(local_path)
#         print(f"   {local_path}: {'✅' if exists else '❌'}")
#         if not exists:
#             all_files_exist = False
            
#     return all_files_exist

# # Check if all model files are present
# models_available = verify_model_files()

# if not models_available:
#     print("❌ Some model files are missing. Please ensure all models are copied to the models/cpu/ directory.")
# else:
#     print("✅ All model files are available locally.")

# ------------------------------------------------------
# Load predictor with local models
# ------------------------------------------------------
predictor = CPUPredictor()
if True:
    try:
        models_loaded = predictor.load_models()
        if models_loaded:
            print("✅ Predictor initialized successfully.")
        else:
            print("❌ Predictor models failed to load.")
    except Exception as e:
        print("❌ Failed to load predictor models:", e)
else:
    print("⚠️  Skipping predictor initialization due to missing model files.")

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

        # Use the new predictor interface
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
            "avg_prediction": avg_prediction,
            "predictions": result_predictions
        })

    except Exception as e:
        print(f"❌ Route error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port)