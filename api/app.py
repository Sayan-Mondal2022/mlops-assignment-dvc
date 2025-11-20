import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predictor import CPUPredictor
from azure.storage.blob import BlobClient
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
MODEL_CONTAINER = "models"
MODEL_PREFIX = "cpu"
MODEL_DIR = "models/cpu"

MODEL_FILES = [
    "label_encoder.pkl",
    "lightgbm_model.pkl",
    "random_forest_model.pkl",
    "scaler.pkl",
    "xgboost_model.pkl"
]

# ------------------------------------------------------
# Helper: download blob
# ------------------------------------------------------
def download_blob(container: str, blob_name: str, local_path: str):
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is missing.")

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    print(f"⬇️ Downloading blob {container}/{blob_name} -> {local_path}")
    blob = BlobClient.from_connection_string(
        conn_str=conn_str,
        container_name=container,
        blob_name=blob_name
    )

    with open(local_path, "wb") as f:
        f.write(blob.download_blob().readall())

    print(f"✔ Downloaded {local_path}")

# ------------------------------------------------------
# Download all model files at startup
# ------------------------------------------------------
def fetch_all_models():
    print("📥 Fetching model files from Azure Blob Storage...")
    for fname in MODEL_FILES:
        blob_name = f"{MODEL_PREFIX}/{fname}"
        local_path = f"{MODEL_DIR}/{fname}"

        if os.path.exists(local_path):
            print(f"✔ Already exists locally: {local_path}")
        else:
            download_blob(MODEL_CONTAINER, blob_name, local_path)

    print("✅ All model files downloaded.")

try:
    fetch_all_models()
except Exception as e:
    print("❌ Failed to download models:", e)

# ------------------------------------------------------
# Verify downloaded files
# ------------------------------------------------------
print("📁 Verifying downloaded files:")
for fname in MODEL_FILES:
    local_path = f"{MODEL_DIR}/{fname}"
    exists = os.path.exists(local_path)
    print(f"   {local_path}: {'✅' if exists else '❌'}")

# ------------------------------------------------------
# Load predictor after models are downloaded
# ------------------------------------------------------
predictor = CPUPredictor()
try:
    models_loaded = predictor.load_models()
    if models_loaded:
        print("✅ Predictor initialized successfully.")
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