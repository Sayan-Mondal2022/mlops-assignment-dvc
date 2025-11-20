import os
import subprocess
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predictor import CPUPredictor

# -----------------------------------------
# 1️⃣ Initialize Flask
# -----------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# -----------------------------------------
# 2️⃣ Run DVC Pull (Download models from Azure)
# -----------------------------------------
def run_dvc_pull():
    try:
        print("🔄 Running DVC pull to fetch model files...")
        subprocess.run(["dvc", "pull"], check=True)
        print("✅ DVC pull completed successfully!")
    except Exception as e:
        print("❌ DVC pull failed:", e)
        print("⚠️ Will continue — app will retry when models load.")

run_dvc_pull()

# -----------------------------------------
# 3️⃣ Load models AFTER DVC finishes
# -----------------------------------------
predictor = CPUPredictor()
try:
    predictor.load_models()
    print("✅ Models loaded successfully!")
except Exception as e:
    print("❌ Failed loading models:", e)

# -----------------------------------------
# 4️⃣ Routes
# -----------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/home", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        required_fields = [
            "cpu_request", "mem_request", "cpu_limit",
            "mem_limit", "runtime_minutes", "controller_kind"
        ]

        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        predictions = predictor.predict_all(data)

        response = {
            "avg_prediction": float(sum(predictions.values()) / len(predictions)),
            "predictions": predictions
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------
# 5️⃣ Start server
# -----------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
