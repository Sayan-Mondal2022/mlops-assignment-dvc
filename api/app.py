from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predictor import CPUPredictor
import os

app = Flask(__name__)
CORS(app)

# Load your ML models
predictor = CPUPredictor()
predictor.load_models()

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
            "mem_limit", "runtime_minutes"
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Run prediction
        predictions = predictor.predict_all(data)

        response = {
            "avg_prediction": float(sum(predictions.values()) / len(predictions)),
            "predictions": predictions
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
