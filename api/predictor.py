import joblib
import pandas as pd
import numpy as np
import os

class CPUPredictor:
    def __init__(self):
        self.models = {}
        self.feature_names = [
            'cpu_request','mem_request','cpu_limit',
            'mem_limit','runtime_minutes','controller_kind'
        ]

    def load_models(self):
        self.scaler = joblib.load("models/scaler.pkl")
        self.label_encoder = joblib.load("models/label_encoder.pkl")

        model_files = {
            'XGBoost': 'models/xgboost_model.pkl',
            'LightGBM': 'models/lightgbm_model.pkl',
            'Random Forest': 'models/random_forest_model.pkl'
        }

        for name, path in model_files.items():
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

    def preprocess_input(self, data):
        df = pd.DataFrame([data])
        df["controller_kind"] = self.label_encoder.transform(df["controller_kind"])
        numeric = [col for col in self.feature_names if col != "controller_kind"]
        df[numeric] = self.scaler.transform(df[numeric])
        return df

    def predict_all(self, data):
        processed = self.preprocess_input(data)
        preds = {}
        for name, model in self.models.items():
            pred = model.predict(processed)[0]
            preds[name] = float(max(pred, 0))
        return preds
