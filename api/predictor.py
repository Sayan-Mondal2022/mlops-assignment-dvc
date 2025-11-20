import os
import joblib
import numpy as np
import pandas as pd

class CPUPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
        
    def load_models(self):
        """Load all trained models and preprocessing objects"""
        try:
            base_path = "models/cpu"
            
            # Load preprocessing objects
            self.scaler = joblib.load(f'{base_path}/scaler.pkl')
            self.label_encoder = joblib.load(f'{base_path}/label_encoder.pkl')
            
            # Load models
            model_files = {
                'XGBoost': f'{base_path}/xgboost_model.pkl',
                'LightGBM': f'{base_path}/lightgbm_model.pkl', 
                'Random Forest': f'{base_path}/random_forest_model.pkl'
            }
            
            for name, path in model_files.items():
                if os.path.exists(path):
                    self.models[name] = joblib.load(path)
                    print(f"✅ Loaded {name} model")
                else:
                    print(f"❌ Model file not found: {path}")
            
            print(f"✔ Loaded {len(self.models)} models from: {base_path}")
            
            # Debug information
            if hasattr(self.scaler, 'n_features_in_'):
                print(f"🔍 Scaler expects {self.scaler.n_features_in_} features")
            print(f"🔍 Label encoder classes: {self.label_encoder.classes_}")
            
            return len(self.models) > 0
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess user input for prediction - matches Streamlit app logic"""
        try:
            # Create DataFrame with the same structure as training
            df = pd.DataFrame([input_data])
            
            print(f"📊 Raw input data: {input_data}")
            
            # Encode controller_kind
            if 'controller_kind' in df.columns:
                df['controller_kind'] = self.label_encoder.transform(df['controller_kind'])
                print(f"🔤 Encoded controller_kind '{input_data['controller_kind']}'")
            
            # Scale numerical features
            numerical_cols = [col for col in self.feature_names if col != 'controller_kind']
            print(f"📊 Numerical features to scale: {numerical_cols}")
            print(f"📊 Features before scaling: {df[numerical_cols].values}")
            
            if self.scaler:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
                print(f"📊 Features after scaling: {df[numerical_cols].values}")
            
            print(f"📊 Final processed data shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {str(e)}")
            return None
    
    def predict_all(self, data):
        """Make predictions using all loaded models - matches Streamlit app logic"""
        predictions = {}
        
        # Preprocess input
        processed_data = self.preprocess_input(data)
        if processed_data is None:
            return None
        
        # Make predictions with each model
        for model_name, model in self.models.items():
            try:
                pred = model.predict(processed_data)[0]
                # Ensure non-negative prediction (same as Streamlit app)
                predictions[model_name] = max(0, float(pred))
                print(f"✅ {model_name} prediction: {predictions[model_name]}")
            except Exception as e:
                print(f"❌ {model_name} prediction error: {e}")
                predictions[model_name] = None
        
        return predictions
    
    def _prepare_features(self, data):
        """Alternative method for direct feature preparation (if needed)"""
        # This method is kept for compatibility with your original code
        return self.preprocess_input(data)

# For backward compatibility with your Flask app
def safe_predict_wrapper(predictor, data):
    """Wrapper to match your original predict_all method signature"""
    predictions = predictor.predict_all(data)
    if predictions:
        # Convert to your original format
        return {
            "lightgbm": predictions.get("LightGBM"),
            "random_forest": predictions.get("Random Forest"), 
            "xgboost": predictions.get("XGBoost")
        }
    return {"lightgbm": None, "random_forest": None, "xgboost": None}