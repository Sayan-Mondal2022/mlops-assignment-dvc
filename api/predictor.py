import os
import joblib
import numpy as np
import pandas as pd

class CPUPredictor:
    def __init__(self, models_dir="/app/models"):
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
        self.models_loaded = False
        self.models_dir = models_dir
        
    def load_models(self):
        """Load all trained models and preprocessing objects from configured models directory"""
        try:
            base_path = self.models_dir
            
            # Check if model directory exists
            if not os.path.exists(base_path):
                print(f"❌ Model directory not found: {base_path}")
                print(f"🔍 Current working directory: {os.getcwd()}")
                print(f"🔍 Directory contents: {os.listdir('.')}")
                return False
            
            print(f"📁 Loading models from: {base_path}")
            print(f"🔍 Models directory contents: {os.listdir(base_path)}")
            
            # Load preprocessing objects
            scaler_path = os.path.join(base_path, 'scaler.pkl')
            label_encoder_path = os.path.join(base_path, 'label_encoder.pkl')
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✅ Loaded scaler")
            else:
                print(f"❌ Scaler file not found: {scaler_path}")
                return False
                
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                print("✅ Loaded label encoder")
            else:
                print(f"❌ Label encoder file not found: {label_encoder_path}")
                return False
            
            # Load models - REPLACED LightGBM with Linear Regression
            model_files = {
                'XGBoost': os.path.join(base_path, 'xgboost_model.pkl'),
                'Linear Regression': os.path.join(base_path, 'linear_regression_model.pkl'), 
                'Random Forest': os.path.join(base_path, 'random_forest_model.pkl')
            }
            
            models_loaded_count = 0
            for name, path in model_files.items():
                if os.path.exists(path):
                    self.models[name] = joblib.load(path)
                    models_loaded_count += 1
                    print(f"✅ Loaded {name} model")
                else:
                    print(f"❌ Model file not found: {path}")
            
            print(f"✔ Loaded {models_loaded_count} models from: {base_path}")
            
            # Debug information
            if hasattr(self.scaler, 'n_features_in_'):
                print(f"🔍 Scaler expects {self.scaler.n_features_in_} features")
            if hasattr(self.label_encoder, 'classes_'):
                print(f"🔍 Label encoder classes: {self.label_encoder.classes_}")
            
            self.models_loaded = models_loaded_count > 0
            return self.models_loaded
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            self.models_loaded = False
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess user input for prediction - matches Streamlit app logic"""
        try:
            # Create DataFrame with the same structure as training
            df = pd.DataFrame([input_data])
            
            print(f"📊 Raw input data: {input_data}")
            
            # Encode controller_kind
            if 'controller_kind' in df.columns and self.label_encoder is not None:
                # Check if the input value is in the encoder's classes
                input_controller = input_data['controller_kind']
                if input_controller in self.label_encoder.classes_:
                    df['controller_kind'] = self.label_encoder.transform([input_controller])[0]
                    print(f"🔤 Encoded controller_kind '{input_controller}' to {df['controller_kind'].iloc[0]}")
                else:
                    print(f"❌ Unknown controller_kind: '{input_controller}'. Available: {self.label_encoder.classes_}")
                    return None
            
            # Scale numerical features
            numerical_cols = [col for col in self.feature_names if col != 'controller_kind']
            print(f"📊 Numerical features to scale: {numerical_cols}")
            print(f"📊 Features before scaling: {df[numerical_cols].values}")
            
            if self.scaler is not None:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
                print(f"📊 Features after scaling: {df[numerical_cols].values}")
            
            print(f"📊 Final processed data shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {str(e)}")
            import traceback
            print(f"🔍 Detailed preprocessing error: {traceback.format_exc()}")
            return None
    
    def predict_all(self, data):
        """Make predictions using all loaded models - matches Streamlit app logic"""
        if not self.models_loaded:
            print("❌ Models not loaded. Cannot make predictions.")
            return None
            
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
        # Convert to your original format - UPDATED to include Linear Regression
        return {
            "linear_regression": predictions.get("Linear Regression"),
            "random_forest": predictions.get("Random Forest"), 
            "xgboost": predictions.get("XGBoost")
        }
    return {"linear_regression": None, "random_forest": None, "xgboost": None}