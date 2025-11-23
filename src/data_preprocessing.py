import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
import joblib
import os
import sklearn

def load_config():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def preprocess_data():
    config = load_config()
    
    print(f"🔧 Environment Info:")
    print(f"   NumPy: {np.__version__}")
    print(f"   Pandas: {pd.__version__}")
    print(f"   Scikit-learn: {sklearn.__version__}")
    
    # Load data
    df = pd.read_csv(config['data']['raw_path'])
    print(f"📊 Loaded data with {len(df)} rows and {len(df.columns)} columns")
    
    # Enhanced data type checking
    print("🔍 Data types before processing:")
    print(df.dtypes)
    
    # Convert problematic dtypes
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try numeric conversion first
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                if not converted.isna().all():  # If some values converted successfully
                    df[col] = converted
                    print(f"🔄 Converted {col} from object to numeric")
            except Exception as e:
                print(f"⚠️ Could not convert {col} to numeric: {e}")
    
    # Handle missing values
    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"🧹 Removed {removed_rows} rows with missing values")
    else:
        print("✅ No missing values found")
    
    # Check if target column exists
    target_col = config['data']['target']
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {df.columns.tolist()}")
    
    # Check if feature columns exist
    missing_features = [feature for feature in config['data']['features'] if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in data: {missing_features}. Available columns: {df.columns.tolist()}")
    
    # Encode categorical variables
    le = LabelEncoder()
    if 'controller_kind' in df.columns:
        df['controller_kind'] = le.fit_transform(df['controller_kind'].astype(str))
        print(f"🔤 Encoded 'controller_kind' with {len(le.classes_)} classes: {le.classes_}")
    
    # Save encoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # Separate features and target
    X = df[config['data']['features']]
    y = df[config['data']['target']]
    
    print(f"🎯 Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"📋 Features: {X.columns.tolist()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['preprocessing']['test_size'],
        random_state=config['preprocessing']['random_state']
    )
    
    print(f"📈 Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Scale numerical features (exclude categorical)
    scaler = StandardScaler()
    numerical_cols = [col for col in X.columns if col != 'controller_kind']
    
    if numerical_cols:
        print(f"⚖️ Scaling numerical columns: {numerical_cols}")
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save processed data with version info
    os.makedirs('data/processed', exist_ok=True)
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'versions': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'sklearn': sklearn.__version__
        }
    }
    
    joblib.dump(processed_data, 'data/processed/processed_data.pkl', protocol=4)
    
    # Print final summary
    print("\n✅ Preprocessing completed successfully!")
    print(f"📁 Processed data saved to: data/processed/processed_data.pkl")
    print(f"🔧 Preprocessing objects saved to: models/")
    print(f"📊 Final data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return processed_data

if __name__ == "__main__":
    preprocess_data()