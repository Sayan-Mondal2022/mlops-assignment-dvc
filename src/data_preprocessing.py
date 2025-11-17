import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
import joblib
import os

def load_config():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def preprocess_data():
    config = load_config()
    
    # Load data
    df = pd.read_csv(config['data']['raw_path'])
    print(f"📊 Loaded data with {len(df)} rows and {len(df.columns)} columns")
    
    # Display basic info
    print("Data columns:", df.columns.tolist())
    print("Data types:\n", df.dtypes)
    print("Missing values:\n", df.isnull().sum())
    
    # Handle missing values
    initial_rows = len(df)
    df = df.dropna()
    print(f"🧹 Removed {initial_rows - len(df)} rows with missing values")
    
    # Check if target column exists
    target_col = config['data']['target']
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {df.columns.tolist()}")
    
    # Check if feature columns exist
    for feature in config['data']['features']:
        if feature not in df.columns:
            raise ValueError(f"Feature column '{feature}' not found in data. Available columns: {df.columns.tolist()}")
    
    # Encode categorical variables
    le = LabelEncoder()
    if 'controller_kind' in df.columns:
        df['controller_kind'] = le.fit_transform(df['controller_kind'].astype(str))
        print("🔤 Encoded 'controller_kind' categorical variable")
    
    # Save encoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # Separate features and target
    X = df[config['data']['features']]
    y = df[config['data']['target']]
    
    print(f"🎯 Features: {X.shape}, Target: {y.shape}")
    
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
    
    if numerical_cols:  # Only scale if there are numerical columns
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        print(f"⚖️ Scaled numerical columns: {numerical_cols}")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist()
    }
    
    joblib.dump(processed_data, 'data/processed/processed_data.pkl')
    
    # Print summary
    print("\n✅ Preprocessing completed successfully!")
    print(f"📁 Processed data saved to: data/processed/processed_data.pkl")
    print(f"🔧 Preprocessing objects saved to: models/")
    
    return processed_data

if __name__ == "__main__":
    preprocess_data()