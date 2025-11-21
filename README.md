# CPU Predictor App – MLOps Assignment  

This project is an end-to-end MLOps learning pipeline where CPU performance is predicted using multiple machine learning models. It integrates DVC for data and model versioning, Docker for containerization, and Azure ML for deployment. The goal is to understand the complete lifecycle of building, tracking, and deploying ML models in a production-like environment.

## 🧠 Project Overview  
This repository contains an end-to-end MLOps workflow built for learning purposes. The goal is to build a predictive model for CPU usage (or a related regression task), version and track data and models using DVC, containerize the solution with Docker, and deploy it in the cloud via Azure ML / Azure Container Apps.

Key features:  
- Three ML models trained and evaluated:  
  1. LightGBM (Light GBM)  
  2. XGBoost Regressor  
  3. RandomForestRegressor (Random Forest)  
- Version control and experiment tracking using DVC  
- Containerization via Docker  
- Cloud deployment on Azure ML / Azure Container Apps  

This project is done **solely for learning purposes**.

## 🚀 Live Demo

You can try the deployed CPU Predictor web application here:

<img width="621" height="908" alt="image" src="https://github.com/user-attachments/assets/d6b4b47b-dfc2-493b-94f5-27eac351da3e" />


🔗 **Live App:** https://cpu-predictor-app.purplestone-5673d9e8.centralindia.azurecontainerapps.io/

This demo is hosted on **Azure Container Apps**, fully containerized using **Docker**, and powered by the trained ML models tracked with **DVC**.  
Use the UI to input values and get real-time CPU performance predictions.


## 📁 Repository Structure  

```bash
mlops-assignment-dvc/
│
├── api/                         # Flask API for model inference
│   ├── static/
│   │   ├── scripts.js
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   ├── app.py                   # Main API entrypoint
│   ├── Dockerfile               # Dockerfile for API container
│   └── requirements.txt         # API dependencies
│
├── data/                        # Datasets (DVC-tracked)
│   ├── raw/
│   │   └── data.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── models/                      # Stored ML models (DVC-tracked)
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   └── random_forest_model.pkl
│
├── src/                         # Source code for pipeline stages
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data cleaning & feature engineering
│   ├── train.py                 # Training script for all 3 models
│   ├── evaluate.py              # Evaluation & metrics generation
│   └── utils.py                 # Helper functions
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_train.py
│   └── test_evaluate.py
│
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Model hyperparameters & config
├── metrics.json                 # Metrics tracked via DVC
├── docker-compose.yml           # Optional multi-container setup
├── .dvc/                        # DVC internal files
├── .gitignore
├── requirements.txt             # Project-level dependencies
├── setup.py                     # Package installation
└── README.md
```

## 🚀 Getting Started  
### Prerequisites  
- Python 3.x  
- Docker  
- DVC  
- Azure CLI (for deployment)  

### Local Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Sayan-Mondal2022/mlops-assignment-dvc.git
   cd mlops-assignment-dvc
    ```

2. Pull data and models via DVC:
   ```bash
    dvc pull
   ```

3. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```

4. Run the pipeline locally (optional):
   ```bash
    dvc repro
   ```

5. Launch the app locally via Docker:
   ```bash
    cd docker
    docker build -t cpu-predictor .
    docker run -p 5000:5000 cpu-predictor
   ```

Then open your browser to [`http://localhost:5000`](http://localhost:5000)
