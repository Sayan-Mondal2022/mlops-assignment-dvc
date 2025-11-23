# CPU Predictor App – MLOps Assignment  

This project is an end-to-end MLOps learning pipeline where CPU performance is predicted using multiple machine learning models. It integrates DVC for data and model versioning, Docker for containerization, and Azure ML for deployment. The goal is to understand the complete lifecycle of building, tracking, and deploying ML models in a production-like environment.

## 🧠 Project Overview  
This repository contains an end-to-end MLOps workflow built for learning purposes. The goal is to build a predictive model for CPU usage (or a related regression task), version and track data and models using DVC, containerize the solution with Docker, and deploy it in the cloud via Azure ML / Azure Container Apps.

Key features:  
- Three ML models trained and evaluated:  
  1. Linear Regression  
  2. XGBoost Regressor  
  3. RandomForestRegressor (Random Forest)  
- Version control and experiment tracking using DVC  
- Containerization via Docker  
- Cloud deployment on Azure ML / Azure Container Apps  

This project is done **solely for learning purposes**.

## 🚀 Live Demo

You can try the deployed CPU Predictor web application here:

***Dashboard***
<img width="1911" height="986" alt="image" src="https://github.com/user-attachments/assets/99265af8-f15a-4b0e-8061-3646245d47cb" />

***Individual Model data***
<img width="1920" height="992" alt="image" src="https://github.com/user-attachments/assets/71b3b13d-c147-4653-8d09-9a98ac890de8" />


***CPU Usage Prediction***
<img width="1909" height="865" alt="image" src="https://github.com/user-attachments/assets/cb779432-db88-48b3-87e8-9092084b3666" />


🔗 **Live App:** [http://cpu-predictor-new.purplestone-5673d9e8.centralindia.azurecontainerapps.io/](http://cpu-predictor-new.purplestone-5673d9e8.centralindia.azurecontainerapps.io/)

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
│   ├── linear_regression_model.pkl
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

## 📊 Models & Metrics

### Model Overview

The three models trained are as follows:

- **Linear Regression** – a simple yet powerful algorithm that models the relationship between variables by fitting a straight line through the data points.
- **XGBoost Regressor** – Well-known distributed gradient boosting framework
- **Random Forest Regressor** – Ensemble of decision trees for robust performance

### 📈 Evaluation Metrics

Model evaluation metrics are tracked via DVC and include:

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error) 
- **R²** (R-squared)
- Additional classification metrics (accuracy, precision, recall, F1-score)

## 🛠️ Why This Setup?

- **Reproducibility**: DVC tracks data, models, metrics and ensures consistent pipeline runs.
- **Modularity**: Separation of data, model training, evaluation, and deployment.
- **Scalability**: Docker + Azure enable the solution to run in production-style environments.
- **Learning Focus**: This project was designed to deepen understanding of the MLOps lifecycle.


## 🟦 Deployment Steps (Azure ML + Azure Container Apps)
Below is the general workflow followed to deploy the ML model using Azure ML, Docker, and Azure Container Apps.

### Either run:
```bash
./deploy.sh
```

*else*

### **1️⃣ Create Azure ML Workspace & Compute Instance**
- Create a new **Azure Machine Learning Workspace**.
- Inside the workspace, create a **Compute Instance (VM)**.
- SSH or open terminal inside the compute instance to run all deployment commands.


### **2️⃣ Build the Docker Image**
From the project root, build the Docker image using the API’s Dockerfile:

```bash
docker build -t <image-name>:latest -f <api-folder>/Dockerfile .
```

### **3️⃣ Tag the Image for Azure Container Registry (ACR)**
Tag the local Docker image so it can be pushed to your Azure Container Registry:

```bash
docker tag <image-name>:latest <acr-name>.azurecr.io/<image-name>:latest
```

### **4️⃣ Push the Image to ACR**
Push the tagged image to the Azure Container Registry:

```bash
docker push <acr-name>.azurecr.io/<image-name>:latest
```

### **5️⃣ Update the Existing Azure Container App**
Update the container app to use the newly pushed image:

```bash
az containerapp update \
  --name <container-app-name> \
  --resource-group <resource-group> \
  --image "<acr-name>.azurecr.io/<image-name>:latest" \
  --set-env-vars PORT=<port-number>
```

### **6️⃣ Restart the Container App Revision**
Restart the active revision to apply changes:

```bash
az containerapp revision restart \
  --name <container-app-name> \
  --resource-group <resource-group> \
  --revision <revision-name>
```

### **7️⃣ Querying the URL**
Restart the active revision to apply changes:

```bash
az containerapp show \
    --name <YOUR_CONTAINER_APP_NAME> \
    --resource-group <YOUR_RESOURCE_GROUP_NAME> \
    --query 'properties.configuration.ingress.fqdn' \
    --output tsv
```

### **8️⃣ View Container Logs (Optional but Recommended)**
To debug issues or verify successful startup:

```bash
az containerapp logs show \
  --name <container-app-name> \
  --resource-group <resource-group> \
  --revision <revision-name> \
  --follow
```

### **✔️ Deployment Complete**

Your containerized ML application is now deployed and running on **Azure Container Apps**, pulling the image from **Azure Container Registry**.

## 🙏 Acknowledgments

I would like to express my sincere gratitude to everyone who contributed to the learning resources, documentation, and tools that made this project possible.  
Special thanks to the Azure ML, DVC, Docker, and open-source communities for providing excellent platforms, guides, and examples that helped me understand the complete MLOps workflow.

## 💙 Thank You

Thank you for taking the time to explore this project.  
This work was created purely for learning and hands-on experience, and I truly appreciate your interest and support.
