# Student Depression Prediction Project

## 1. Propblem Description

Mental health issues, particularly depression, among university students have become a critical concern affecting academic performance, social engagement, and overall quality of life. Early detection allows educational institutions and counselors to offer timely support.

This project aims to predict the likelihood of student depression using machine learning based on gemographic, academic, social and economic information, such as family background, relationship and marital status, living style, cgpa, ect.

The trained machine learning model is served as a REST API built with FastAPI. Educational platforms or university counseling services can send student profile data to the /predict endpoint and receive a real-time risk assessment to prioritize support interventions.

Link to the dataset:
https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset?resource=download

## 2. EDA

All EDA steps and visualizations can be found in the notebook.ipynb file. 

## 3. Model Training & Hyperparameter Tuning

Multiple model architectures were trained and evaluated:

1. Linear Model: Logistic Regression (Baseline)

2. Tree-Based Models: Random Forest, XGBoost Classifier

Hyperparameter tuning was conducted (adjusting parameters such as max_depth, n_estimators, and learning_rate) to maximize the ROC-AUC metric. The Random Forest was selected as the best performing model with the Validation ROC-AUC of 0.756071 and serialized for deployment.

## 4. Dependency and Environment Management

This project uses uv.

1. Setting Up the Environment

```bash
git clone "https://github.com/emorozo2006/student-depression-detection"
cd student-depression-detection
```
```bash
uv sync
source .venv/bin/activate    
```

2. Reproducibility Instructions

To reproduce the model training pipeline locally:

```bash
python src/student_depression_detection/train.py
```
This script loads the dataset, performs data preprocessing, trains the optimal model, and saves the final binary artifact to the models/ directory.

Local Model Deployment:

```bash
uvicorn src.student_depression_detection.predict:app --host 0.0.0.0 --port 9696 --reload
```

Containerization with Docker:

```bash
docker build -t student-depression-api .
docker run -it --rm -p 9696:9696 student-depression-api
```
Cloud Deployment:

The model is deployed live on Fly.io as a cloud container service.
Base Service URL: https://falling-bird-831.fly.dev
Prediction Endpoint: https://falling-bird-831.fly.dev/predict
Interactive API Documentation: https://falling-bird-831.fly.dev/docs

Run the included test.py script to test the cloud instance.