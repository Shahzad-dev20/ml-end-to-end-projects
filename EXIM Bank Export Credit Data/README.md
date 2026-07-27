# EXIM Bank Export Credit Deal Cancellation Risk Model

## Project Overview
This project builds an end-to-end machine learning pipeline to predict the probability that an approved EXIM Bank export credit deal will later be cancelled. The solution is designed for risk monitoring and triage rather than automated approval or rejection decisions.

The model uses approval-time features only, helping prevent data leakage and making the solution suitable for real-world deployment scenarios.

## Problem Statement
EXIM Bank handles many export credit deals each year. Some approved deals are later cancelled, creating operational and financial risk. The goal of this project is to identify high-risk deals early so they can be reviewed more carefully.

## Business Value
- Helps prioritize deals for closer review
- Supports proactive risk monitoring
- Improves decision support for loan/credit operations
- Provides a deployable ML workflow from data preparation to model evaluation

## Project Workflow
1. Data exploration and analysis
2. Data cleaning and feature engineering
3. Model training and evaluation
4. Model selection and validation
5. Export of a deployable pipeline artifact

## Dataset
The project uses EXIM Bank export credit data for deal approval and cancellation analysis.

Download the dataset here:
https://drive.google.com/drive/folders/1I90Ub1sYCjIXIvvcM74L-8igya47CU0K?usp=drive_link

## Model Summary
The final model selected is an XGBoost-based pipeline trained to predict cancellation risk.

### Key Performance Metrics
- Test ROC-AUC: 0.881
- Test PR-AUC (Average Precision): 0.178

### Notes on Performance
The model is best used as a risk-ranking tool rather than a strict automated decision-maker, because cancellation is a rare event and the dataset is highly imbalanced.

## Files in This Project
- notebooks/ML_Pipeline_Training.ipynb — full training notebook
- models/cancellation_model_pipeline.pkl — trained deployable model pipeline
- artifacts/model_config.json — feature configuration and training metadata
- artifacts/model_card.txt — model documentation
- artifacts/model_comparison_curves.png — model comparison plots
- artifacts/confusion_matrix_best_model.png — confusion matrix
- artifacts/feature_importance.png — feature importance plot

## Tech Stack
- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- Jupyter Notebook

## How to Use
1. Install the dependencies from requirements.txt
2. Place the dataset in the project folder
3. Open the notebook in notebooks/
4. Load the trained model from models/cancellation_model_pipeline.pkl for inference

Example inference:
```python
import pickle
import pandas as pd

with open("models/cancellation_model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

risk_scores = model.predict_proba(new_deals)[:, 1]
```

## Project Status
Completed and organized as an end-to-end ML project with training, evaluation, and model artifact delivery.

## Author
Ahmed Shahzad
