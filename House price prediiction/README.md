# House Price Prediction Project

## Project Overview
This project builds an end-to-end machine learning solution for predicting house prices based on property-related features. It includes data preprocessing, exploratory analysis, model training, and a simple web application for making predictions.

## Problem Statement
Real-estate pricing depends on multiple factors such as location, size, number of bedrooms, and other property characteristics. The goal of this project is to create a reliable predictive model that can estimate house prices accurately and make the workflow accessible through a simple app.

## Business Value
- Helps estimate property values quickly
- Supports decision-making for buyers, sellers, and real-estate analysts
- Demonstrates a complete ML pipeline from data preparation to deployment
- Provides a practical prediction interface through a web app

## Project Workflow
1. Data analysis and exploratory investigation
2. Data cleaning and preprocessing
3. Feature engineering and model training
4. Model evaluation and selection
5. Deployment through a lightweight application

## Dataset
The project uses a house price dataset containing property attributes and target price values.

Download the dataset here:
https://drive.google.com/drive/folders/1qFpVzE41vVOaLfq2gwIkRgart-MyATX1?usp=drive_link

## Model Summary
The final model is a machine learning pipeline trained for house price prediction.

## Project Structure
- notebooks/analysis.ipynb — exploratory analysis
- notebooks/Data_Preprocessing.ipynb — data cleaning and preprocessing steps
- notebooks/Production_ML_Pipeline.ipynb — training and production pipeline workflow
- models/house_price_xgb_pipeline.pkl — trained model artifact
- APP/app.py — prediction web application
- APP/feature_encoder.py — feature encoding utilities
- artifacts/city_stats.json — city-level statistics
- artifacts/locations.json — location mapping details
- artifacts/overall_stats.json — overall dataset statistics

## Tech Stack
- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- Flask / Python app interface
- Jupyter Notebook

## How to Use
1. Install dependencies from requirements.txt
2. Download and place the dataset in the project folder
3. Open the notebooks for preprocessing and training
4. Run the app in APP/ to make predictions

## Running the Application
From the project folder, run:
```bash
python APP/app.py
```

## Project Status
Completed and organized as a full house price prediction project with analysis, modeling, and deployment components.

## Author
Machine Learning Project Portfolio
