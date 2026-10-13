# House Price Prediction Project

## Project overview
This project builds a machine learning workflow for house price prediction using structured property and location features. It includes exploratory analysis, preprocessing, a production-style training pipeline, and a Streamlit application for inference.

## Problem statement
Real estate pricing depends on several factors, including location, size, and property attributes. The aim is to create a model that can estimate prices from a standardized set of features and provide a simple user-facing interface.

## Business value
- Helps estimate property values quickly
- Supports buyers, sellers, and analysts in decision-making
- Demonstrates a full ML workflow from preprocessing to deployment
- Provides a lightweight web-based prediction interface

## Workflow
1. Data exploration and analysis
2. Data cleaning and preprocessing
3. Feature engineering and model training
4. Model evaluation and comparison
5. Model export and web inference

## Dataset
The project relies on a house price dataset containing property features and target values.

Download the dataset here:
https://drive.google.com/drive/folders/1qFpVzE41vVOaLfq2gwIkRgart-MyATX1?usp=drive_link

## Model summary
The final pipeline is trained as a supervised regression model using an XGBoost-based workflow with custom frequency encoding for categorical features.

## Project structure
- notebooks/analysis.ipynb — exploratory analysis
- notebooks/Data_Preprocessing.ipynb — cleaning and preprocessing
- notebooks/Production_ML_Pipeline.ipynb — training and pipeline logic
- APP/app.py — Streamlit prediction interface
- APP/feature_encoder.py — custom frequency encoder used in the production pipeline
- APP/requirements.txt — app dependencies
- models/house_price_xgb_pipeline.pkl — saved trained model
- artifacts/city_stats.json — city-level aggregated stats
- artifacts/locations.json — location mappings
- artifacts/overall_stats.json — dataset summary stats

## Tech stack
- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- plotly
- Streamlit
- Jupyter Notebook

## Setup
From the project directory:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r APP/requirements.txt
```

## Run the app
Use the app entry point expected by the project implementation:

```bash
streamlit run APP/app.py
```

The README entry `python APP/app.py` is not the correct launch syntax for this project because the application is built as a Streamlit app rather than a plain Python script.

## Project status
This project is complete as a portfolio-level regression exercise with data preparation, modeling, artifact export, and inference tooling.

## Author
Ahmed Shahzad
