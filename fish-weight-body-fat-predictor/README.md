# Fish Weight and Body Fat Predictor

Author: Ahmed Shahzad
Developer: Ahmed Shahzad

This project uses two regression models to predict fish weight and body fat percentage from structured tabular features. It includes a Flask web app, a Gradio interface, and Jupyter notebooks for data preparation, EDA, and model training.

## Project overview

| Dataset | Prediction | Rows used | Test R² | Test MAE |
|---|---|---:|---:|---:|
| Fish Market | Fish weight (g) | 158 | 0.94 | 66.5 g |
| Body Fat Prediction | Body fat (%) | 247 | 0.65 | 3.6 points |

## Dataset download

The raw datasets used in this project are stored externally and are not included in this GitHub repository.

Download the dataset files here:
https://drive.google.com/drive/folders/1JXvItDWQ7jnjQU7xKIiJgCJTEI_DDGc9

## Project structure

```text
01_data_inception.ipynb          Data loading, cleaning, and validation
02_eda.ipynb                     Exploratory data analysis and visualizations
03_model_training_pipeline.ipynb Model training, evaluation, and saved artifacts
gradio_app.py                    Gradio frontend for model predictions
flask_app.py                     Flask backend + route handling
common.py                        Shared prediction and metadata logic
models/                          Trained model files and metadata
templates/index.html             Flask UI template
requirements.txt                 Python project dependencies
data/raw/                        Original dataset files
data/processed/                  Cleaned dataset outputs
```

## Requirements

- Python 3.10+
- pip

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the apps

```bash
# Flask app
python flask_app.py
# Open: http://127.0.0.1:5000

# Gradio app
python gradio_app.py
# Open: http://127.0.0.1:7860
```

## Notebook flow

Run the notebooks in order if you want to rebuild the datasets and models:

```bash
jupyter notebook
```

Then open:
1. 01_data_inception.ipynb
2. 02_eda.ipynb
3. 03_model_training_pipeline.ipynb

## Notes

- The notebooks are already executed and the trained model files are included in the repository.
- The project reads from the local `models/model_info.json` file so the app forms stay aligned with the saved models.
- The Flask app also exposes a JSON API for predictions.

## Author

This project was developed by Ahmed Shahzad.
