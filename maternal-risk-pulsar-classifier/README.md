# Two Pulses: Maternal Health Risk and Pulsar Spotter

Author: Ahmed Shahzad
Developer: Ahmed Shahzad

This project trains two classification models and serves them through a Flask web app and a Gradio interface:

- **Maternal Risk** predicts a pregnancy risk level (low / mid / high) from six vital signs.
- **Pulsar Spotter** decides whether a radio-telescope candidate is a real pulsar or noise from eight signal statistics.

The Flask page is built around a live oscilloscope: in the maternal tab it draws an ECG at the heart rate you enter, coloured by the predicted risk; in the pulsar tab it draws a pulse train that gets taller and cleaner as the pulsar probability rises. Next to the verdict you get class probabilities, per-input effects, a radar profile against typical cases, and an honest model report card (scores, confusion matrix, feature importance).

## Project overview

| Dataset | Prediction | Rows used | Selected model | Headline test scores |
|---|---|---:|---|---|
| Maternal Health Risk (UCI) | Risk level (3 classes) | 451 (after cleaning) | Random Forest | Accuracy 0.69, macro F1 0.60, high-risk recall 0.78 |
| HTRU2 Pulsar Candidates (UCI) | Pulsar vs noise (binary, 9% positives) | 17,898 | Random Forest + tuned threshold (0.54) | Precision 0.906, recall 0.878, F1 0.892, PR-AUC 0.930 |

### Honest notes about the results

- **Maternal data had 562 exact duplicate rows out of 1,014 (55%).** Splitting before removing them leaks copies of the same row into the test set and inflates scores. We deduplicate first, so the accuracy above is the realistic figure. One impossible heart rate (7 bpm) was also removed.
- **Mid risk is the hard class** (low recall): its vitals overlap with both neighbours, and some identical-vitals rows carry different labels. Mix-ups between low and high are rare.
- **Pulsar accuracy is not the metric.** Only about 9% of candidates are pulsars, so "always say no" scores 91%. Models were compared on cross-validated PR-AUC, and the decision threshold was tuned on out-of-fold training predictions.
- The maternal model is an educational demo, **not a medical device**.

## Dataset download

The raw datasets are not committed to Git (see `.gitignore`). Download them from the project dataset folder below and place the files in `data/raw/` before running the notebooks:

- Google Drive dataset folder: https://drive.google.com/drive/folders/1QRYcR21y9LlZ7DQxja7K2IUN8_-ukfHC?usp=drive_link

Expected files: `data/raw/Maternal_Health_Risk_Data_Set.csv` and `data/raw/HTRU_2.csv` (no header row).

## Project structure

```text
01_data_inception.ipynb          Data loading, cleaning, and validation
02_eda.ipynb                     Exploratory data analysis and visualizations
03_model_training_pipeline.ipynb Model comparison, threshold tuning, evaluation, saved artifacts
gradio_app.py                    Gradio frontend for model predictions
flask_app.py                     Flask backend + JSON API
common.py                        Shared prediction and metadata logic
models/                          Trained model files and model_info.json
templates/index.html             Flask UI (oscilloscope, sliders, readout, model report card)
tests/test_common.py             Tests for the prediction logic and API
reports/figures/                 Saved EDA and evaluation charts
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
# Flask app (full interface)
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

## JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/predict/maternal \
  -H "Content-Type: application/json" \
  -d '{"Age":35,"SystolicBP":140,"DiastolicBP":90,"BS":13,"BodyTemp":98,"HeartRate":70}'
```

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Web interface |
| `/api/info` | GET | Fields, ranges, metrics, confusion matrices, importance, real examples |
| `/api/predict/<maternal\|pulsar>` | POST | Prediction with probabilities, per-input effects and warnings |
| `/health` | GET | Health check |

Invalid input returns HTTP 400 with a readable `error` message. Values outside the training range are accepted but come back with a warning.

## Tests

```bash
pytest -q
```

## Notes

- The notebooks are already executed and the trained model files are included in the repository.
- Models were saved with scikit-learn 1.8. If you install a different version and see a version warning, rerun `03_model_training_pipeline.ipynb` to rebuild them.
- The apps read `models/model_info.json`, so the input forms, ranges, scores, and the pulsar threshold always match the saved models.

## Author

This project was developed by Ahmed Shahzad.
