# Penguin Identification in NYC

**Live Penguin Article**: https://frederikbdsstudent.github.io/Penguin-Prediction-Project/

_Every day, a penguin is mysteriously found somewhere in New York City — this is a key premise given by the assignment. Our job is to identify its species using physical measurements and generate a fully written news-style article to report the finding._

---

## Project Overview

This project is a mix of machine learning, automated inference, and creative storytelling.

A trained ML model predicts a penguin's species using real feature data. The result is turned into a dynamic article styled like a breaking news post in NYC.

---

## Features Used to Predict Species

- Bill Length (mm)
- Bill Depth (mm)
- Flipper Length (mm)

These features are fed into a machine learning model after preprocessing to produce the species classification.

---

## Model Training

We tested and compared four feature selection techniques:

- Filter Method (Mutual Information)
- Wrapper Method (RFE + Logistic Regression)
- Embedded Method (Random Forest)
- Permutation Importance

After evaluation using cross-validation, the Wrapper Method was selected as the most reliable, achieving an accuracy of 0.9910. This model was saved using joblib, along with the fitted scaler.

We used:
- StandardScaler for feature scaling
- Logistic Regression for classification
- Recursive Feature Elimination (RFE) for feature selection

The final model and scaler are saved as:
- `models/wrapper_model.pkl`
- `models/scaler.pkl`

---

## Technologies Used

- Python (scikit-learn, seaborn, pandas, joblib)
- HTML and JavaScript for the frontend
- GitHub Pages for hosting the web app
- JSON for frontend-backend communication

---

## Automated Prediction Script

The script `Auto_Penguin_Prediction.py` performs the following steps:

1. Fetches penguin feature data from an API
2. Scales the data using the saved StandardScaler
3. Uses the trained model to predict the species
4. Saves the result in `data/prediction_result.json`

---

## Frontend: Penguin News Article

The `index.html` file renders a newspaper-style article using the latest prediction.

Each article includes:

- A randomized narrative about how the penguin was found in NYC
- A specific NYC location
- The current weather and time of day
- A unique paragraph describing the predicted species
- A randomly assigned name for the penguin
- Bill and flipper measurements, rounded to 2 decimal places

All of this content is dynamically generated with JavaScript based on the latest prediction.
