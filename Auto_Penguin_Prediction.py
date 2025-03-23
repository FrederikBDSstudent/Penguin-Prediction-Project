import json
import joblib
import requests
import datetime
import os
import sys

def log(msg):
    print(f"[🐧 LOG] {msg}")

# --- Load model and scaler ---
try:
    model = joblib.load("models/wrapper_model.pkl")
    log("Model loaded successfully.")
except Exception as e:
    log(f"Error loading model: {e}")
    sys.exit(1)

try:
    scaler = joblib.load("models/scaler.pkl")
    log("Scaler loaded successfully.")
except Exception as e:
    log(f"Error loading scaler: {e}")
    sys.exit(1)

# --- Get new penguin data ---
url = "http://130.225.39.127:8000/new_penguin/"
try:
    response = requests.get(url)
    if response.status_code != 200:
        log(f"Failed to fetch data. HTTP {response.status_code}")
        sys.exit(1)
    data = response.json()
    log(f"Fetched penguin data: {data}")
except Exception as e:
    log(f"Error fetching or parsing data from API: {e}")
    sys.exit(1)

# --- Validate and extract features ---
try:
    bill_length = float(data["bill_length_mm"])
    bill_depth = float(data["bill_depth_mm"])
    flipper_length = float(data["flipper_length_mm"])
    features = [[bill_length, bill_depth, flipper_length]]
    log(f"Extracted features: {features}")
except (KeyError, TypeError, ValueError) as e:
    log(f"Invalid or missing input data: {e}")
    sys.exit(1)

# --- Scale features ---
try:
    scaled_features = scaler.transform(features)
    log(f"Scaled features: {scaled_features}")
except Exception as e:
    log(f"Error scaling features: {e}")
    sys.exit(1)

# --- Predict species ---
try:
    predicted_class = model.predict(scaled_features)[0]
    predicted_species = predicted_class  # Assuming it's already a string
    log(f"Predicted species: {predicted_species}")
except Exception as e:
    log(f"Error during prediction: {e}")
    sys.exit(1)

# --- Prepare output ---
prediction_result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "bill_length_mm": bill_length,
    "bill_depth_mm": bill_depth,
    "flipper_length_mm": flipper_length,
    "predicted_species": predicted_species
}

# --- Save result to file ---
try:
    os.makedirs("data", exist_ok=True)
    with open("data/prediction_result.json", "w") as f:
        json.dump(prediction_result, f, indent=4)
    log(f"Prediction saved to data/prediction_result.json")
except Exception as e:
    log(f"Error saving prediction: {e}")
    sys.exit(1)
