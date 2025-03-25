import json
import joblib
import requests
import datetime
import os
import sys

def log(msg):
    print(f"[🐧 LOG] {msg}")

# --- Debug: Print working directory and target JSON file path ---
cwd = os.getcwd()
log(f"Current working directory: {cwd}")
json_file_path = os.path.abspath("data/prediction_result.json")
log(f"Prediction file will be saved at: {json_file_path}")

# --- Load model, scaler, and label encoder ---
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

try:
    label_encoder = joblib.load("models/label_encoder.pkl")
    log("Label encoder loaded successfully.")
except Exception as e:
    log(f"Error loading label encoder: {e}")
    sys.exit(1)

# --- Get new penguin data from API ---
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

# --- Validate and extract features (all four features) ---
try:
    bill_length = float(data["bill_length_mm"])
    bill_depth = float(data["bill_depth_mm"])
    flipper_length = float(data["flipper_length_mm"])
    body_mass = float(data["body_mass_g"])
    features = [[bill_length, bill_depth, flipper_length, body_mass]]
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
    predicted_species = label_encoder.inverse_transform([predicted_class])[0]
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
    "body_mass_g": body_mass,
    "predicted_species": predicted_species
}

# --- Save result to file ---
try:
    os.makedirs("data", exist_ok=True)
    with open("data/prediction_result.json", "w") as f:
        json.dump(prediction_result, f, indent=4)
    log(f"Prediction saved to {os.path.abspath('data/prediction_result.json')}")
except Exception as e:
    log(f"Error saving prediction: {e}")
    sys.exit(1)

# --- Optional: Verify file update by reading back the JSON ---
try:
    with open("data/prediction_result.json", "r") as f:
        saved_data = json.load(f)
    log(f"Verification: Contents of the saved JSON: {saved_data}")
except Exception as e:
    log(f"Error reading back prediction: {e}")
    sys.exit(1)
