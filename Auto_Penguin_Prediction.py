import json
import joblib
import requests
import datetime

# Load the model trained on all 4 features, label encoder, and scaler
clf = joblib.load("models/penguin_classifier_all_features.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

# Fetch new penguin data
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# Extract all 4 features
features = [[
    data["bill_length_mm"],
    data["bill_depth_mm"],
    data["flipper_length_mm"],
    data["body_mass_g"]
]]

# Scale the features
scaled_features = scaler.transform(features)

# Predict species and decode the result
species_encoded = clf.predict(scaled_features)[0]
species = label_encoder.inverse_transform([species_encoded])[0]

# Save the prediction result as JSON
prediction_result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "bill_length_mm": data["bill_length_mm"],
    "bill_depth_mm": data["bill_depth_mm"],
    "flipper_length_mm": data["flipper_length_mm"],
    "body_mass_g": data["body_mass_g"],
    "predicted_species": species
}

with open("data/prediction_result.json", "w") as f:
    json.dump(prediction_result, f, indent=4)
