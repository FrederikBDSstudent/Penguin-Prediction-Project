import json
import joblib
import requests
import datetime

# Load our model, label encoder, and scaler
clf = joblib.load("models/model.pkl")
# Uncomment below if you're using a label encoder:
# label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

# API endpoint where we fetch new penguin data
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# Log the fetched data for debugging
print(f"[🐧 LOG] Fetched penguin data: {data}")

# Extract the features expected by the model (all 4 features)
features = [[
    data["bill_length_mm"],
    data["bill_depth_mm"],
    data["flipper_length_mm"],
    data["body_mass_g"]  # Include this fourth feature
]]
print(f"[🐧 LOG] Extracted features: {features}")

# Scale the features before making predictions
scaled_features = scaler.transform(features)
print(f"[🐧 LOG] Scaled features: {scaled_features}")

# Use the model to predict species
species = clf.predict(scaled_features)[0]
# If you use a label encoder, decode the prediction:
# species = label_encoder.inverse_transform([species])[0]

# Save the prediction result as JSON (matching your YAML file)
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

print(f"Prediction saved: {prediction_result}")
