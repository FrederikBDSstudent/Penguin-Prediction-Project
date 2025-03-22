import json
import joblib
import requests
import datetime

# Load the model and scaler
model = joblib.load("models/wrapper_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# (Optional) load label encoder if you used it
# label_encoder = joblib.load("models/label_encoder.pkl")

# API endpoint to get new penguin data
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# Features (must match training order used in wrapper model)
features = [[
    data["bill_length_mm"],
    data["bill_depth_mm"],
    data["flipper_length_mm"]
]]

# Scale the input
scaled_features = scaler.transform(features)

# Predict
predicted_class = model.predict(scaled_features)[0]

# If you trained with a LabelEncoder (y = 0/1/2), decode it
# predicted_species = label_encoder.inverse_transform([predicted_class])[0]
# If you trained directly with species names, use as is:
predicted_species = predicted_class

# Save prediction
prediction_result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "bill_length_mm": data["bill_length_mm"],
    "bill_depth_mm": data["bill_depth_mm"],
    "flipper_length_mm": data["flipper_length_mm"],
    "predicted_species": predicted_species
}

# Save to file
with open("data/prediction_result.json", "w") as f:
    json.dump(prediction_result, f, indent=4)

print(f"Prediction saved: {prediction_result}")
