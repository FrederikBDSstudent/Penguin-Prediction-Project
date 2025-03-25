import os
import json
import joblib
import requests
import datetime
import matplotlib.pyplot as plt

# Ensure the "data" folder exists
os.makedirs("data", exist_ok=True)

# 1) Load the model, label encoder, and scaler
clf = joblib.load("models/wrapper_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

# 2) Fetch new penguin data
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# 3) Extract and scale all 4 features
features = [[
    data["bill_length_mm"],
    data["bill_depth_mm"],
    data["flipper_length_mm"],
    data["body_mass_g"]
]]
scaled_features = scaler.transform(features)

# 4) Apply the same feature selection used during training
#    (Here we assume the first 3 features were kept by RFE)
selected_features = scaled_features[:, :3]

# 5) Predict species using the wrapper model
species_encoded = clf.predict(selected_features)[0]
species = label_encoder.inverse_transform([species_encoded])[0]

# 6) Save the prediction result as JSON
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

# 7) Update statistics in prediction_historical_stats.json
STATS_FILE = "data/prediction_historical_stats.json"

# Load existing stats if available
if os.path.exists(STATS_FILE):
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
else:
    stats = {}

# Increment the count for the predicted species
stats[species] = stats.get(species, 0) + 1

# Save updated stats
with open(STATS_FILE, "w") as f:
    json.dump(stats, f, indent=4)

# 8) Generate and save a pie chart of species distribution
labels = list(stats.keys())
sizes = list(stats.values())

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Penguin Species Distribution")
plt.axis('equal')  # Ensures the pie is drawn as a circle

plt.savefig("data/species_distribution.png")
plt.close()
