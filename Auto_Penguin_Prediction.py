import os
import json
import joblib
import requests
import datetime
import matplotlib

# For headless environments (like GitHub Actions), use a non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
#    (Assuming the first 3 features were kept by RFE)
selected_features = scaled_features[:, :3]

# 5) Predict species using the wrapper model
species_encoded = clf.predict(selected_features)[0]
species = label_encoder.inverse_transform([species_encoded])[0]

# 6) Build the prediction result dictionary
prediction_result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "bill_length_mm": data["bill_length_mm"],
    "bill_depth_mm": data["bill_depth_mm"],
    "flipper_length_mm": data["flipper_length_mm"],
    "body_mass_g": data["body_mass_g"],
    "predicted_species": species
}

# 7) Save the latest prediction result as JSON (overwriting the old file)
with open("data/prediction_result.json", "w") as f:
    json.dump(prediction_result, f, indent=4)

# 8) Append the prediction result to a cumulative history file
HISTORY_FILE = "data/prediction_history.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        try:
            history = json.load(f)
        except json.decoder.JSONDecodeError:
            history = []
else:
    history = []

history.append(prediction_result)
with open(HISTORY_FILE, "w") as f:
    json.dump(history, f, indent=4)

# 9) Update statistics in prediction_historical_stats.json
STATS_FILE = "data/prediction_historical_stats.json"
if os.path.exists(STATS_FILE):
    with open(STATS_FILE, "r") as f:
        try:
            stats = json.load(f)
        except json.decoder.JSONDecodeError:
            stats = {}
else:
    stats = {}

stats[species] = stats.get(species, 0) + 1

with open(STATS_FILE, "w") as f:
    json.dump(stats, f, indent=4)

# 10) Generate and save a pretty pie chart of species distribution
labels = list(stats.keys())
sizes = list(stats.values())

# Set a clean style with Seaborn
sns.set_style("white")
fig, ax = plt.subplots(figsize=(6, 6))
# Choose a soft "Blues" palette to complement your news article layout
colors = sns.color_palette("Blues", n_colors=len(labels))
wedges, text_labels, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"color": "#2c3e50", "fontsize": 10}
)

ax.axis('equal')  # Ensure the pie is drawn as a circle
plt.setp(autotexts, size=10, weight="bold", color="white")

ax.set_title("Penguin Species Distribution", fontsize=14, fontweight="bold", color="#2c3e50")

plt.savefig("data/species_distribution.png", bbox_inches='tight', dpi=300)
plt.close()
