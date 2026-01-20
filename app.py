import streamlit as st
import folium
from streamlit_folium import st_folium
import pickle
import pandas as pd
from geopy.geocoders import Nominatim
import os

st.set_page_config(page_title="Wildfire Risk Prediction", layout="wide")
st.title("Wildfire Risk Prediction")

# -----------------------------
# Load trained model safely ✅
# -----------------------------
model = None
load_err = None

try:
    # Works on Windows + GitHub + Streamlit Cloud
    model_path = os.path.join(os.path.dirname(__file__), "wildfire_model.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

except Exception as e:
    load_err = e
    st.error(f"❌ Could not load model from {model_path}: {e}")

# Sidebar inputs
st.sidebar.header("🌍 Location Input")
city = st.sidebar.text_input("City", "Delhi")
country = st.sidebar.text_input("Country", "India")

# Detect coordinates automatically
geolocator = Nominatim(user_agent="wildfire_app")
location = None
latitude, longitude = None, None

if city and country:
    try:
        query = f"{city}, {country}"
        location = geolocator.geocode(query, timeout=10)
        if location is not None:
            latitude, longitude = location.latitude, location.longitude
            st.sidebar.success(f"📍 Found location: {location.address}")
        else:
            st.sidebar.error("❌ Location not found. Please adjust city/country.")
    except Exception as e:
        st.sidebar.error(f"⚠️ Geocoding error: {e}")

# Manual fallback if geocoding failed
if latitude is None or longitude is None:
    latitude = st.sidebar.number_input("Latitude", value=28.6139, format="%.6f")
    longitude = st.sidebar.number_input("Longitude", value=77.2090, format="%.6f")

# Show map
m = folium.Map(location=[latitude, longitude], zoom_start=5)
folium.Marker([latitude, longitude], popup=f"{city}, {country}").add_to(m)
st_map = st_folium(m, width=700, height=400)

# Weather inputs
st.sidebar.header("🌦️ Weather Conditions")
temperature = st.sidebar.slider("🌡️ Temperature (°C)", -10, 50, 30)
humidity = st.sidebar.slider("💧 Humidity (%)", 0, 100, 40)
wind_speed_kmh = st.sidebar.slider("💨 Wind Speed (km/h)", 0, 150, 20)
rainfall = st.sidebar.slider("🌧️ Rainfall (mm)", 0, 300, 10)

# Convert wind speed km/h -> m/s to match typical Open-Meteo training units
wind_speed = wind_speed_kmh / 3.6

# Vegetation (auto-detect with override)
st.sidebar.header("🌳 Vegetation Features")
auto_lc = None
tl = f"{city or ''} {country or ''}".lower()

if "forest" in tl:
    auto_lc = "Forest"
elif "desert" in tl:
    auto_lc = "Barren"
elif "city" in tl or (country or "").lower() in ["singapore", "monaco"]:
    auto_lc = "Urban"

lc_options = ["Forest", "Grassland", "Cropland", "Urban", "Barren"]
land_cover = st.sidebar.selectbox(
    "Land Cover Type (auto-detected, can override)",
    lc_options,
    index=(lc_options.index(auto_lc) if auto_lc in lc_options else 0),
)

# Base feature dict (engineered features consistent with training)
feature_values = {
    "temperature": float(temperature),
    "humidity": float(humidity),
    "wind_speed": float(wind_speed),
    "rainfall": float(rainfall),
    "temp_humidity_ratio": float(temperature) / (float(humidity) + 1e-5),
    "temp_anomaly": 0.0,  # no history in-app; keep placeholder
    # land cover one-hot placeholders
    "forest": 0.0,
    "grassland": 0.0,
    "cropland": 0.0,
    "urban": 0.0,
    "barren": 0.0,
}

# Activate selected land cover one-hot
lc_map = {
    "Forest": "forest",
    "Grassland": "grassland",
    "Cropland": "cropland",
    "Urban": "urban",
    "Barren": "barren"
}
feature_values[lc_map[land_cover]] = 1.0

# Build DataFrame and align to model columns
df_input = pd.DataFrame([feature_values])

if model is not None and hasattr(model, "feature_names_in_"):
    model_cols = list(model.feature_names_in_)

    # add missing columns as 0.0
    for c in model_cols:
        if c not in df_input.columns:
            df_input[c] = 0.0

    # drop extras and reorder
    df_input = df_input[model_cols]

# Prediction
if st.button("🔮 Predict Wildfire Risk"):
    if model is None:
        st.error("Model not loaded, cannot predict.")
    else:
        try:
            proba = model.predict_proba(df_input)

            # pick positive class probability robustly using classes_
            if hasattr(model, "classes_"):
                try:
                    pos_idx = list(model.classes_).index(1)
                except ValueError:
                    pos_idx = int(proba.argmax())
            else:
                pos_idx = 1

            prob = float(proba[0, pos_idx])

            st.subheader(f"🔥 Wildfire Risk Probability: {prob:.2%}")
            if prob > 0.7:
                st.error("⚠️ High Risk! Immediate precautions recommended.")
            elif prob > 0.4:
                st.warning("🟠 Moderate Risk. Stay alert.")
            else:
                st.success("✅ Low Risk. Conditions are relatively safe.")

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")


