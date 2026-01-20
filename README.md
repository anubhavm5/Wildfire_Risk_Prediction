<div align="center">

# 🔥 WildFire Risk Prediction  
### 🌍 ML + Streamlit Dashboard for Wildfire Risk Forecasting

Predict wildfire risk using weather + land conditions with an interactive map-based UI.

<br/>

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Model-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

<br/>

</div>

---

## 📌 About the Project

WildFire Risk Prediction is a **Machine Learning powered web application** that predicts the probability of wildfire occurrence based on environmental and weather inputs.  
It comes with an interactive **Streamlit dashboard** + live map interface where users can:

✅ Select a city/country  
✅ Auto-fetch coordinates  
✅ Enter weather details  
✅ Choose vegetation type  
✅ Instantly get wildfire risk probability + risk level

https://wildfireriskprediction.streamlit.app

---

## ✨ Features

✅ **City & Country Input** with automatic geocoding  
🗺️ **Interactive Map Visualization** (Folium + Streamlit)  
🌦️ Weather parameters supported:
- 🌡️ Temperature  
- 💧 Humidity  
- 🌬️ Wind Speed  
- 🌧️ Rainfall  

🌲 Land cover / Vegetation type:
- Forest  
- Grassland  
- Cropland  
- Urban  
- Barren  

🎯 **Risk Probability Output** + Classification:
- 🟢 Low Risk  
- 🟡 Moderate Risk  
- 🔴 High Risk  

---

## 🧠 Machine Learning Model

The ML model is trained using weather + land features and generates a wildfire risk probability score.

📌 **Prediction Output Includes:**
- Risk Probability Score (0 to 1)
- Risk Category (Low / Moderate / High)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/WildFire_Risk_Prediction.git
cd WildFire_Risk_Prediction

