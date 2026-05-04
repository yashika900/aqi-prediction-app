import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AQI Prediction App", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("🌫 Air Quality Index (AQI) Prediction App")
st.markdown("Enter pollutant values to predict AQI using Machine Learning")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv('AQI-and-Lat-Long-of-Countries.csv')  # keep file in same folder
    data = data.dropna()
    data.columns = [col.strip().lower() for col in data.columns]
    return data

data = load_data()

# -----------------------------
# TRAIN MODEL
# -----------------------------
X = data[['co aqi value', 'ozone aqi value', 'no2 aqi value', 'pm2.5 aqi value']]
y = data['aqi value']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# INPUT UI
# -----------------------------
st.subheader("📥 Enter Pollution Values")

col1, col2 = st.columns(2)

with col1:
    co = st.number_input("CO AQI Value", min_value=0.0, value=50.0)
    ozone = st.number_input("Ozone AQI Value", min_value=0.0, value=30.0)

with col2:
    no2 = st.number_input("NO2 AQI Value", min_value=0.0, value=20.0)
    pm25 = st.number_input("PM2.5 AQI Value", min_value=0.0, value=60.0)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🔍 Predict AQI"):
    
    input_data = np.array([[co, ozone, no2, pm25]])
    prediction = model.predict(input_data)[0]

    st.success(f"✅ Predicted AQI: {round(prediction, 2)}")

    # AQI CATEGORY
    if prediction <= 50:
        category = "Good 😊"
    elif prediction <= 100:
        category = "Moderate 😐"
    elif prediction <= 150:
        category = "Unhealthy for Sensitive 😷"
    elif prediction <= 200:
        category = "Unhealthy ⚠"
    else:
        category = "Hazardous 🚨"

    st.info(f"🌍 AQI Category: {category}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
