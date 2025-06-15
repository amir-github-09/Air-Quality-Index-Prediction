import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import os
import requests
from streamlit_lottie import st_lottie

# ────────────────────────────────
# Load Lottie animation JSON
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# ────────────────────────────────
# Load model and dataset
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "air_quality_model.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "Air_Quality.csv"))

# ────────────────────────────────
# Streamlit page config
st.set_page_config(page_title="Air Quality App", layout="wide", page_icon="🌫️")

# ────────────────────────────────
# Title and description
st.title("🌍 Air Quality Prediction App")
st.markdown("""
<div style="font-size:18px">
Predict the <b>Air Quality Index (AQI)</b> using atmospheric pollutant data.  
Use the <span style='color:#ff4b4b'><b>Sidebar</b></span> to navigate between the <b>Prediction Panel</b> and the <b>EDA Dashboard</b>.
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────
# Sidebar with Lottie and navigation
with st.sidebar:
    st.title("🔎 Navigation")
    # Load and display Lottie animation
    valid_lottie_url = "https://assets4.lottiefiles.com/packages/lf20_qp1q7mct.json"
    lottie_json = load_lottieurl(valid_lottie_url)

    if lottie_json:
        st_lottie(lottie_json, height=200, key="aqi-animation")
    else:
        st.warning("⚠️ Animation failed to load.")
    option = st.radio("Go to", ["🔮 Prediction", "📈 EDA Dashboard"])
# ────────────────────────────────
# Prediction Page
if option == "🔮 Prediction":
    st.header("🧠 Predict AQI")

    col1, col2, col3 = st.columns(3)
    with col1:
        co = st.number_input("CO", value=258)
        co2 = st.number_input("CO2", value=462)
    with col2:
        no2 = st.number_input("NO2", value=22)
        so2 = st.number_input("SO2", value=12)
    with col3:
        o3 = st.number_input("O3", value=60)
        pm25 = st.number_input("PM2.5", value=17.0)
        pm10 = st.number_input("PM10", value=35.0)

    if st.button("🚀 Predict Now"):
        input_df = pd.DataFrame({
            "CO": [co], "CO2": [co2], "NO2": [no2], "SO2": [so2],
            "O3": [o3], "PM2.5": [pm25], "PM10": [pm10],
        })
        prediction = model.predict(input_df)[0]
        st.success(f"🌫️ Predicted AQI: **{round(prediction, 2)}**")

        if prediction <= 50:
            st.balloons()
            st.info("🟢 Good Air Quality")
        elif prediction <= 100:
            st.warning("🟡 Moderate Air Quality")
        else:
            st.error("🔴 Unhealthy Air Quality")

# ────────────────────────────────
# EDA Page
elif option == "📈 EDA Dashboard":
    st.header("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Overview", "📌 City-wise Analysis", "📉 Correlation", "📊 Visualizations"
    ])

    with tab1:
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📈 Summary Statistics")
        st.dataframe(df.describe())

        st.subheader("❓ Missing Values")
        st.dataframe(df.isnull().sum())

        st.subheader("🔤 Data Types")
        st.dataframe(df.dtypes.to_frame(name="Type"))

        st.subheader("🔢 Unique Values Per Column")
        st.dataframe(df.nunique().to_frame(name="Unique Count"))

    with tab2:
        st.subheader("📍 City-wise Distribution")
        st.plotly_chart(px.histogram(df, x='City', title="City-wise Record Count", color='City'))

        st.subheader("📊 Average Pollutant Levels by City")
        mean_values = df.groupby('City').mean(numeric_only=True).reset_index()
        pollutants = ['CO', 'CO2', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']
        for col in pollutants:
            if col in mean_values.columns:
                fig = px.bar(mean_values, x='City', y=col, color=col, title=f"Average {col} Levels")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🧮 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
        st.pyplot(fig)

    with tab4:
        st.subheader("📊 Feature Distributions")
        for col in ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']:
            fig = px.histogram(df, x=col, nbins=30, marginal='box')
            fig.update_layout(title=f"Distribution of {col}")
            st.plotly_chart(fig)

        st.subheader("📌 Scatter Plots vs AQI")
        for col in ['CO', 'CO2', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']:
            fig = px.scatter(df, x=col, y='AQI', color='City', title=f"{col} vs AQI", trendline='ols')
            st.plotly_chart(fig)
