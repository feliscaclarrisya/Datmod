import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Risiko Stunting", layout="wide")
st.title("🧒 Prediksi Risiko Stunting")
st.markdown("**LAB IS411 Data Modelling - Kelompok UTS**")

# Load Model
model = joblib.load('stunting_rf_model.pkl')

st.sidebar.header("Input Data Negara")

col1, col2 = st.columns(2)

with col1:
    stunting_prev = st.slider("Stunting Prevalence (%)", 0.0, 60.0, 15.0)
    poverty = st.slider("Poverty Rate (%)", 0.0, 100.0, 15.0)
    fertility = st.slider("Fertility Rate", 1.0, 8.0, 2.5)
    gdp = st.number_input("GDP per Capita (USD)", 100, 40000, 5000)

with col2:
    urban = st.slider("Urban Population (%)", 10.0, 100.0, 60.0)
    male_lit = st.slider("Male Literacy Rate (%)", 10.0, 100.0, 90.0)

# Tombol Prediksi
if st.button("🔍 Prediksi Risiko Stunting", type="primary"):
    input_df = pd.DataFrame({
        'Stunting_Prevalence': [stunting_prev],
        'Poverty_Rate': [poverty],
        'Urban_Population_Percent': [urban],
        'GDP_per_Capita': [gdp],
        'Fertility_Rate': [fertility],
        'Male_Literacy_Rate': [male_lit]
    })
    
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    
    risk_map = {0: "🟢 Rendah", 1: "🟡 Sedang", 2: "🔴 Tinggi"}
    
    st.success(f"**Risiko Stunting: {risk_map[pred]}**")
    
    st.bar_chart({
        "Rendah": [prob[0]],
        "Sedang": [prob[1]],
        "Tinggi": [prob[2]]
    })

# Dataset
st.subheader("Dataset Overview")
try:
    df = pd.read_csv("Child_Malnutrition_with_Socioeconomic_Factors_dataset.csv")
    st.dataframe(df.head(10))
except:
    st.info("Dataset tidak ditemukan.")
