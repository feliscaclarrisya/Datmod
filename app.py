import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Risiko Stunting",
    page_icon="🧒",
    layout="wide"
)

# Header dengan desain cantik
st.markdown("""
    <h1 style='text-align: center; color: #1E88E5;'>
        🧒 Prediksi Risiko Stunting
    </h1>
    <h4 style='text-align: center; color: #555;'>
        LAB IS411 Data Modelling - Kelompok UTS
    </h4>
    <hr>
""", unsafe_allow_html=True)

# Load Model
model = joblib.load('stunting_rf_model.pkl')

# Sidebar Input
with st.sidebar:
    st.header("📋 Input Data Negara")
    st.markdown("Isi data berikut dengan akurat")

    col1, col2 = st.columns(2)
    
    with col1:
        stunting_prev = st.slider("Stunting Prevalence (%)", 0.0, 60.0, 15.0, help="Persentase anak yang mengalami stunting")
        poverty = st.slider("Poverty Rate (%)", 0.0, 100.0, 15.0)
        fertility = st.slider("Fertility Rate", 1.0, 8.0, 2.5)
    
    with col2:
        urban = st.slider("Urban Population (%)", 10.0, 100.0, 60.0)
        gdp = st.number_input("GDP per Capita (USD)", 100, 40000, 5000)
        male_lit = st.slider("Male Literacy Rate (%)", 10.0, 100.0, 90.0)

# Tombol Prediksi yang menonjol
if st.button("🔍 Prediksi Risiko Stunting", type="primary", use_container_width=True):
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
    color_map = {0: "green", 1: "orange", 2: "red"}
    
    st.success(f"**Risiko Stunting: {risk_map[pred]}**", icon="✅")
    
    # Visualisasi Probabilitas
    st.subheader("Probabilitas Prediksi")
    prob_df = pd.DataFrame({
        "Kategori": ["Rendah", "Sedang", "Tinggi"],
        "Probabilitas (%)": [round(p*100, 2) for p in prob]
    })
    st.bar_chart(prob_df.set_index("Kategori"))

# Penjelasan Singkat
st.markdown("---")
st.subheader("📌 Penjelasan Singkat")
st.info("""
    Aplikasi ini memprediksi tingkat risiko stunting menggunakan **Random Forest**.
    Faktor yang paling berpengaruh adalah Stunting Prevalence, Poverty Rate, dan Fertility Rate.
""")

# Dataset
with st.expander("📊 Lihat Dataset"):
    try:
        df = pd.read_csv("Child_Malnutrition_with_Socioeconomic_Factors_dataset.csv")
        st.dataframe(df.head(10), use_container_width=True)
    except:
        st.warning("Dataset tidak ditemukan.")

# Footer
st.caption("Developed by Kelompok UTS | Patricia, Lovely, Evan")
