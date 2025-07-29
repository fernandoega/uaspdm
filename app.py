import streamlit as st
import joblib

# Load model yang sudah dilatih
model = joblib.load("svm_tiktok_model.joblib")
vectorizer = joblib.load("vectorizer_tiktok.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# UI
st.set_page_config(page_title="Analisis Sentimen TikTok Shop", layout="centered")
st.title("🔍 Analisis Sentimen TikTok Shop")
st.markdown("Masukkan komentar pengguna untuk diprediksi sentimennya.")

# Input
text = st.text_area("Masukkan komentar:")

if st.button("Analisis Sentimen"):
    if text.strip() == "":
        st.warning("Komentar tidak boleh kosong.")
    else:
        vector = vectorizer.transform([text])
        pred = model.predict(vector)[0]
        label = label_encoder.inverse_transform([pred])[0]
        st.success(f"Prediksi Sentimen: **{label.capitalize()}**")