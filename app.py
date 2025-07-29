import streamlit as st
import pickle

# Load model yang sudah dilatih
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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