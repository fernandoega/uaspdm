import streamlit as st

st.set_page_config(page_title="Analisis Sentimen TikTok Shop", layout="centered")

st.title("🛍️ Analisis Sentimen TikTok Shop")

st.write("Masukkan komentar pengguna di bawah ini untuk melihat prediksi sentimennya:")

# Input teks dari pengguna
user_input = st.text_area("Komentar Pengguna", placeholder="Contoh: Pengiriman cepat dan produknya bagus banget!")

# Tombol untuk analisis
if st.button("Analisis Sentimen"):
    if user_input:
        # Logika simulasi prediksi
        if "bagus" in user_input.lower() or "cepat" in user_input.lower():
            st.success("✅ Sentimen Positif")
        elif "jelek" in user_input.lower() or "lama" in user_input.lower():
            st.error("❌ Sentimen Negatif")
        else:
            st.info("ℹ️ Sentimen Netral / Tidak Teridentifikasi")
    else:
        st.warning("⚠️ Harap masukkan komentar terlebih dahulu.")
