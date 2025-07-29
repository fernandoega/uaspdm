import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.title("Analisis Sentimen TikTok Shop - SVM")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Pilih kolom fitur dan label
    feature_col = st.selectbox("Pilih kolom fitur (teks)", df.columns)
    label_col = st.selectbox("Pilih kolom label (kelas)", df.columns)

    if st.button("Latih Model SVM"):
        from sklearn.feature_extraction.text import TfidfVectorizer

        X_text = df[feature_col].astype(str)
        y = df[label_col].astype(str)

        # Tampilkan distribusi label
        st.write("Distribusi Label:")
        st.write(y.value_counts())

        # Hanya gunakan label yang umum (jika memungkinkan)
        common_labels = ['positif', 'negatif', 'netral']
        df_filtered = df[df[label_col].str.lower().isin(common_labels)]

        if df_filtered.empty:
            st.error("Tidak ada label yang cocok dengan ['positif', 'negatif', 'netral']. Harap bersihkan data.")
        elif df_filtered[label_col].nunique() < 2:
            st.error("Jumlah kelas terlalu sedikit untuk klasifikasi.")
        else:
            X_text = df_filtered[feature_col].astype(str)
            y = df_filtered[label_col].str.lower()

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(X_text)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Latih model SVM
            model = SVC(kernel='linear')
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(ax=ax, cmap='Blues')
            st.pyplot(fig)
