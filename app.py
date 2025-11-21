import streamlit as st
import pickle
import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Sentimen Analisis Super",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# 2. LOAD MODEL & ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    try:
        # Pastikan nama file ini SAMA PERSIS dengan yang didownload dari Colab
        model = pickle.load(open('model_sentiment.pkl', 'rb'))
        vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_assets()

# Inisialisasi Preprocessing (Sama seperti saat training)
stopword = StopWordRemoverFactory().create_stop_word_remover()

def clean_text_input(text):
    # Lowercase
    text = text.lower()
    # Hapus simbol
    text = re.sub(r'[^a-z\s]', '', text)
    # Hapus huruf berulang (cth: baguuus -> bagus)
    text = re.sub(r'(.)\1+', r'\1', text)
    # Hapus stopword
    text = stopword.remove(text)
    return text

# ==========================================
# 3. TAMPILAN UI (INTERFACE)
# ==========================================
st.title("🧠 Analisis Sentimen Ulasan")
st.markdown("Aplikasi cerdas untuk mendeteksi sentimen **Positif** atau **Negatif** menggunakan *Ensemble Learning* (Voting Classifier).")
st.write("---")

# Area Input Data
user_input = st.text_area("📝 Masukkan Ulasan Produk di sini:", height=150, placeholder="Contoh: Barangnya bagus banget, pengiriman cepat!")

# Tombol Prediksi
if st.button("🔍 Analisis Sekarang"):
    if model is None:
        st.error("⚠️ File model belum ditemukan! Pastikan 'model_sentiment.pkl' dan 'tfidf_vectorizer.pkl' ada satu folder dengan app.py.")
    elif user_input.strip() == "":
        st.warning("⚠️ Harap masukkan teks ulasan terlebih dahulu.")
    else:
        # --- PROSES PREDIKSI ---
        
        # 1. Bersihkan teks input user
        clean_input = clean_text_input(user_input)
        
        # 2. Ubah teks ke angka (Vektorisasi)
        vec_input = vectorizer.transform([clean_input])
        
        # 3. Prediksi
        prediction = model.predict(vec_input)[0]
        proba = model.predict_proba(vec_input)
        confidence = float(max(proba[0])) * 100

        # --- TAMPILKAN HASIL ---
        st.write("---")
        st.subheader("Hasil Prediksi:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Jika Prediksi 1 (Positif)
            if prediction == 1:
                st.success("### 😊 POSITIF")
                st.metric("Akurasi Keyakinan", f"{confidence:.2f}%")
            # Jika Prediksi 0 (Negatif)
            else:
                st.error("### 😡 NEGATIF")
                st.metric("Akurasi Keyakinan", f"{confidence:.2f}%")

        with col2:
            st.text("Data yang diproses model:")
            st.code(clean_input)

# Sidebar Informasi
st.sidebar.header("Tentang Model")
st.sidebar.info("Model ini menggabungkan **Naive Bayes** dan **Logistic Regression** untuk mencapai akurasi di atas 90%.")
