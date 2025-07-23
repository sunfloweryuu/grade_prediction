import streamlit as st  
import numpy as np
import joblib
import os

# Custom CSS
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 18px;
    }

    .stApp {
        max-width: 100%;
        padding: 2rem 4rem;
    }

    body {
        background-color: #f0f2f6;
    }

    .main-title {
        color: #000000;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }

    .sub-header {
        color: #31333f;
        font-size: 18px;
        text-align: center;
        margin-bottom: 25px;
    }

    div[data-baseweb="input"] input {
        font-size: 18px;
        padding: 0.5em 0.75em;
    }

    div[data-baseweb="input"] input::placeholder {
        font-size: 16px;
    }

    div.stButton > button {
        font-size: 16px;
        padding: 0.5em 2em;
    }

    .prediction-box {
        background-color: #ffffff;
        padding: 20px 30px;
        border-radius: 12px;
        box-shadow: 0px 0px 12px rgba(0,0,0,0.1);
        font-size: 18px;
        width: 100%;
        max-width: 585px;
        margin: 10px auto;
    }

    .stForm {
        background-color: #ffffff;
        padding: 20px 30px;
        border-radius: 12px;
        box-shadow: 0px 0px 12px rgba(0,0,0,0.08);
        margin: 45px auto;
        max-width: 585px;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding: 0.5rem;
    }

    .info-box {
        line-height: 1.5;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Load best model & label encoder
model = joblib.load("models/best_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# Cek apakah model butuh scaler (SVM)
scaler_path = "models/scaler.pkl"
use_scaler = os.path.exists(scaler_path) and "SVC" in str(type(model))
if use_scaler:
    scaler = joblib.load(scaler_path)

# Fungsi konversi huruf ke bobot
def huruf_ke_bobot(huruf):
    mapping = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'E': 0.0}
    return mapping.get(huruf, 0.0)

# Sidebar untuk keterangan
with st.sidebar:
    st.markdown("""
    <div class='info-box'>
    <Hasil>Aplikasi ini digunakan untuk memprediksi nilai mahasiswa berdasarkan input 
    nilai UTM, UAM, UAS, Praktikum-Tugas (Nilai Laporan BBDM), dan BBDM. Hasil prediksi mencakup nilai akhir, 
    nilai huruf, dan bobotnya. Prediksi dilakukan menggunakan model machine learning 
    yang telah dilatih.</p>
    <br>
    <h4>ℹ️ Keterangan</h4>
    <p>📌 <b>A</b> : 80 - 100   → Bobot: 4.0</p>
    <p>📌 <b>B</b> : 70 - 79.99 → Bobot: 3.0</p>
    <p>📌 <b>C</b> : 60 - 69.99 → Bobot: 2.0</p>
    <p>📌 <b>D</b> : 51 - 59.99 → Bobot: 1.0</p>
    <p>📌 <b>E</b> : ≤ 50.99     → Bobot: 0.0</p>
    </div>
    """, unsafe_allow_html=True)

# Title
st.markdown("<div class='main-title'>📊 Prediksi Nilai Mahasiswa</div>", unsafe_allow_html=True)
# st.markdown("<div class='sub-header'>Silahkan masukan nilai di bawah ini.</div>", unsafe_allow_html=True)

# Form Input
with st.form("prediksi_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        utm = st.text_input("Nilai UTM", placeholder="0.00")
    with col2:
        uam = st.text_input("Nilai UAM", placeholder="0.00")
    with col3:
        uas = st.text_input("Nilai UAS", placeholder="0.00")

    praktikum = st.text_input("Nilai Praktikum/Tugas (Nilai Laporan BBDM)", placeholder="0.00")
    bbdm = st.text_input("Nilai BBDM", placeholder="0.00")

    submit = st.form_submit_button("🎯 Prediksi Nilai")

if submit:
    try:
        utm = float(utm)
        uam = float(uam)
        uas = float(uas)
        praktikum = float(praktikum)
        bbdm = float(bbdm)

        # Validasi nilai rentang 0-100
        if any(n < 0 or n > 100 for n in [utm, uam, uas, praktikum, bbdm]):
            st.error("❗ Semua nilai harus berada dalam rentang 0 - 100.")
        else:
            # Feature engineering
            teori_sblm_uas = (utm + uam) / 2
            teori_terbaik = max(teori_sblm_uas, uas)
            nilai_akhir = teori_terbaik * 0.6 + praktikum * 0.3 + bbdm * 0.1

            features = np.array([[utm, uam, uas, praktikum, bbdm, teori_sblm_uas, teori_terbaik, nilai_akhir]])

            if use_scaler:
                features = scaler.transform(features)

            # Prediksi
            pred = model.predict(features)
            huruf_pred = le.inverse_transform(pred)[0]
            bobot_pred = huruf_ke_bobot(huruf_pred)

            st.markdown(f"""
            <div class='prediction-box'>
                <h5>📘 Hasil Prediksi</h5>
                <p><b>Nilai Akhir           :</b> {nilai_akhir:.2f}</p>
                <p><b>Nilai Huruf           :</b> {huruf_pred}</p>
                <p><b>Bobot Nilai           :</b> {bobot_pred}</p>
            </div>
            """, unsafe_allow_html=True)

    except ValueError:
        st.error("❗ Semua nilai harus diisi dengan angka valid.")
