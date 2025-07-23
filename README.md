# 💡 Implementasi Machine Learning untuk Prediksi Nilai Mahasiswa Berdasarkan Komponen Penilaian Akademik

Aplikasi ini merupakan implementasi dari pendekatan machine learning untuk memprediksi nilai akhir mahasiswa berdasarkan komponen-komponen penilaian akademik, seperti UTM, UAM, UAS, nilai praktikum/tugas (nilai laporan BBDM), dan BBDM. Tujuannya adalah membantu mahasiswa dan pengajar memperoleh gambaran lebih awal mengenai performa akademik mahasiswa.

---

## 🧠 Latar Belakang

Penilaian akhir mahasiswa dalam sistem perkuliahan sering kali didasarkan pada gabungan dari beberapa komponen seperti nilai teori (UTM, UAM, UAS), nilai praktikum, serta BBDM. Dalam praktiknya, banyak mahasiswa kesulitan memperkirakan nilai akhirnya karena setiap komponen memiliki bobot tersendiri.

Berangkat dari permasalahan ini, dibuatlah aplikasi **Grade Prediction** sebagai bagian dari proyek “**Implementasi Machine Learning untuk Prediksi Nilai Mahasiswa Berdasarkan Komponen Penilaian Akademik**” khusunya untuk mahasiswa Kedokteran. Proyek ini bertujuan untuk memanfaatkan machine learning agar dapat:

- Membantu mahasiswa memperkirakan hasil akhirnya lebih objektif
- Memberikan alat bantu cepat bagi dosen dalam evaluasi awal
- Menunjukkan potensi teknologi AI/ML dalam bidang pendidikan

---

## 🎯 Tujuan Aplikasi

- Menggunakan model Machine Learning untuk memprediksi nilai huruf akhir mahasiswa
- Menganalisis bobot kontribusi tiap komponen (teori, praktikum, laporan) terhadap nilai akhir
- Menghasilkan prediksi berbasis data aktual yang telah dilatih sebelumnya
- Memberikan output hasil prediksi dalam bentuk nilai akhir, huruf, dan bobot akademik

---

## 🚀 Fitur Aplikasi

- Form input nilai:
  - UTM
  - UAM
  - UAS
  - Praktikum / Tugas (Nilai Laporan BBDM)
  - BBDM
- Perhitungan nilai akhir berbasis formula:
  Nilai Akhir = 60% teori terbaik + 30% praktikum + 10% BBDM
- Model ML untuk prediksi nilai huruf (A, B, C, D, E)
- Konversi nilai huruf ke bobot skala 4.0
- GUI interaktif menggunakan **Streamlit**

---

## 🛠️ Teknologi yang Digunakan

- **Python 3**
- **Streamlit** – membangun tampilan GUI web
- **scikit-learn** – training model ML
- **NumPy** – perhitungan numerik
- **joblib** – serialisasi model
- **GitHub + Streamlit Cloud** – deployment

---
