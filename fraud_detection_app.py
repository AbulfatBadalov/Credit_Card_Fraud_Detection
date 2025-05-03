import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Modeli yükle
model = joblib.load("credit_card_model.pkl")

# Sayfa başlığı
st.title("💳 Kredi Kartı Dolandırıcılığı Tespiti")

st.markdown("""
Bu uygulama, bir işlem bilgisini girerek bu işlemin dolandırıcılık olup olmadığını tahmin eder.
Model: Karar Ağacı (Decision Tree)
""")

# Özellik giriş alanları (V1-V28 + Amount)
st.header("🔢 İşlem Özelliklerini Girin")

input_data = []
for i in range(1, 29):  # V1'den V28'e kadar
    val = st.number_input(f"V{i}", format="%.4f")
    input_data.append(val)

amount = st.number_input("Amount (Tutar)", format="%.2f")
input_data.append(amount)

if st.button("📊 Tahmin Et"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    prediction_proba = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Bu işlem dolandırıcılık olabilir! (İhtimal: {prediction_proba:.2%})")
    else:
        st.success(f"✅ Bu işlem güvenli görünüyor. (İhtimal: {prediction_proba:.2%})")
