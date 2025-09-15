import streamlit as st
import pandas as pd
import joblib

model = joblib.load("klasifikasi_kualitas_kopi.joblib")

st.title("klasifikasi kualitas kopi")
st.markdown("Prediksi Kualitas Kopi berdasarkan kafein, tingkat keasaman, dan jenis proses")

kadar_kafein = st.slider("kadar kafein",50.0,200.0,110.0)
tingkat_keasaman = st.slider("tingkat keasaman",1.0,7.0,5.0)
jenis_proses = st.pills("jenis proses",["Natural","Honey","Washed"],default="Natural")

if st.button("prediksi",type = "primary"):
	data_baru=pd.DataFrame([[kadar_kafein,tingkat_keasaman,jenis_proses]],columns=["Kadar Kafein","Tingkat Keasaman","Jenis Proses"])
	
	prediksi=model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"prediksi *{prediksi}* presentase *{presentase*100:.2f}%*")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :coffe: oleh saffa ")