import streamlit as st
import pickle
import numpy as np

with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("Prediksi Hasil Pertandingan NBA (Menang atau Kalah) 🏀")
st.markdown("Masukkan data statistik pemain untuk memprediksi apakah tim akan menang (1) atau kalah (0).")

mp = st.slider("Menit Bermain (MP)", 0, 48, 30)
fg = st.number_input("Field Goals (FG)", 0.0, 30.0, 5.0)
pts = st.number_input("Points (PTS)", 0.0, 70.0, 20.0)
ast = st.number_input("Assists (AST)", 0.0, 20.0, 5.0)
stl = st.number_input("Steals (STL)", 0.0, 10.0, 1.0)
blk = st.number_input("Blocks (BLK)", 0.0, 10.0, 1.0)

all_teams = list(le.classes_)
team = st.selectbox("Nama Tim", all_teams, index=all_teams.index("LAL") if "LAL" in all_teams else 0)
opponent = st.selectbox("Lawan", all_teams, index=all_teams.index("BOS") if "BOS" in all_teams else 1)

if team in le.classes_:
    encoded_team = le.transform([team])[0]
else:
    st.warning(f"Tim '{team}' tidak dikenali. Pastikan sesuai dataset.")
    encoded_team = 0

if opponent in le.classes_:
    encoded_opp = le.transform([opponent])[0]
else:
    st.warning(f"Tim lawan '{opponent}' tidak dikenali. Pastikan sesuai dataset.")
    encoded_opp = 0

input_data = [
    encoded_team,   # Tm
    encoded_opp,    # Opp
    0,              # Res
    mp,             # MP
    fg,             # FG
    0,              # FGA
    0.0,            # FG%
    0,              # 3P
    0,              # 3PA
    0.0,            # 3P%
    0,              # FT
    0,              # FTA
    0.0,            # FT%
    0,              # ORB
    0,              # DRB
    0,              # TRB
    ast,            # AST
    stl,            # STL
    blk,            # BLK
    0,              # TOV
    0,              # PF
    pts,            # PTS
]

input_array = np.array(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_array)

if st.button("Prediksi"):
    prediction = model.predict(scaled_input)[0]
    result = "Menang (1)" if prediction == 1 else "Kalah (0)"
    st.subheader(f"Hasil Prediksi: **{result}**")