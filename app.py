import streamlit as st
import pandas as pd 
import joblib

model = joblib.load("KNN_heart.pkl")
scalers = joblib.load("scaler.pkl")
expected_columns = joblib.load('columns.pkl')


st.title("Heart Stroke Prediction")
st.markdown("Provide the following details ")
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })





    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col]=0

    input_data = input_data[expected_columns]     

    scaled_input = scalers.transform(input_data)  
    prediction = model.predict(scaled_input)[0]

    if prediction==1 :
        st.error("⚠️ Heart Disease Detected")

    else:
        st.success("✅ No Heart Disease Detected")
            





            



    
   