import streamlit as st
import pandas as pd
import joblib

model = joblib.load('random_forest_model1.pkl')
model_columns = joblib.load('model_columns2.pkl')


st.title('Heart Disease Risk Prediction')


age = st.number_input('Age', min_value=0, max_value=120, value=50)
sex = st.selectbox('Sex', options=['male', 'female'])
cigsPerDay = st.number_input('Cigarettes per day', min_value=0, value=0)
totChol = st.number_input('Total cholesterol', min_value=0, value=200)
sysBP = st.number_input('Systolic BP', min_value=0, value=120)
diaBP = st.number_input('Diastolic BP', min_value=0, value=80)
BMI = st.number_input('Body Mass Index (BMI)', min_value=0.0, value=25.0)
heartRate = st.number_input('Heart rate', min_value=0, value=70)
glucose = st.number_input('Glucose', min_value=0, value=80)


sex = 1 if sex == 'male' else 0


input_data = pd.DataFrame([[age, sex, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose]], 
                          columns=['age', 'sex', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'])


input_data = input_data.reindex(columns=model_columns, fill_value=0)


if st.button('Predict'):
    
    prediction = model.predict(input_data)
    
    
    if prediction[0] == 1:
        st.write('Predicted 10-year risk of coronary heart disease: **Yes**')
    else:
        st.write('Predicted 10-year risk of coronary heart disease: **No**')
