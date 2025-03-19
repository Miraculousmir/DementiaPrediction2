import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def show_page():
    # Streamlit app title
    st.header('Predict with demographic data', divider='blue')
    st.subheader('Health Information Input Form')

    # Input fields based on the columns
    diabetic = st.selectbox('Diabetic Status', [1, 0])
    alcohol_level1 = st.text_input("Alcohol Level:", value="0.084973629")
    try:
        # Attempt to convert to float
        alcohol_level = float(alcohol_level1)

    except ValueError:
        # Handle the error if conversion fails
        st.error("Please enter a valid decimal number.")
    heart_rate = st.number_input('Heart Rate', min_value=50, max_value=120, step=1)
    blood_oxygen1 = st.text_input('Blood Oxygen Level (%)', value="96.23074296")
    try:
        # Attempt to convert to float
        blood_oxygen = float(blood_oxygen1)

    except ValueError:
        # Handle the error if conversion fails
        st.error("Please enter a valid decimal number.")
    body_temp1 = st.text_input('Body Temperature (°C)', value="36.22485168")
    try:
        # Attempt to convert to float
        body_temp = float(body_temp1)

    except ValueError:
        # Handle the error if conversion fails
        st.error("Please enter a valid decimal number.")
    weight1 = st.text_input('Weight (kg)', value="57.56397754")
    try:
        # Attempt to convert to float
        weight = float(weight1)

    except ValueError:
        # Handle the error if conversion fails
        st.error("Please enter a valid decimal number.")
    mri_delay1 = st.text_input('MRI Delay ', value="36.42102798")
    try:
        # Attempt to convert to float
        mri_delay = float(mri_delay1)

    except ValueError:
        # Handle the error if conversion fails
        st.error("Please enter a valid decimal number.")
    prescription = st.text_input('Prescription')
    dosage = st.number_input('Dosage (mg)', min_value=0.0, max_value=30.0, step=0.5)
    age = st.number_input('Age', min_value=0, max_value=100, step=1)
    education_level = st.selectbox('Education Level',
                                   ['Primary School', 'Secondary School', 'Diploma/Degree', 'No School'])
    dominant_hand = st.selectbox('Dominant Hand', ['Right', 'Left'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    family_history = st.selectbox('Family History', ['Yes', 'No'])
    smoking_status = st.selectbox('Smoking Status', ['Current Smoker', 'Former Smoker', 'Never Smoked'])
    apoe_e4 = st.selectbox('APOE ε4 Status', ['Positive', 'Negative'])
    physical_activity = st.selectbox('Physical Activity ', ['Sedentary', 'Moderate Activity', 'Mild Activity'])
    depression_status = st.selectbox('Depression Status', ['Yes', 'No'])
    cognitive_scores = st.number_input('Cognitive Test Scores', min_value=0, max_value=10, step=1)
    medication_history = st.selectbox('Medication History', ['Yes', 'No'])
    nutrition_diet = st.selectbox('Nutrition/Diet Quality', ['Low-Carb Diet', 'Mediterranean Diet', 'Balanced Diet'])
    sleep_quality = st.selectbox('Sleep Quality', ['Poor', 'Good', 'Bad'])
    chronic_conditions = st.selectbox('Chronic Health Conditions',
                                      ['Diabetes', 'Heart Disease', 'Hypertension', 'None'])

    with open('rfc.pkl', 'rb') as file:
        rfc = pickle.load(file)

    # Button to process input
    if st.button('Submit'):

        # Displaying a confirmation message and the entered information
        st.success('Submitted Successfully!')
        input_data = np.array([[diabetic,
                                alcohol_level,
                                heart_rate,
                                blood_oxygen,
                                body_temp,
                                weight,
                                mri_delay,
                                prescription,
                                dosage,
                                age,
                                education_level,
                                dominant_hand,
                                gender,
                                family_history,
                                smoking_status,
                                apoe_e4,
                                physical_activity,
                                depression_status,
                                cognitive_scores,
                                medication_history,
                                nutrition_diet,
                                sleep_quality,
                                chronic_conditions]])
        df = pd.DataFrame(input_data, columns=['Diabetic', 'AlcoholLevel', 'HeartRate', 'BloodOxygenLevel',
                                               'BodyTemperature', 'Weight', 'MRI_Delay', 'Prescription',
                                               'Dosage in mg', 'Age', 'Education_Level', 'Dominant_Hand', 'Gender',
                                               'Family_History', 'Smoking_Status', 'APOE_ε4', 'Physical_Activity',
                                               'Depression_Status', 'Cognitive_Test_Scores', 'Medication_History',
                                               'Nutrition_Diet', 'Sleep_Quality', 'Chronic_Health_Conditions'])
        df["Prescription"].fillna("None", inplace=True)
        df["Dosage in mg"].fillna(0, inplace=True)
        df["Chronic_Health_Conditions"].fillna("None", inplace=True)
        cats = ["Prescription", "Education_Level", "Dominant_Hand", "Gender", "Family_History", "Smoking_Status",
                "APOE_ε4",
                "Physical_Activity", "Depression_Status", "Medication_History", "Nutrition_Diet", "Sleep_Quality",
                "Chronic_Health_Conditions"]
        le = LabelEncoder()
        for i in cats:
            df[i] = le.fit_transform(df[i])
        x = df.iloc[:, :].values

        scaling = MinMaxScaler()
        x = scaling.fit_transform(x)

        prediction = rfc.predict(x)
        ans = prediction[0]

        if ans == 1:
            st.write("Dementia")
        else:
            st.write("No Dementia")
