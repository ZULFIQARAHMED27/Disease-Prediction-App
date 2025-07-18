import streamlit as st
import numpy as np
import pickle

# Load models
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
heart_model = pickle.load(open("models/heart_model.pkl", "rb"))

# Sidebar - Navigation
st.sidebar.title("🔍 Disease Selection")
disease = st.sidebar.radio("Choose a disease to predict:", ["Diabetes", "Heart Disease"])

# Sidebar - App Info
st.sidebar.markdown("---")
st.sidebar.write("🧠 **About App**")
st.sidebar.info(
    "This AI-powered app predicts whether a person has **Diabetes** or **Heart Disease** based on medical parameters. "
    "Models used: `Random Forest`."
)

st.title("🩺 Disease Prediction Web App")
st.markdown("Enhancing early diagnosis with machine learning!")

# --------------------------
# 🟩 Diabetes Prediction
# --------------------------
if disease == "Diabetes":
    st.header("Diabetes Prediction")
    st.markdown("**Description:** Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0)
        Glucose = st.number_input("Glucose", min_value=0)
        BloodPressure = st.number_input("Blood Pressure", min_value=0)
        SkinThickness = st.number_input("Skin Thickness", min_value=0)
    with col2:
        Insulin = st.number_input("Insulin", min_value=0)
        BMI = st.number_input("BMI", min_value=0.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        Age = st.number_input("Age", min_value=0)

    if st.button("🔍 Predict Diabetes"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = diabetes_model.predict(input_data)
        result = "🟥 Diabetic" if prediction[0] == 1 else "🟩 Not Diabetic"
        st.success(f"**Prediction:** {result}")

# --------------------------
# 🟥 Heart Disease Prediction
# --------------------------
elif disease == "Heart Disease":
    st.header("Heart Disease Prediction")
    st.markdown("**Description:** Heart disease describes a range of conditions that affect your heart like blocked blood vessels, chest pain, and irregular rhythms.")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=0)
        Sex = st.selectbox("Sex", ["Female", "Male"])
        Cp = st.selectbox("Chest Pain Type", ["0 - Typical Angina", "1 - Atypical Angina", "2 - Non-anginal Pain", "3 - Asymptomatic"])
        Trestbps = st.number_input("Resting Blood Pressure", min_value=0)
        Chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
        Fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    with col2:
        Restecg = st.selectbox("Resting ECG", ["0 - Normal", "1 - ST-T Wave Abnormality", "2 - Left Ventricular Hypertrophy"])
        Thalach = st.number_input("Max Heart Rate Achieved", min_value=0)
        Exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        Oldpeak = st.number_input("ST Depression", min_value=0.0)
        Slope = st.selectbox("Slope of ST Segment", ["0 - Upsloping", "1 - Flat", "2 - Downsloping"])

    if st.button("🔍 Predict Heart Disease"):
        # Encode categorical fields
        sex = 1 if Sex == "Male" else 0
        cp = int(Cp[0])
        fbs = 1 if Fbs == "Yes" else 0
        restecg = int(Restecg[0])
        exang = 1 if Exang == "Yes" else 0
        slope = int(Slope[0])

        input_data = np.array([[Age, sex, cp, Trestbps, Chol, fbs,
                                restecg, Thalach, exang, Oldpeak, slope]])
        prediction = heart_model.predict(input_data)
        result = "🟥 Heart Disease Detected" if prediction[0] == 1 else "🟩 No Heart Disease"
        st.success(f"**Prediction:** {result}")
