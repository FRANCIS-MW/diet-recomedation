import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("diet_recommendation_model.pkl")

# Feature list
FEATURES = [
    "Age", "Gender", "Weight_kg", "Height_cm", "BMI", "Disease_Type", "Severity",
    "Physical_Activity_Level", "Daily_Caloric_Intake", "Cholesterol_mg/dL",
    "Blood_Pressure_mmHg", "Glucose_mg/dL", "Dietary_Restrictions", "Allergies",
    "Preferred_Cuisine", "Weekly_Exercise_Hours", "Adherence_to_Diet_Plan",
    "Dietary_Nutrient_Imbalance_Score"
]

# Diet code explanation
diet_labels = {
    0: "High Protein Diet",
    1: "Low Carb Diet",
    2: "Vegetarian Diet",
    3: "Ketogenic Diet",
    4: "Mediterranean Diet"
}

# Streamlit UI
st.title("🍽️ Diet Recommendation System")

st.markdown("### Enter Patient Information Below:")

# Create form
with st.form("diet_form"):
    Age = st.number_input("Age", min_value=0, step=1)
    Gender = st.selectbox("Gender", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    Weight_kg = st.number_input("Weight (kg)", step=1.0)
    Height_cm = st.number_input("Height (cm)", step=1.0)
    BMI = st.number_input("BMI", step=0.1)
    Disease_Type = st.selectbox("Disease Type", [(0, "None"), (1, "Diabetes"), (2, "Hypertension")])[0]
    Severity = st.selectbox("Severity", [(0, "Mild"), (1, "Moderate"), (2, "Severe")])[0]
    Physical_Activity_Level = st.selectbox("Physical Activity Level", [(0, "Low"), (1, "Moderate"), (2, "High")])[0]
    Daily_Caloric_Intake = st.number_input("Daily Caloric Intake (kcal)", step=10.0)
    Cholesterol = st.number_input("Cholesterol (mg/dL)", step=1.0)
    Blood_Pressure = st.number_input("Blood Pressure (mmHg)", step=1.0)
    Glucose = st.number_input("Glucose (mg/dL)", step=1.0)
    Dietary_Restrictions = st.selectbox("Dietary Restrictions", [(0, "None"), (1, "Lactose Intolerant"), (2, "Gluten Free")])[0]
    Allergies = st.selectbox("Allergies", [(0, "None"), (1, "Peanuts"), (2, "Seafood")])[0]
    Preferred_Cuisine = st.selectbox("Preferred Cuisine", [(0, "Kenyan"), (1, "Indian"), (2, "Italian")])[0]
    Weekly_Exercise_Hours = st.number_input("Weekly Exercise Hours", step=1.0)
    Adherence_to_Diet_Plan = st.slider("Adherence to Diet Plan (1-10)", 1, 10)
    Imbalance_Score = st.number_input("Dietary Nutrient Imbalance Score", step=1.0)

    submitted = st.form_submit_button("Submit")

    if submitted:
        input_data = pd.DataFrame([[
            Age, Gender, Weight_kg, Height_cm, BMI, Disease_Type, Severity,
            Physical_Activity_Level, Daily_Caloric_Intake, Cholesterol,
            Blood_Pressure, Glucose, Dietary_Restrictions, Allergies,
            Preferred_Cuisine, Weekly_Exercise_Hours, Adherence_to_Diet_Plan,
            Imbalance_Score
        ]], columns=FEATURES)

        prediction = model.predict(input_data)[0]
        predicted_label = diet_labels.get(int(prediction), "Unknown")

        st.success(f"✅ Your recommended diet code is: {prediction}")
        st.info(f"🔍 Diet type: **{predicted_label}**")
