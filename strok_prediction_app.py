import streamlit as st
import numpy as np
import pickle

# Load the KNN model and scaler
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to predict stroke using the KNN model
def predict_stroke(model, input_data):
    # Ensure input_data is the same shape as the training data
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Streamlit app
st.title("Stroke Prediction App")

# Create input fields
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 0, 100, 50)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.slider("Avg Glucose Level", 50.0, 300.0, 100.0)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Create a dictionary to map categorical inputs to encoded values
gender_dict = {"Male": 0, "Female": 1, "Other": 2}
binary_dict = {"No": 0, "Yes": 1}
work_type_dict = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
residence_type_dict = {"Urban": 0, "Rural": 1}
smoking_status_dict = {"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3}

# Encode categorical inputs
gender_encoded = gender_dict[gender]
hypertension_encoded = binary_dict[hypertension]
heart_disease_encoded = binary_dict[heart_disease]
ever_married_encoded = binary_dict[ever_married]
work_type_encoded = work_type_dict[work_type]
residence_type_encoded = residence_type_dict[residence_type]
smoking_status_encoded = smoking_status_dict[smoking_status]

# One-hot encoding for categorical variables
input_data = [
    age,
    hypertension_encoded,
    heart_disease_encoded,
    avg_glucose_level,
    bmi,
]

# Append one-hot encoded variables
input_data += [1 if gender_encoded == i else 0 for i in range(3)]
input_data += [1 if ever_married_encoded == i else 0 for i in range(2)]
input_data += [1 if work_type_encoded == i else 0 for i in range(5)]
input_data += [1 if residence_type_encoded == i else 0 for i in range(2)]
input_data += [1 if smoking_status_encoded == i else 0 for i in range(4)]

# Reshape input_data to match scaler input
input_data = np.array(input_data).reshape(1, -1)

# Predict stroke
if st.button("Predict"):
    prediction = predict_stroke(knn_model, input_data)

    if prediction == 1:
        st.error("The model predicts that the person is at risk of a stroke.")
    else:
        st.success("The model predicts that the person is not at risk of a stroke.")
