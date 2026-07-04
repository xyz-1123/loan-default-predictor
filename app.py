import streamlit as st
import joblib
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, 'loan_default_model.pkl'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))

st.title("Loan Default Predictor")
st.write("Enter the loan details below to predict if a person will default.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
months_employed = st.number_input("Months Employed", min_value=0, value=24)
num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
loan_term = st.number_input("Loan Term (months)", min_value=0, value=36)
dti_ratio = st.number_input("DTI Ratio", min_value=0.0, value=0.3)
education = st.selectbox("Education", [0, 1, 2, 3])
employment_type = st.selectbox("Employment Type", [0, 1, 2, 3])
marital_status = st.selectbox("Marital Status", [0, 1, 2])
has_mortgage = st.selectbox("Has Mortgage", [0, 1])
has_dependents = st.selectbox("Has Dependents", [0, 1])
loan_purpose = st.selectbox("Loan Purpose", [0, 1, 2, 3, 4])
has_cosigner = st.selectbox("Has Co-Signer", [0, 1])

if st.button("Predict"):
    a = age
    b = income
    c = loan_amount
    d = credit_score
    e = months_employed
    f = num_credit_lines
    g = interest_rate
    h = loan_term
    i = dti_ratio
    j = education
    k = employment_type
    l = marital_status
    m = has_mortgage
    n = has_dependents
    o = loan_purpose
    p = has_cosigner
    input_data = np.array([[a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    if prediction[0] == 1:
        st.error("High Risk! This person is likely to DEFAULT. Probability: " + str(round(probability[0][1]*100, 2)) + "%")
    else:
        st.success("Low Risk! This person is NOT likely to default. Probability: " + str(round(probability[0][0]*100, 2)) + "%")
