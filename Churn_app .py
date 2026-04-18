import streamlit as st
import pandas as pd
import joblib

# Load the model from the ipynb file

model = joblib.load("best_churn_pipeline.pk1")

st.title("Customer Churn Prediction")

st.write("Enter customer details:")


# input the feature u want tk get the output of

credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure (years)", 0, 20, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.number_input("Number of Products", 1, 4, 1)
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])


# predict button
if st.button("Predict"):
    data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'EstimatedSalary': [salary],
        'Geography': [geography],
        'Gender': [gender],
        'HasCrCard': [has_card],
        'IsActiveMember': [is_active]
    })

    import numpy as np

    # balance per product
    data['balance per product'] = data['Balance'] / data['NumOfProducts'].replace(0, np.nan)
    data['balance per product'].fillna(0, inplace=True)

    # salary ratio
    data['salary_balance_ratio'] = data['EstimatedSalary'] / data['Balance'].replace(0, np.nan)
    data['salary_balance_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    data['salary_balance_ratio'].fillna(0, inplace=True)

    # age group
    bins=[0,25,35,45,55,65,100]
    labels=['<25','25-31','35-44','45-54','55-64','65+']
    data['age_group']=pd.cut(data['Age'], bins=bins, labels=labels)

    # tenure buckets
    data['tenure_buckets']=pd.cut(
        data['Tenure'],
        bins=[-1,0,2,5,10,100],
        labels=['0','1-2','3-5','6-10','10+']
    )

    # high balance (use fixed threshold!)
    data['high_balance'] = (data['Balance'] > 100000).astype(int)

    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(f"Customer will churn ⚠️ (Probability: {prob:.2f})")
    else:
        st.success(f"Customer will stay ✅ (Probability: {prob:.2f})")