import streamlit as st
import pandas as pd
import joblib

# Load model
model=joblib.load("svm_model.pkl")

st.title("Titanic Survival Prediction using SVM")

pclass=st.selectbox("Passenger Class",[1,2,3])

sex=st.selectbox("Sex",["male","female"])

age=st.number_input("Age",0.0,100.0,25.0)

fare=st.number_input("Fare",0.0,600.0,50.0)

# Convert sex into number
sex=0 if sex=="male" else 1

if st.button("Predict"):

    data=pd.DataFrame({
        "Pclass":[pclass],
        "Sex":[sex],
        "Age":[age],
        "Fare":[fare]
    })

    prediction=model.predict(data)

    if prediction[0]==1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")