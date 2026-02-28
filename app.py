import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("iris.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Iris Flower Prediction")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Iris Species: {species[prediction[0]]}")