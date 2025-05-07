import joblib
import numpy as np
import streamlit as st

# Loading the pickle file of best performing ML-model i.e. XGBoost & the scaled dataset
model= joblib.load('/xgboost_final_model.pkl')
scaler= joblib.load('/Deployment using Django/scaler.pkl')
print(type(scaler))

# Custom CSS
url = "https://www.iriseyehospitals.in/assets/img/treatment/diabetic-retinopathy-screening-and-treatment.webp"
custom_css = f"""
<style>
    .main {{
        background-image: url("{url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }}
</style>
"""
# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)
st.title('Diabetic Retinopathy Risk Predictor')

# Features
age= st.number_input('Age: ', min_value= 0, step= 1)
systolic_bp= st.number_input('Systolic Bp: ', min_value= 0, step= 1)
diastolic_bp= st.number_input('Diastolic Bp: ', min_value= 0, step= 1)
cholestrol= st.number_input('Cholestrol: ', min_value= 0, step= 1)

# Predict Button with animation
if st.button('Predict'):
    # Create a numpy array from the user input
    input_data= np.array([[age, systolic_bp, diastolic_bp, cholestrol]])
    input_data_scaled= scaler.transform(input_data)

    # Make prediction
    probability= model.predict(input_data_scaled)

    # Display the input values
    st.subheader('Input Values: ')
    st.write(f'Age: {age}')
    st.write(f'Systolic Bp: {systolic_bp}')
    st.write(f'Diastolic Bp: {diastolic_bp}')
    st.write(f'Cholestrol: {cholestrol}')

    # Map prediction
    mapping = {0: 'No Retinopathy', 1: 'Retinopathy'}
    result= mapping.get(probability[0], 'Unknown')

    # Display the prediction with an animation\
    st.subheader('Retinopathy Prediction: ')
    st.write(f'The predicted output is: {result}')

    # Add cool animations
    st.balloons()


    