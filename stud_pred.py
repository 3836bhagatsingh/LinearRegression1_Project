import numpy as np
import pandas as pd
import streamlit as st
import pickle

def load_model():
    with open('./Student_performance.pkl','rb') as file:
        model,scaler,encoder = pickle.load(file)
    return model,scaler,encoder

def data_preprocessing(data,scaler,encoder):
    df = pd.DataFrame([data])
    df['Extracurricular Activities'] = encoder.transform(df[['Extracurricular Activities']])
    df_scaled = scaler.transform(df)
    return df_scaled

def predict(data,model):
    prediction = model.predict(data)
    return prediction

def main():
    st.title("Prediction of Student Performance.")

    st.write("Please Enter the details to get prediction.")

    user_input = {
        'Hours Studied' : st.number_input('Number of study hours ',min_value = 0,max_value = 15,step=1),
        'Previous Scores': st.number_input('Scores in previous exams',min_value=0,max_value=100,step=1),
        'Extracurricular Activities': st.selectbox('Participation in EC activities ',['Yes','No']),
        'Sleep Hours': st.number_input('Number of sleeping hours',min_value=4,max_value=12,step =1),
        'Sample Question Papers Practiced': st.number_input("Number of PYQ's solved ",min_value=0,max_value=30,step = 1)
    }

    if st.button('Predict'):
        model, scaler, encoder = load_model()
        df_final = data_preprocessing(user_input,scaler,encoder)
        result = predict(df_final,model)
        st.success(f"Predicted Score: {result[0]:.2f}")

    


if __name__ == '__main__':
    main()
