#importing libraries
import numpy as np
import pandas as pd
import streamlit as st
import pickle
# importing essential pymonngo modules
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# connection string
uri = "mongodb+srv://gauravsingh:gaurav1234@cluster0.thw3pky.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client['Student']
collection = db['student_pred']



# Loading the model by opening .pkl file and unpacking all 3 components
def load_model():
    with open('./Student_performance.pkl','rb') as file:
        model,scaler,encoder = pickle.load(file)
    return model,scaler,encoder


# processing data enter by user so that it can be pass to model
def data_preprocessing(data,scaler,encoder):
    df = pd.DataFrame([data])
    df['Extracurricular Activities'] = encoder.transform(df[['Extracurricular Activities']])
    df_scaled = scaler.transform(df)
    return df_scaled


# prediction after reprocessing
def predict(data,model):
    prediction = model.predict(data)
    return prediction


# main function to run
def main():
    st.title("Prediction of Student Performance.")

    st.write("Please Enter the details to get prediction.")

    # taking input from user
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
        # adding predicted result in user_input so that we got full data to insert in mongodb
        user_input['prediction'] = float(result)
        st.success(f"Predicted Score: {result[0]:.2f}")
    collection.insert_one(user_input)
        

    


if __name__ == '__main__':
    main()
