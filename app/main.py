import streamlit as st
import pickle as pickle
import pandas as pd
import numpy as np





def get_clean_data():
    data=pd.read_csv("data/heart.csv")


    data=data.drop(["age","sex"],axis=1)
    
    return data


def add_predictions(input_data):
    model=pickle.load(open("model/model.pkl","rb"))
    scaler=pickle.load(open("model/scaler.pkl","rb"))

    input_array=np.array(list(input_data.values())).reshape(1,-1)
    input_array_scaled=scaler.transform(input_array)
    prediction=model.predict(input_array_scaled)
    

    st.subheader("TEST PREDICTION")

    if prediction[0]==0:
        st.write("<span class='target target1'>NEGATIVE</span>",unsafe_allow_html=True)
    else:
        st.write("<span class='target target2'>POSITIVE</span>",unsafe_allow_html=True)
    
    st.write("Probability of being negative:",model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being positive:",model.predict_proba(input_array_scaled)[0][1])
    st.write("This app assissts medical proffessionals in making the required diagnosis")

def main():
    st.set_page_config(page_title="Heart Disease predictor",
                       page_icon=":heart:",
                       layout="wide",
                       initial_sidebar_state="expanded"
                       )
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html=True)
    

    input_data=add_sidebar()
    
    with st.container():
        st.title("HEART DISEASE PREDICTOR")
        st.subheader("created by AYYAPPADAS M.T. , CSE STUDENT @ AMRITA VISHWAVIDYAPEETHAM")
        st.write("The Heart Disease Predictor ML model is an advanced machine learning tool designed to predict the likelihood of an individual developing heart disease. Utilizing a robust dataset encompassing various health indicators such as age, gender, cholesterol levels, blood pressure, and lifestyle factors, the model leverages sophisticated algorithms to analyze patterns and correlations within the data.")
    col1,col2=st.columns([2,4])

    with col1:
        add_predictions(input_data)
    with col2:
        st.write("")

def add_sidebar():
    st.sidebar.header("TEST INPUTS")
    data=get_clean_data()


    slider_labels=[("Chest Pain Type (cp)","cp"),("Resting Blood Pressure (trestbps)","trestbps"),("Serum Cholestoral (chol)","chol"),("Fasting Blood Sugar > 120 mg/dl (fbs)","fbs"),("Resting Electrocardiographic Results (restecg)","restecg"),("Maximum Heart Rate Achieved (thalach)","thalach"),("Exercise Induced Angina (exang)","exang"),("ST Depression Induced by Exercise (oldpeak)","oldpeak"),("Slope of the Peak Exercise ST Segment (slope)","slope"),("Number of Major Vessels Colored by Fluoroscopy (ca)","ca"),("Thalassemia (thal)","thal")]
    input_dict={}
    for label,key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max())

        )
    return input_dict
    


  






  
if __name__=='__main__':
    main()