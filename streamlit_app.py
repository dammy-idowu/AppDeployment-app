# Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier

# Inserting header for app
st.title(":reminder_ribbon: Breast Cancer Prediction app :reminder_ribbon:")
st.info("Early detection is key to positive prognosis. Click on the sidebar icon to make breast cancer prediction")

# Importing dataset
with st.expander('Breast Cancer Data'):
    st.write('**Raw data**')
    df = pd.read_csv("https://raw.githubusercontent.com/dammy-idowu/AppDeployment-app/refs/heads/main/streamlit_data.csv")
    df

# Splitting dataset into independent and dependent variable    
    st.write('Independent features')
    X_raw = df.drop(['diagnosis_result'], axis=1)
    X_raw
    
    st.write('**y**')
    y_raw = df.diagnosis_result
    y_raw

# Input features
#year, age, menopause, tumorsize, invnodes,breast, metastasis,breastquadrant, history,diagnosis_result
# Data preparations
with st.sidebar:
    st.header('Input features')
    year = st.selectbox('Year', (2019,2020,2021,2022,2023,2024,2025))
    age = st.slider('Age', 13, 77, 44)
    menopause = st.selectbox('Menopause', (0, 1))
    tumorsize = st.slider('Tumor Size', 1, 12, 6)
    invnodes = st.selectbox('Inv nodes', (0, 1, 3))
    breast = st.selectbox('Breast position', ('Right', 'Left'))
    metastasis = st.selectbox('Metastasis', (0, 1))
    breastquadrant = st.selectbox('Breast Quadrant', ('Upper inner','Upper outer', 'Lower outer', 'Lower inner'))
    history = st.selectbox('History', (0, 1))

# Create a DataFrame for the input features
    data = {'year': year,
            'age': age,
            'menopause': menopause,
            'tumorsize': tumorsize,
            'invnodes': invnodes,
            'breast': breast,
            'metastasis': metastasis,
            'breastquadrant': breastquadrant,
            'history': history}
    input_df = pd.DataFrame(data, index=[0])
    input_bcdata = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
    st.write('**Features (X)**')
    input_df
    st.write('**Merged (original and encoded) Breast cancer data**')
    input_bcdata

    ## Data preprocessing
    # Encode X
    encode = ['breast', 'breastquadrant']
    df_bcdata = pd.get_dummies(input_bcdata, prefix=encode)
    df_bcdata[:1]
    # input_row_drop = df_bcdata[:1]
    # Dropping irrelevant 'extra Upper outer' column
    # input_row = input_row_drop.drop(['breastquadrant_Upper outer '], axis=1)
    input_row = df_bcdata[:1]
    X = df_bcdata[1:]

    # Encode Y
    target_mapper = {'Malignant': 1,'Benign': 0}
    def target_encode(val):
        return target_mapper[val]
        
    y = y_raw.apply(target_encode)
    #y
    #y_raw


with st.expander('Data preprocessing'):
    st.write('**Encoded independent feature(X)**')
    input_row
    st.write('**Encoded dependent feature (y)**')
    y


# Model training and inference
## Train the ML model
clf = GradientBoostingClassifier()
clf.fit(X, y)

# Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Viewing prediction outcome
df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Benign', 'Malignant']
df_prediction_proba.rename(columns={0: 'Benign', 1: 'Malignant'})

# Prediction summary preparation
st.subheader('**Predicted Outcome**')
st.dataframe(df_prediction_proba,
             column_config={
                 'Benign': st.column_config.ProgressColumn(
                     'Benign',
                     format='%.2f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
                 'Malignant': st.column_config.ProgressColumn(
                     'Malignant',
                     format='%.2f',
                     width='medium',
                     min_value=0,
                     max_value=1
                 ),
             }, hide_index=True)

# Prediction outcome
prediction_outcome = np.array(['Benign', 'Malignant'])
st.success(str(prediction_outcome[prediction][0]))
prediction_outcome_benign = prediction_outcome.tolist()
prediction_benign = prediction_outcome_benign[0]
prediction_malignant = prediction_outcome_benign[1]
st.write(" :round_pushpin: Tap icon  for recommendation tips:round_pushpin: ")

# Button to submit user input
if st.button("Recommendation"):
    # Success message
    st.success("Recommendation generated successfully!")
    
    # Display recommendation
    st.write("Based on the app prediction, we recommend:")
    
    if prediction_benign in str(prediction_outcome[prediction][0]):
        st.success("Regular self and clinical breast examinations be conducted to monitor breast cancer health risk. Also, healthy lifestyle practises such as maintaining healthy weight, exercising, are strongly recommended in order to attenuate mechanisms that may promote breast cancer carcinogenesis. Kia Ora")
    else:
        if prediction_malignant in (str(prediction_outcome[prediction][0])):
            st.success("You are fine :smile:, this is just a demo app. However, adopt healthy lifestyle practises and conduct regular self/clinical breast cancer examination checks. Consult a physician if you observe unusual symptoms in your breast. Kia Ora")

