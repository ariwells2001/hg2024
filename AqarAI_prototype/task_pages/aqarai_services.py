import streamlit as st
import os
import requests,json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from imblearn.over_sampling import SMOTE
import joblib

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn.pipeline import make_pipeline
import requests,json

def show_aqarai_service(token):
    st.title("aqarAI Service")

    uploaded_file = st.file_uploader("Choose a model file", type=["joblib"])
    features_list = []    
    if uploaded_file is not None:
        try:
            pipeline_model = joblib.load(uploaded_file)    
        except Exception as e:
            st.error(f"Error: {e}")
    
    placeholder_service = st.empty()
    co2,temperature,humidity,motion,door,fp2,occupancy,predicted = st.columns(8)
    co2 = st.empty()
    temperature = st.empty()
    humidity = st.empty()
    motion = st.empty()
    door = st.empty()
    fp2 = st.empty()
    occupancy = st.empty()
    predicted = st.empty()
    access_token = token
    
    end_point = 'http://127.0.0.1:8000/backend/random/'
    headers = {
        'Authorization': 'Token {}'.format(access_token),
        'Content-Type': 'application/json;charset=UTF-8',
        'DN':'3900'
    }
    
    
    with placeholder_service.form("Monitoring"):
        monitoringButton = st.form_submit_button("Monitoring")

    if monitoringButton:
        if uploaded_file is None:
            print("undefined")
            pipeline_model = joblib.load("./models/xgboost.joblib")

        while(True):
            response = requests.get(url=end_point,headers=headers)
    
            data = json.loads(response.text)
            status = response
            df = pd.DataFrame(data)
            df['id'] = df['id']-1
            print('status is {}'.format(status))
            df = df.sort_values('id',ascending=True)
            df.set_index('id',drop=True,inplace=True)
            df.index.name=None
            df['timestamp'] = df['timestamp'].replace(r'[TZ]',r' ',regex=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df[['co2','door','motion','fp2','occupancy']] = df[['co2','door','motion','fp2','occupancy']].astype(int)
            df[['temperature','humidity']] = df[['temperature','humidity']]/100
            features_list = list(df.drop(["account","timestamp","occupancy","user"],axis=1).columns)
            X_test = df.drop(["account","timestamp","occupancy","user"],axis=1)
            X_test = pd.DataFrame(X_test)
            y_test = df['occupancy']
            co2.text("CO2:"+str(X_test.iloc[0,0]))
            temperature.text("Temperature:"+str(X_test.iloc[0,1]))
            humidity.text("Humidity:"+str(X_test.iloc[0,2]))
            motion.text("Motion:"+str(X_test.iloc[0,3]))
            door.text("Door:"+str(X_test.iloc[0,4]))
            fp2.text("FP2:"+str(int(X_test.iloc[0,5])))
            occupancy.text("Occupancy:"+str(int(y_test)))
            print(f"{X_test}")
            predicted.text("Predicted:"+str(pipeline_model.predict(X_test)))
            time.sleep(10)
   # pred = pipeline_model.predict(X_test[0])
    
    