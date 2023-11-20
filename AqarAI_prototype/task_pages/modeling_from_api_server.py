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
from .modeling import show_model_training

def show_api_modeling(token):

    access_token = token
    print(access_token)


    end_point = 'http://127.0.0.1:8000/backend/pattern/'
    headers = {
        'Authorization': 'Token {}'.format(access_token),
        'Content-Type': 'application/json;charset=UTF-8',
        'DN':'3900'
    }
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
    #st.write(df.head())

    st.title("Modeling")
    placeholder_modeling = st.empty()
    with placeholder_modeling.form("Model Selection To Be Trained"):
        model_name = st.radio("Model",('RandomForest','Xgboost','LightGBM','Catboost'))
        features = st.multiselect("Features",features_list,features_list)
        n_estimators = st.slider("n_estimators",min_value=50,max_value=1000,value=100,step=50)
        max_depth = st.slider("max_depth",min_value=1,max_value=10,value=3,step=1)
        min_samples_split = st.slider("min_samples_split",min_value=2,max_value=10,value=3,step=1)
        learning_rate = st.slider("learning_rate",min_value=0.01,max_value=0.5,value=0.02,step=0.01)
        subsample = st.slider("subsample",min_value=0.2,max_value=1.0,value=0.5,step=0.1)
        colsample_bytree = st.slider("colsample_bytree",min_value=0.2,max_value=1.0,value=0.5,step=0.1)
        default_option = st.slider("default_option",min_value=0,max_value=1,value=0,step=1)
        trainingButton = st.form_submit_button("Training")

    if trainingButton:
        print(features)
        df_temp = df.copy()
        features.insert(0,'timestamp')
        features.append('occupancy')
        print(features)
        df_temp = df_temp[features]
        show_model_training(df_temp,model_name,n_estimators,max_depth,min_samples_split,learning_rate,subsample,colsample_bytree,default_option)