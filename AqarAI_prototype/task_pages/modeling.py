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

def show_model_training(dataset,model_name,n_estimators,max_depth,min_samples_split,learning_rate,subsample,colsample_bytree,default_option):
    st.header(f"Training {model_name} using Aqara Dataset")
    if default_option:
        st.subheader(f"Using default options for {model_name}")
    if not default_option:
        rf_options = [n_estimators,max_depth,min_samples_split]
        rf_options_labels = ["n_estimators","max_depth","min_samples_split"]
        xgb_options = [n_estimators,max_depth,learning_rate,subsample,colsample_bytree]
        xgb_options_labels = ["n_estimators","max_depth","learning_rate","subsample","colsample_bytree"]
        lgbm_options = [n_estimators,max_depth,learning_rate,subsample,colsample_bytree]
        lgbm_options_labels = ["n_estimators","max_depth","learning_rate","subsample","colsample_bytree"]
        catboost_options = [n_estimators,max_depth,learning_rate,subsample]
        catboost_options_labels = ["n_estimators","max_depth","learning_rate","subsample"]
        if model_name == "RandomForest":
            st.subheader(f"{model_name} options:")
            st.subheader(f"{rf_options_labels[0]}: {rf_options[0]}, \
                         {rf_options_labels[1]}: {rf_options[1]},{rf_options_labels[2]}: {rf_options[2]}" )
        elif model_name == "Xgboost":
            st.subheader(f"{model_name} options:")
            st.subheader(f"{xgb_options_labels[0]}: {xgb_options[0]}, \
                         {xgb_options_labels[1]}: {xgb_options[1]},{xgb_options_labels[2]}: {xgb_options[2]} \
                         {xgb_options_labels[3]}: {xgb_options[3]},{xgb_options_labels[4]}: {xgb_options[4]}" )
        elif model_name == "LightGBM":
            st.subheader(f"{model_name} options:")
            st.subheader(f"{lgbm_options_labels[0]}: {lgbm_options[0]}, \
                         {lgbm_options_labels[1]}: {lgbm_options[1]},{lgbm_options_labels[2]}: {lgbm_options[2]} \
                         {lgbm_options_labels[3]}: {lgbm_options[3]},{lgbm_options_labels[4]}: {lgbm_options[4]}" ) 
        elif model_name == "Catboost":
            st.subheader(f"{model_name} options:")
            st.subheader(f"{catboost_options_labels[0]}: {catboost_options[0]}, \
                         {catboost_options_labels[1]}: {catboost_options[1]}, \
                         {catboost_options_labels[2]}: {catboost_options[2]},{catboost_options_labels[3]}: {catboost_options[3]}")

    df = dataset.copy()
    X = df.drop(['timestamp','occupancy'],axis=1)
    y = df['occupancy']
    temp = list(X.columns)
    columns_list = '_'.join(temp)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,shuffle=False)


    smote = SMOTE(random_state=2023,k_neighbors=3)
    X_smote,y_smote = smote.fit_resample(X_train,y_train)
    if default_option:
        rf_pipeline = make_pipeline(
            StandardScaler(),
            RandomForestClassifier()
        )
        
        xgb_pipeline = make_pipeline(
            StandardScaler(),
            XGBClassifier()
        )
        
        lgbm_pipeline = make_pipeline(
            StandardScaler(),
            LGBMClassifier()
        )

        catboost_pipeline = make_pipeline(
            StandardScaler(),
            CatBoostClassifier()
        )

    if not default_option:
        rf_pipeline = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=rf_options[0],
                                   max_depth=rf_options[1],
                                   min_samples_split=rf_options[2]
                                   )
        )
        
        xgb_pipeline = make_pipeline(
            StandardScaler(),
            XGBClassifier(n_estimators=xgb_options[0],
                          max_depth=xgb_options[1],
                          learning_rate=xgb_options[2],
                          subsample=xgb_options[3],
                          colsample_bytree=xgb_options[4])
        )
        
        lgbm_pipeline = make_pipeline(
            StandardScaler(),
            LGBMClassifier(n_estimators=lgbm_options[0],
                          max_depth=lgbm_options[1],
                          learning_rate=lgbm_options[2],
                          subsample=lgbm_options[3],
                          colsample_bytree=lgbm_options[4])
        )

        catboost_pipeline = make_pipeline(
            StandardScaler(),
            CatBoostClassifier(iterations=catboost_options[0],
                          depth=catboost_options[1],
                          learning_rate=catboost_options[2],
                          subsample=catboost_options[3])
        )
    folder_for_models = './models/'
    saved_model =  columns_list + "_" + str(datetime.now().year) + "{:02d}".format(datetime.now().month) + "{:02d}".format(datetime.now().day)+"_" + model_name + ".joblib"

    if model_name == "RandomForest":
        rf_pipeline.fit(X_smote,y_smote)
        # st.subheader(rf_pipeline.score(X_test,y_test))
        print(f"{model_name} selected")
        pred = rf_pipeline.predict(X_test)
        joblib.dump(rf_pipeline,os.path.join(folder_for_models,saved_model))
    elif model_name == "Xgboost":
        xgb_pipeline.fit(X_smote,y_smote)
        # st.subheader(xgb_pipeline.score(X_test,y_test))
        print(f"{model_name} selected")
        pred = xgb_pipeline.predict(X_test)
        joblib.dump(xgb_pipeline,os.path.join(folder_for_models,saved_model))

    elif model_name == "LightGBM":
        lgbm_pipeline.fit(X_smote,y_smote)
        # st.subheader(lgbm_pipeline.score(X_test,y_test))
        print(f"{model_name} selected")
        pred = lgbm_pipeline.predict(X_test)
        joblib.dump(lgbm_pipeline,os.path.join(folder_for_models,saved_model))

    elif model_name == "Catboost":
        catboost_pipeline.fit(X_smote,y_smote)
        # st.subheader(catboost_pipeline.score(X_test,y_test))
        print(f"{model_name} selected")
        pred = catboost_pipeline.predict(X_test)
        joblib.dump(catboost_pipeline,os.path.join(folder_for_models,saved_model))


    tn,fp,fn,tp = confusion_matrix(y_test,pred).ravel()
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    npv = tn/(tn+fn)
    f1_score = 2*precision*sensitivity/(precision+sensitivity)
    st.subheader('--------------------------------------------------')
    st.header('Confusion_Matrix:')
    st.subheader(f"{confusion_matrix(y_test,pred)[0]}")
    st.subheader(f"{confusion_matrix(y_test,pred)[1]}")
    st.subheader('--------------------------------------------------')
    st.subheader(f'Accuracy = {accuracy}')
    st.subheader(f'Precision = {precision}')
    st.subheader(f'Sensitivity(Recall) = {sensitivity}')
    st.subheader(f'Specificity = {specificity}')
    st.subheader(f'NPV = {npv}')
    st.subheader(f'F1_score = {f1_score}')
    st.subheader('-----------------------------------------------')
    st.subheader('\n')