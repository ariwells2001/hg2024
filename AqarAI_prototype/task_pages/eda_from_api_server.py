import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests,json
import numpy as np

def show_eda(token):
    st.title("Exploratory Data Analysis")
    access_token = token
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
    features_list = list(df.drop(["account","timestamp","fp2","occupancy","user"],axis=1).columns)
    df['occupancy'] = np.where(df['occupancy']==0,"non_occupancy","occupancy")
    df['fp2'] = np.where(df['fp2']==0,"non_detection","detection")
    X = df.drop(["account","timestamp","occupancy","user"],axis=1)
    y = df['occupancy']
    df_final = df.drop(["account","user"],axis=1)

    placeholder_graph = st.empty()
    with placeholder_graph.form("EDA"): 
        graph = st.radio(
        "Please choose a plot or a table you would like to explore:",
        ["Data Table", "Box Plot","Histogram","Bar Plot"]
        )
        x_axis = st.selectbox("X_label",features_list)
        y_axis = st.selectbox("Y_label",["fp2","occupancy"])
        edaButton = st.form_submit_button("EDA")

    if edaButton:
        if graph == "Data Table":
            st.dataframe(df.head(20))
        elif graph == "Box Plot":
            fig = plt.figure(figsize=(10,4))
            sns.boxplot(x=y_axis,y=x_axis,data=df_final)
            st.pyplot(fig)
        elif graph == "Histogram":
            fig = plt.figure(figsize=(10,4))
            sns.histplot(x=x_axis,hue=y_axis,data=df_final)
            st.pyplot(fig)
        elif graph == "Bar Plot":
            fig = plt.figure(figsize=(10,4))
            sns.countplot(x=y_axis,data=df_final)
            st.pyplot(fig)
