import streamlit as st
from task_pages.eda_from_api_server import show_eda
from task_pages.modeling_from_api_server import show_api_modeling
from task_pages.modeling_from_csv_file import show_csv_modeling
from task_pages.aqarai_services import show_aqarai_service
from task_pages.blank import show_blank
from apscheduler.schedulers.background import BackgroundScheduler

placeholder = st.sidebar.empty()

with placeholder.form("Access_Token"):
    access_token = st.text_input("Enter your access token!",type="password")
    tokenButton = st.form_submit_button("Access_Token")

if tokenButton:
    print("The token has been passed")
    # sched = BackgroundScheduler()
    # sched.add_job(show_aqarai_service,'interval',seconds=10,id="myid_3",args=[access_token])
    # sched.start()

task = st.sidebar.radio(
    "Please choose a task you would like to do:",
    ["None","Exploratory Data Analysis","Modeling_from_API_Server", "Modeling_from_CSV_File","aqarAI Service"]
)


if task == "None":
    show_blank()
elif task == "Exploratory Data Analysis":
    show_eda(access_token)
elif task == "Modeling_from_API_Server":
    show_api_modeling(access_token)
elif task == "Modeling_from_CSV_File":
    show_csv_modeling()
elif task == "aqarAI Service":
    show_aqarai_service(access_token)
