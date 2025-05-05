import os
import subprocess
import time

import pandas as pd
import requests
import streamlit as st

venv_python = os.path.join('/', 'home', 'adminuser', 'venv', 'bin', 'python')
subprocess.Popen([venv_python, "server.py"])
time.sleep(2)

st.set_page_config(layout="wide")
st.title("SHL Assessment Recommendation System")
debug_mode = False


def get_response(input_text):
    response = requests.post("http://localhost:8003/recommend", json={'query': input_text})
    return response.json()


input_text = st.text_input("Write your query!", key='user_input')

if input_text:
    st.markdown('**Recommended Assessments**')
    response = get_response(input_text)
    df = pd.json_normalize(response['recommended_assessments'])
    columns = ['name', 'remote_support', 'adaptive_support', 'duration', 'test_type', 'url']
    df_table = df[columns]

    df_table = df_table.rename(columns={
        'name': 'Assessment Name',
        'remote_support': 'Remote Testing',
        'adaptive_support': 'Adaptive/IRT',
        'duration': 'Test Duration (in min)',
        'test_type': 'Test Type',
        'url': 'Assessment Link'
    })

    st.dataframe(df_table, use_container_width=True, hide_index=True, column_config= {
        "Assessment Link" : st.column_config.LinkColumn()
    })

    if debug_mode:
        print('Response ', response)