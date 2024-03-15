import streamlit as st #all streamlit commands will be available through the "st" alias
import os
import requests
from dataclasses import dataclass, field
import boto3
import numpy as np
# from langchain.llms.bedrock import Bedrock
import json
import pytz
# from webapp import write_metadata_to_json
from PIL import Image
import io
import time
from streamlit.web.server.websocket_headers import _get_websocket_headers
import webapp
import s3fs
import plotly.graph_objs as go
# set up the UI


# st.set_page_config(page_title="RAG-LLM-Evaluator") #HTML title
headers = _get_websocket_headers()
tenantid = headers.get("X-Auth-Request-Tenantid")
st.image('aws_logo.png',width=100)
#TODO
#前端提示语


s3=boto3.client('s3')

def check_file_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False

def get_file(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(obj['Body'].read().decode('utf-8'))
    return data

def display_results(data):
    for i, obj in enumerate(data):
        st.write(f"Model ID: {obj['model_id']}")
        st.write(f"Similarity All: {obj['similarity_all']}")
        st.write(f"Mean: {obj['mean']}")
        
bucket = f'model-eva-output-{tenantid}'
folder_name = 'answer-similarity'
# photo_path1='answer_score_per_question.png'
# photo_path2='result_table.png'
key = f'answer-similarity/answer-similarity.json'
# key1 = f'answer-similarity/answer_score_per_question.png'
# key2 = f'answer-similarity/result_table.png'


if st.button('Get Results'):
    with st.spinner('please wait'):
        while not check_file_exists(bucket, key):
            time.sleep(5)  # Wait for 5 seconds before checking again
            st.snow()
    st.write('<span style="color: green;">Results are ready!</span>', unsafe_allow_html=True)
    s3fs = s3fs.S3FileSystem()
    with s3fs.open(f's3://model-eva-output-{tenantid}/answer-similarity/answer-similarity.json') as f:
        data = json.load(f)
        fig = go.Figure()
        for item in data:
            model_id = item['model_id']
            similarity_all = item['similarity_all']
            fig.add_trace(go.Scatter(
                x=list(range(len(similarity_all))), 
                y=similarity_all,
                name=model_id
              ))
            fig.update_layout(
                title='Similarity Comparison',
                xaxis_title='question',
                yaxis_title='Answer Similarity'
            )
            # st.plotly_chart(fig)
            means = [item['mean'] for item in data] 
            var = [item['variance'] for item in data]
            
            model_ids = [item['model_id'] for item in data]
            
            # model_ids = [m.split('.')[0] for m in model_ids]
            model_ids = [m for m in model_ids]  
            layout = go.Layout(
                title='means and vars',
                xaxis=dict(
                    tickvals=[i for i in range(len(data))], 
                    ticktext=model_ids 
              )
            )
            # 添加traces
            fig_2 = go.Figure(layout=layout)
            fig_2.add_trace(go.Bar(x=model_ids, y=means,name="means"))
            fig_2.add_trace(go.Bar(x=model_ids, y=var,name="variance"))
        st.plotly_chart(fig)    
        st.plotly_chart(fig_2)
