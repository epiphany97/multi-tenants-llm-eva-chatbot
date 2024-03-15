
import os
import streamlit as st
import boto3
from streamlit.web.server.websocket_headers import _get_websocket_headers
import json

Sessions_table = os.getenv('SESSIONS_TABLE')
random_num = Sessions_table.split('_')[1]

st.write(f"RANDOM_STRING:{random_num}")
headers = _get_websocket_headers()
tenantid = headers.get("X-Auth-Request-Tenantid")
user_email = headers.get("X-Auth-Request-Email")

sts_client = boto3.client('sts')
account_number = sts_client.get_caller_identity().get('Account')

eks_sa = f'arn:aws:iam::{account_number}:role/multitenant-chatapp-{tenantid}-chatbot-access-role-{random_num}'

def get_sa():
    return eks_sa
