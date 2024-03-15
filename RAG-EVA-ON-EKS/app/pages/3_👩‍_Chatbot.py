"""
A simple web application to implement a chatbot. This app uses Streamlit 
for the UI and the Python requests package to talk to an API endpoint that
implements text generation and Retrieval Augmented Generation (RAG) using
Amazon Bedrock and FAISS as the vector database.
"""
import os
import httpx
import json
from datetime import datetime
import boto3
import streamlit as st
import requests as req
from typing import List, Tuple, Dict
import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers

from sessions import Sessions


# Page title
st.set_page_config(page_title='Multi-tenants Chatbot 👩‍💻', layout='wide')

table_name = os.getenv('SESSIONS_TABLE')

# Create DynamoDB client
dynamodb = boto3.resource("dynamodb")

# global constants
STREAMLIT_SESSION_VARS: List[Tuple] = [("generated", []), ("past", []), ("input", ""), ("stored_session", [])]
HTTP_OK: int = 200

MODE_RAG: str = 'RAG'
MODE_VALUES: List[str] = [MODE_RAG]

TEXT2TEXT_MODEL_LIST: List[str] = ["cohere.command-text-v14", "ai21.j2-ultra-v1","amazon.titan-text-lite-v1","amazon.titan-text-express-v1","anthropic.claude-instant-v1"]
TEXT2TEXT_MODEL_LIST: List[str] = ["anthropic.claude-instant-v1"]
EMBEDDINGS_MODEL_LIST: List[str] = ["amazon.titan-embed-text-v1"]

# API endpoint
api: str = "http://127.0.0.1:8000"
api_rag_ep: str = f"{api}/api/v1/llm/rag"
# api_text2text_ep: str = f"{api}/api/v1/llm/text2text"
print(f"api_rag_ep={api_rag_ep}")

####################
# Streamlit code
####################
# Get request headers
headers = _get_websocket_headers()
tenantid = headers.get("X-Auth-Request-Tenantid")
user_email = headers.get("X-Auth-Request-Email")

tenant_id = tenantid + ":" + user_email
dyn_resource = boto3.resource('dynamodb')
IDLE_TIME = 600                                     # seconds
current_time = int(datetime.now().timestamp())


# keep track of conversations by using streamlit_session
_ = [st.session_state.setdefault(k, v) for k,v in STREAMLIT_SESSION_VARS]


def update_slider():
    st.session_state.slider = st.session_state.numeric
    
def update_numin_k():
    st.session_state.numeric = st.session_state.slidera
def update_numin_p():
    st.session_state.numeric = st.session_state.sliderb
def update_numin_length():
    st.session_state.numeric = st.session_state.sliderc
def update_numin_temperature():
    st.session_state.numeric = st.session_state.sliderd

def get_parmeters_k() -> int:
    # top_k_input = st.sidebar.number_input( "top_k", value=DEFAULT_K,placeholder=DEFAULT_K,key="numeric",on_change=update_slider)
    top_k_sidebar = st.sidebar.slider("Top_k",min_value=1,max_value=500,step=1,key="slidera",
    on_change=update_numin_k)
    
    top_k = top_k_sidebar
    
    return top_k 
def get_parmeters_p() -> float:
    # top_k_input = st.sidebar.number_input( "top_k", value=DEFAULT_K,placeholder=DEFAULT_K,key="numeric",on_change=update_slider)
    top_p_sidebar = st.sidebar.slider("Top_p",min_value=0.1,max_value=1.0,step=0.005,key="sliderb",
    on_change=update_numin_p)
    
    top_p = top_p_sidebar
    return top_p
def get_parmeters_length() -> int:
    # top_k_input = st.sidebar.number_input( "top_k", value=DEFAULT_K,placeholder=DEFAULT_K,key="numeric",on_change=update_slider)
    max_length_sidebar= st.sidebar.slider("Max_length",min_value=1,max_value=2048,step=2,key="sliderc",
    on_change=update_numin_length)
    
    max_length = max_length_sidebar
    return max_length
def get_parmeters_temperature() -> float:
    # top_k_input = st.sidebar.number_input( "top_k", value=DEFAULT_K,placeholder=DEFAULT_K,key="numeric",on_change=update_slider)
    temperature_sidebar= st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,step=0.005,key="sliderd",
    on_change=update_numin_temperature)
    
    temperature = temperature_sidebar
    return temperature
top_p = get_parmeters_p()
top_k = get_parmeters_k()
max_length = get_parmeters_length()
temperature = get_parmeters_temperature()


# Define function to get user input
def get_user_input() -> str:
    """
    Returns the text entered by the user
    """
    print(st.session_state)    
    input_text = st.text_input("You: ",
                               st.session_state["input"],
                               key="input",
                               placeholder="Ask me a question and I will consult the knowledge base to answer...", 
                               label_visibility='hidden')
    return input_text


# sidebar with options
with st.sidebar.expander("⚙️", expanded=True):
    text2text_model = st.selectbox(label='Text2Text Model', options=TEXT2TEXT_MODEL_LIST)
    embeddings_model = st.selectbox(label='Embeddings Model', options=EMBEDDINGS_MODEL_LIST)
    mode = st.selectbox(label='Mode', options=MODE_VALUES)

# streamlit app layout sidebar + main panel
# the main panel has a title, a sub header and user input textbox
# and a text area for response and history
st.title("👩‍💻 Virtual assistant for a knowledge base")
# st.subheader(f" Powered by :blue[{TEXT2TEXT_MODEL_LIST[0]}] for text generation and :blue[{EMBEDDINGS_MODEL_LIST[0]}] for embeddings")
st.markdown(f"**Tenant ID:** :blue[{tenantid}] &emsp; &emsp; **User:** :blue[{user_email}]")

# get user input
user_input: str = get_user_input()

#get model
selected_model = text2text_model
# st.write(model)
# based on the selected mode type call the appropriate API endpoint
if user_input:
    try:
        sessions = Sessions(dyn_resource)
        sessions_exists = sessions.exists(table_name)
        if sessions_exists:
            session = sessions.get_session(tenant_id)
            if session:
                if ((current_time - session['last_interaction']) < IDLE_TIME):
                    sessions.update_session_last_interaction(tenant_id, current_time)
                    updated_session = sessions.get_session(tenant_id)
                    print(updated_session['session_id'])
                else:
                    sessions.update_session(tenant_id, current_time)
                    updated_session = sessions.get_session(tenant_id)
            else:
                sessions.add_session(tenant_id)
                session = sessions.get_session(tenant_id)
    except Exception as e:
        print(f"Something went wrong: {e}")

    # headers for request and response encoding, same for both endpoints
    headers: Dict = {"accept": "application/json",
                     "Content-Type": "application/json"
                    }
    output: str = None
    if mode == MODE_RAG:
        user_session_id = tenant_id + ":" + session["session_id"]
        data = {"q": user_input,"user_session_id": user_session_id, 
        "top_k":top_k,"top_p":top_p,"max_length":max_length,"temperature":temperature,"verbose": True}
        # data = {"q": user_input, "user_session_id": user_session_id, "verbose": True}
        resp = req.post(api_rag_ep, headers=headers, json=data)
        if resp.status_code != HTTP_OK:
            output = resp.text
        else:
            resp = resp.json()
            sources = list(set([d['metadata']['source'] for d in resp['docs']]))
            output = f"{resp['answer']} \n \n Sources: {sources}"
    else:
        print("error")
        output = f"unhandled mode value={mode}"
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(output) 

# download the chat history
download_str: List = []
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="❓") 
        st.success(st.session_state["generated"][i], icon="👩‍💻")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download', download_str)