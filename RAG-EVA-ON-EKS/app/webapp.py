# !/usr/bin/env python3
import pandas as pd
import streamlit as st
import json
import os
import boto3
import io
import pprint
import time
from utility import create_bedrock_execution_role, create_oss_policy_attach_bedrock_execution_role,create_policies_in_oss
import random
from opensearchpy import OpenSearch, RequestsHttpConnection,AWSV4SignerAuth
from retrying import retry
from typing import List, Tuple, Dict
import pytz
from datetime import datetime
from collections import OrderedDict
from streamlit.web.server.websocket_headers import _get_websocket_headers

headers = _get_websocket_headers()
tenantid = headers.get("X-Auth-Request-Tenantid")
user_email = headers.get("X-Auth-Request-Email")

st.session_state['suffix'] = random.randint(200, 900)

st.write("# Welcome to RAG-LLM-Evaluator! 👋")


# only for test
role_arn = os.environ.get('AWS_ROLE_ARN')
# st.write(f"role_Arn:{role_arn}")


tenant_id = tenantid + ":" + user_email
st.markdown(f"**Tenant ID:** :blue[{tenantid}] &emsp; &emsp; **User:** :blue[{user_email}]")
st.sidebar.success("Please choose function from sidebar")

TEXT2TEXT_MODEL_LIST: List[str] = ["anthropic.claude-instant-v1"]
EMBEDDINGS_MODEL_LIST: List[str] = ["amazon.titan-embed-text-v1"]

with st.expander("📌Automatic model evaluate "):
    st.write("📢  \
    1. Upload your file(pdf,txt) to build knowledge base \
    2. Choose LLM \
    3. Upload your Questions and Answers file (json) \
    4. Press 'Get Results' button, get the rank and score" )

with st.expander("📌Human evaluate"):
    st.write("📢 \
    1. We combine the Questions and Answers files \
    2. For one question, has different Answers \
    3. Human give their queue")
    
with st.expander("📌Chatbot"):
    st.markdown(f"📢 \
    Powered by :blue[{TEXT2TEXT_MODEL_LIST[0]}] for text generation and :blue[{EMBEDDINGS_MODEL_LIST[0]}] for embeddings")

s3 = boto3.client('s3')
def check_file_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False
    
    
        

uploaded_rag_file = st.file_uploader("Choose A File Use For Rag",type=["txt", "pdf"])
if uploaded_rag_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_rag_file.getvalue()
    
    bucket_name=f'rag-input-{tenantid}'
    folder_name='build-knowledge-base'
    s3.upload_fileobj(
        Fileobj=io.BytesIO(bytes_data), 
        Bucket=bucket_name,
        Key=f'{folder_name}/{uploaded_rag_file.name}'
    )
    
    
    if st.button("Upload RAG File"):
        st.write('<span style="color: green;">Your RAG File Upload Success!</span>', unsafe_allow_html=True)
        
        start_time = time.time()
        with st.spinner('Building knowledge base'):
            #Step 1 - Create OSS policies and collection
            if 'suffix' not in st.session_state:
                st.session_state['suffix'] = random.randint(200, 900)
            # suffix = random.randrange(200, 900)
            st.session_state['suffix'] = random.randint(200, 900)
            suffix = st.session_state['suffix']
            boto3_session = boto3.session.Session()
            region = boto3_session.region_name
            bedrock_agent_client = boto3_session.client('bedrock-agent', region_name=region)
            service = 'aoss'
            s3 = boto3.client('s3')
            sts_client = boto3.client('sts')
            account_id = sts_client.get_caller_identity()["Account"]
            bucket_name = f'rag-input-{tenantid}'
            
            vector_store_name = f'bedrock-sample-rag-{suffix}'
            index_name = f"bedrock-sample-rag-index-{suffix}"
            aoss_client = boto3_session.client('opensearchserverless')
            
            iam = boto3.client("iam")
            # 创建执行角色和策略
            
            bedrock_kb_execution_role = f"AmazonBedrockFoundationModelPolicyForKnowledgeBase_{suffix}"
            response = {}
            def check_policy_exists(bedrock_kb_execution_role):
                iam = boto3.client("iam")
                arn = ""
                try:
                    response = iam.get_policy(
                  PolicyArn=bedrock_kb_execution_role
                    )
                    arn = response['Policy']['Arn']
                    st.write(f"Policy {bedrock_kb_execution_role} exists. ARN is {arn}")
                except Exception as e:
                    pass
                return arn
                
             
            policy_arn = check_policy_exists(bedrock_kb_execution_role)
            if policy_arn:
                st.write(f"Policy already exists: {policy_arn}")
                bedrock_kb_execution_role = f"AmazonBedrockFoundationModelPolicyForKnowledgeBase_{suffix}"
                # return policy_arn
            else:
                bucket_name = f'rag-input-{tenantid}'
                bedrock_kb_execution_role = create_bedrock_execution_role(bucket_name)
                bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
            encryption_policy, network_policy, access_policy = create_policies_in_oss(
                vector_store_name=vector_store_name,
                aoss_client=aoss_client,
                bedrock_kb_execution_role_arn=bedrock_kb_execution_role_arn
            )
            
            bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']
            collection = aoss_client.create_collection(name=vector_store_name,type='VECTORSEARCH')
            
            collection_id = collection['createCollectionDetail']['id']
            host = collection_id + '.' + region + '.aoss.amazonaws.com'
            # st.write(f"test2: host:{host}")
            
            response = aoss_client.batch_get_collection(names=[vector_store_name])
            
            # 周期性检查collection状态
            while (response['collectionDetails'][0]['status']) == 'CREATING':
                print('Creating collection...')
                time.sleep(30)
                response = aoss_client.batch_get_collection(names=[vector_store_name])
            st.write("<span style='color:green'>Collection successfully created</span>", unsafe_allow_html=True)
            
            print(response["collectionDetails"])
            
            create_oss_policy_attach_bedrock_execution_role(collection_id=collection_id,
                                                            bedrock_kb_execution_role=bedrock_kb_execution_role)
            #   Create vector index
            credentials = boto3.Session().get_credentials()
            awsauth = AWSV4SignerAuth(credentials,region, 'aoss')
            
            index_name = f"bedrock-sample-index-{suffix}"
            body_json = {
              "settings": {
                  "index.knn": "true",
                  "number_of_shards": 1,
                  "knn.algo_param.ef_search": 512,
                  "number_of_replicas": 0,
              },
              "mappings": {
                  "properties": {
                     "vector": {
                        "type": "knn_vector",
                        "dimension": 1536,
                         "method": {
                             "name": "hnsw",
                             "engine": "nmslib",
                             "space_type": "cosinesimil",
                             "parameters": {
                                 "ef_construction": 512,
                                 "m": 16
                             },
                         },
                     },
                     "text": {
                        "type": "text"
                     },
                     "text-metadata": {
                        "type": "text"         }
                  }
              }
            }
            # Build the OpenSearch client
            oss_client = OpenSearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=300
            )
            # # It can take up to a minute for data access rules to be enforced
            time.sleep(60)
            response = oss_client.indices.create(index=index_name, body=json.dumps(body_json))
            st.write("<span style='color:green'>\nCreating index...</span>", unsafe_allow_html=True)
            print(response)
            time.sleep(60) # index creation can take up to a minute
                #upload files
                # s3_client.upload_file("./",bucket_name,)
            
            opensearchServerlessConfiguration = {
                        "collectionArn": collection["createCollectionDetail"]['arn'],
                        "vectorIndexName": index_name,
                        "fieldMapping": {
                            "vectorField": "vector",
                            "textField": "text",
                            "metadataField": "text-metadata"
                        }
                    }
            
            chunkingStrategyConfiguration = {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 512,
                    "overlapPercentage": 20
                }
            }
            
            s3Configuration = {
                "bucketArn": f"arn:aws:s3:::{bucket_name}",
                "inclusionPrefixes":["build-knowledge-base"] # you can use this if you want to create a KB using data within s3 prefixes.
            }
            
            embeddingModelArn = f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v1"
            
            name = f"bedrock-sample-knowledge-base-{suffix}"
            description = "User's personal knowledge base."
            roleArn = bedrock_kb_execution_role_arn
            # Create a KnowledgeBase
            
            @retry(wait_random_min=1000, wait_random_max=2000,stop_max_attempt_number=7)
            def create_knowledge_base_func():
                create_kb_response = bedrock_agent_client.create_knowledge_base(
                    name = name,
                    description = description,
                    roleArn = roleArn,
                    knowledgeBaseConfiguration = {
                        "type": "VECTOR",
                        "vectorKnowledgeBaseConfiguration": {
                            "embeddingModelArn": embeddingModelArn
                        }
                    },
                    storageConfiguration = {
                        "type": "OPENSEARCH_SERVERLESS",
                        "opensearchServerlessConfiguration":opensearchServerlessConfiguration
                    }
                )
                return create_kb_response["knowledgeBase"]
            try:
                kb = create_knowledge_base_func()
                st.write("<span style='color:green'>\nsuccessfully create knowledge base</span>", unsafe_allow_html=True)
            except Exception as err:
                st.write(f"{err=}, {type(err)=}")
                
            get_kb_response = bedrock_agent_client.get_knowledge_base(knowledgeBaseId = kb['knowledgeBaseId'])
            # print(kb:f{get_kb_response})
            
            # Create a DataSource in KnowledgeBase 
            create_ds_response = bedrock_agent_client.create_data_source(
                name = name,
                description = description,
                knowledgeBaseId = kb['knowledgeBaseId'],
                dataSourceConfiguration = {
                    "type": "S3",
                    "s3Configuration":s3Configuration
                },
                vectorIngestionConfiguration = {
                    "chunkingConfiguration": chunkingStrategyConfiguration
                }
            )
            ds = create_ds_response["dataSource"]
                
                # Start an ingestion job
            start_job_response = bedrock_agent_client.start_ingestion_job(knowledgeBaseId = kb['knowledgeBaseId'], dataSourceId = ds["dataSourceId"])
            job = start_job_response["ingestionJob"]
            
            # Get job 
            while(job['status']!='COMPLETE' ):
                get_job_response = bedrock_agent_client.get_ingestion_job(
                  knowledgeBaseId = kb['knowledgeBaseId'],
                    dataSourceId = ds["dataSourceId"],
                    ingestionJobId = job["ingestionJobId"]
              )
                job = get_job_response["ingestionJob"]
            
            tagging = {
                'TagSet': [
                    {
                        'Key': 'kb_id',
                        'Value': kb["knowledgeBaseId"]
                    }
                ]
            }
            s3.put_bucket_tagging(Bucket=bucket_name, Tagging=tagging)
   
        end_time=time.time()
        execution_time = end_time - start_time
        st.write(f'using:{execution_time} s to build knowledge base')
        st.write("<span style='color:green'>\nSuccessfully build knowledge base!You can upload your QAs files</span>", unsafe_allow_html=True)
        
options = ['cohere.command-text-v14', 'anthropic.claude-instant-v1','anthropic.claude-v2:1', 'ai21.j2-ultra-v1','ai21.j2-mid-v1','amazon.titan-text-lite-v1','amazon.titan-text-express-v1',]
selected = st.multiselect('Choose LLM Model:', options, default=options[0])
#打印出选择的大语言模型
if st.button('submit'):
    # 用户点击提交按钮后的操作
    st.write('You have choosed:', selected)

# 将用户选择的大模型名字返回值保存到s3桶
    num_selected_options = len(selected)
    file_path = 'output_file.txt'
    if num_selected_options >= 1:
        with open(file_path, 'w') as file:
            json.dump(selected, file)
    
    
    # 保存模型选择信息到 S3 桶
    bucket_name1 = f'rag-input-{tenantid}'
    
    folder_name1='llm_options'
    s3_key = 'llm_options.txt'
    
    s3.upload_file(file_path, bucket_name1, f'{folder_name1}/{s3_key}')

uploaded_QA_file = st.file_uploader("Upload Questions and Answers File", type=["json"])
if uploaded_QA_file :
    
    if selected == []:
        st.write('<span style="color: red;">Please choose llm model first then upload the Questions and Answers file!</span>', unsafe_allow_html=True)
    else:
        bytes_data = uploaded_QA_file.getvalue()
       
        local_file_name = uploaded_QA_file.name
        
        bucket_name2 = f'rag-input-{tenantid}'
        folder_name2 = 'input_qa'
        
        s3_location2 = f'{folder_name2}/{local_file_name}'
        
        s3.upload_fileobj(Fileobj=io.BytesIO(bytes_data),Bucket=bucket_name2, Key=s3_location2)    
        if st.button("Upload Questions and Answers File"):
            st.write('<span style="color: green;">Your Questions and Answers File Upload Success!</span>', unsafe_allow_html=True)
            #删除可能之前的文件
            s3.delete_object(Bucket=f'rag-output-{tenantid}',Key='output-rag.json')
            # s3.delete_object(Bucket=f'model-eva-output-{tenantid}',Key='answer-similarity/answer_score_per_question.png')
            # s3.delete_object(Bucket=f'model-eva-output-{tenantid}',Key='answer-similarity/result_table.png')
            # ######处理RAG生成的文件，使其文件格式符合标注文件格式######
            with st.spinner('Generating New QAs'):
                while not check_file_exists(bucket=f'rag-output-{tenantid}', key='output-rag.json'): 
                    time.sleep(5)
            st.write('<span style="color: green;">New QAs finished!</span>', unsafe_allow_html=True)
            obj=s3.get_object(Bucket=f'rag-output-{tenantid}',Key='output-rag.json')
            # 使用函数获取元数据
            # write_metadata_to_json("output-rag.json",f'rag-output-{tenantid}')
            
            #读取s3中标注文件的数据
            data = json.loads(obj['Body'].read().decode('utf-8'))
            new_data = {}

            # Iterate over each object in the data
            for i,obj in enumerate(data):
                # Extract the required values
                query = obj['query']
                responses_0 = obj['user_given_answer']
                responses = {f'response_{i+1}': response['response'] for i, response in enumerate(obj['responses'])}
                responses = OrderedDict([('response_0', responses_0)] + list(responses.items()))  
                model_ids = [response['model_id'] for response in obj['responses']]
                first_content = obj['retrievalResults'][0]['content']['text']
            
                # Create a new object with the extracted values
                new_obj = {
                    'query': query,
                    **responses,
                    'content':first_content,
                    'model_ids': model_ids if obj == data[0] else []  # Only include model_ids for the first object
                }
            
                # Add the new object to the new data list
                new_data[str(i)]=new_obj
            new_data_str = json.dumps(new_data, indent=4)
            
            # 定义文件路径
            output_file_path = os.path.join('data', 'new_output.json')
            
            # 将数据写入文件
            with open(output_file_path, 'w') as file:
                file.write(new_data_str)
        

obj = s3.get_object(Bucket=f'rag-input-{tenantid}', Key='llm_options/llm_options.txt')
data_str = obj['Body'].read().decode('utf-8')

# Parse the string into a list
new_data = json.loads(data_str)

# 用户选择的模型数
rank_list_len = len(new_data)

with open(f'./data/{rank_list_len}models_human_eva.tsv', 'w') as f:
    f.write('id\tquery')
    for i in range(1, rank_list_len + 2):
        f.write(f'\trank{i}')
