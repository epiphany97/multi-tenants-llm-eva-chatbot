import streamlit as st
import json
import boto3
from botocore.exceptions import NoCredentialsError
import base64
import os
import tempfile


from streamlit.web.server.websocket_headers import _get_websocket_headers
headers = _get_websocket_headers()
tenantid = headers.get("X-Auth-Request-Tenantid")
def read_metadata_from_json(json_file):
    # 读取 JSON 文件
    with open(json_file, "r") as file:
        data = [json.loads(line) for line in file]

    return data

def get_unique_files_and_last_modified(data):
    # 提取文件和最后修改时间数据
    files = set(entry["file"] for entry in data)
    last_modified_mapping = {}
    version_id_mapping = {}
    
    for entry in data:
        last_modified_mapping.setdefault(entry["file"], []).append(entry["last_modified_human_readable"])
        version_id_mapping[(entry["file"], entry["last_modified_human_readable"])] = entry["version_id"]
    
    return files, last_modified_mapping, version_id_mapping

# 文件名映射，将原始文件名映射为显示给用户的名称
file_mapping = {
    "answer-similarity/answer_score_per_question.png": "Answer_score_per_question of Model Evaluate",
    "answer-similarity/result_table.png": "Result_table of Model Evaluate",
    "output-rag.json":"Rag Question and Answers"
    # 添加其他映射规则
}

# S3 配置
s3_bucket_name = f'model-eva-output-{tenantid}' # 替换为你的 S3 存储桶名称
s3_client = boto3.client("s3")

# Streamlit 应用
st.title("History Data Record")

# 从 JSON 文件读取数据
metadata = read_metadata_from_json("metadata/file_modify_record.json")

# 提取文件、最后修改时间和version_id数据
unique_files, last_modified_mapping, version_id_mapping = get_unique_files_and_last_modified(metadata)

# 用户选择文件的下拉菜单
selected_file = st.selectbox("Select a file", list(unique_files), format_func=lambda x: file_mapping.get(x, x))

# 获取用户选择文件对应的最后修改时间和version_id
selected_last_modified_values = last_modified_mapping.get(selected_file, [])
selected_version_id = version_id_mapping.get((selected_file, selected_last_modified_values[0]), "")

# 展示最后修改时间的下拉菜单
selected_last_modified_human_readable = st.selectbox("Select last modified time", selected_last_modified_values)

###########################test显示用户选择的文件、最后修改时间和version_id##################
# st.write(f"Selected File: {selected_file} (Original: {file_mapping.get(selected_file, selected_file)})")
# st.write(f"Selected Last Modified: {selected_last_modified_human_readable}")
# st.write(selected_file)
# st.write(f"Version ID: {selected_version_id}")

# 判断文件名是否以 ".png" 结尾
def get_binary_file_downloader_html(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file_path}" target="_blank">Click here to download the file</a>'
    return href
    # 添加下载按钮
if st.button("Download File"):
    if selected_file.lower().endswith(".png"):
        try:
            # 下载文件
            response = s3_client.get_object(Bucket=s3_bucket_name, Key=selected_file,VersionId=selected_version_id)
            content = response["Body"].read()
            
            st.download_button(
            label="Click here to download the file",
            data=content,
            file_name=selected_file,
            mime="image/png")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
    else:
        try:
            # 下载文件
            response = s3_client.get_object(Bucket=f'-rag-output-{tenantid}', Key=selected_file,VersionId=selected_version_id)
            content = response["Body"].read()
            st.download_button(
            label="Click here to download the file",
            data=content,
            file_name=selected_file,
            mime="application/json")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    