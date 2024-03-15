import streamlit as st
import pandas as pd
import json
import boto3
from collections import OrderedDict
import os

from streamlit.web.server.websocket_headers import _get_websocket_headers
headers = _get_websocket_headers()
tenantid = headers.get("X-Auth-Request-Tenantid")


s3=boto3.client('s3')

obj = s3.get_object(Bucket=f'rag-input-{tenantid}', Key='llm_options/llm_options.txt')
data_str = obj['Body'].read().decode('utf-8')

# Parse the string into a list
new_data = json.loads(data_str)

# 用户选择的模型数
rank_list_len = len(new_data)

CONFIGS = {
    # "input_bucket": f'rag-output-{tenantid}',  # S3 bucket where the input file is stored
    "input_path":"./data/new_output.json",
    # "input_key": "new_output_example.json",  # Key of the input file in the S3 bucket
  "output_path": f"./data/{rank_list_len}models_human_eva.tsv",  # 标注数据集的存放文件
    "rank_list_len": rank_list_len+1,
}

######################## 页面配置初始化 ###########################
RANK_COLOR = ["red", "green", "blue", "orange", "violet"]
label_tab, dataset_tab = st.tabs(["Label", "Dataset"])


######################### 页面定义区（标注页面） ########################
with label_tab:
    with st.expander("🔍 Setting Prompts", expanded=True):
        # obj = s3.get_object(Bucket=CONFIGS["input_bucket"], Key=CONFIGS["input_key"])
        # data = json.loads(obj['Body'].read().decode('utf-8'))
        with open(CONFIGS["input_path"], "r", encoding="utf-8") as f:
            data = json.load(f)

        query_ids = list(data.keys())
        query_index_number = st.number_input(
            "Current query number (click on the one on the right) ➕➖Jump back and forth",
            min_value=0,
            max_value=len(query_ids) - 1,
            value=0,
        )

        current_query_id = query_ids[query_index_number]
        current_query = data[current_query_id]["query"]
        # current_history = data[current_query_id]["history"]

        st.markdown(f"**Query:** {current_query}")
        # st.markdown("**History:**")
        # for history_item in current_history:
        #     st.write(f"- {history_item[0]}")
        #     st.write(f"  {history_item[1]}")

    # 排序功能
    with st.expander("💡 Generate Results", expanded=True):
        rank_results = []
        for i in range(CONFIGS["rank_list_len"]):
            # st.write(f'**Response {i + 1}:**，请标注其排名')
            response_text = data[current_query_id][f"response_{i}"]
            rank = st.selectbox(
                f"Please indicate the ranking of answer {i + 1} ",
                ["default"] + list(range(1, CONFIGS["rank_list_len"] + 1)),
                help="Choose a ranking for the current response, the better the quality of the answer, the higher the ranking. (default represents that the current sentence has not been ranked yet)",
            )

            conflict_index = next(
                (idx + 1 for idx, r in enumerate(rank_results) if r == rank), None
            )
            if conflict_index is not None and rank != -1:
                st.info(
                    f"The current ranking [{rank}] is already occupied by the sentence [{conflict_index}].Please set the sentence occupying the ranking to default first before assigning the ranking to the current sentence."
                )
            else:
                rank_results.append(rank)

            st.markdown(
                f"<span style='color:{RANK_COLOR[i]}'>{response_text}</span>",
                unsafe_allow_html=True,
            )
            # st.write(f'当前排名：**{rank}**')
            # st.write('---')
        output_data=[]
        # 排序存储功能
        if -1 not in rank_results:
            save_button = st.button("Store Current Sort")
            dataset_file = CONFIGS["output_path"]
            if save_button:
                dataset_file = CONFIGS["output_path"]
                # output_data_str = json.dumps(processed_data, indent=4)
                # s3.put_object(Body=output_data_str, Bucket=CONFIGS["output_bucket"], Key=CONFIGS["output_key"])
                df = pd.read_csv(dataset_file, delimiter="\t", dtype=str)
                # print(df)
                existing_ids = df["id"].tolist()

                rank_texts = [
                    data[current_query_id][f"response_{rank - 1}"]
                    for rank in rank_results
                ]
                # line = [current_query_id, current_query, current_history] + rank_texts
                line = [current_query_id, current_query] + rank_texts
                output_data.append(line)
                new_row = pd.DataFrame([line], columns=df.columns)

                if current_query_id in existing_ids:
                    df = df[df["id"] != current_query_id]  # 删除已存在的行

                df = pd.concat([df, new_row], ignore_index=True)  # 追加新行

                df.to_csv(dataset_file, index=False, sep="\t")  # 保存到文件
                st.success(f"Item {current_query_id} Data Saved!")
                df = pd.read_csv(dataset_file, delimiter="\t", dtype=str)


                query_index_number += 1
                if query_index_number >= len(query_ids):
                    st.write("All Queries Have Been Annotated")
                upload_to_s3_button = st.button('Upload to S3')
                if upload_to_s3_button:
                    # 调用上传函数
                    upload_success = s3.upload_file(dataset_file, f"rag-output-{tenantid}")
            
                    if upload_success:
                        st.success("文件成功上传到S3")
                    else:
                        st.error("文件上传失败")
            
        else:
            st.error("Please Complete Sorting Before Storing！", icon="🚨")
            
       
    # with st.expander('🥇 Rank Results', expanded=True):
    #     columns = st.columns([1] * CONFIGS['rank_list_len'])
    #     for i, c in enumerate(columns):
    #         with c:
    #             st.write(f'Rank {i+1}：')
    #             if i + 1 in rank_results:
    #                 color = RANK_COLOR[rank_results.index(i+1)] if rank_results.index(i+1) < len(RANK_COLOR) else 'white'
    #                 st.markdown(f":{color}[{st.session_state['current_results'][rank_results.index(i+1)]}]")

######################### 页面定义区（数据集页面） #######################
with dataset_tab:
    try:
        dataset_file = CONFIGS["output_path"]
        df = pd.read_csv(dataset_file, delimiter="\t", dtype=str)
    
        st.dataframe(df)
    except Exception as e:
        st.error(f"an error occured: {e}")