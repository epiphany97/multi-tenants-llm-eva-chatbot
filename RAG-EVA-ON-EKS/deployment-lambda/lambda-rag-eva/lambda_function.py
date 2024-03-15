from __future__ import annotations
import os
from ragas.llms import LangchainLLM
from langchain.chat_models import BedrockChat
from langchain.embeddings import BedrockEmbeddings
from ragas import evaluate
import json
import urllib.parse
import boto3
import typing as t
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
from plottable import Table

import numpy as np
from datasets import Dataset
from sentence_transformers import CrossEncoder

from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager


print("boto3 version is: ", boto3.__version__)

s3 = boto3.client('s3')

bedrock_runtime = boto3.client('bedrock-runtime')

if os.environ['OUTPUT_BUCKET'] is not None:
    output_bucket = os.environ['OUTPUT_BUCKET']
    output_key = os.environ['OUTPUT_KEY']
    output_filename = os.environ['OUTPUT_FILENAME']
    output_fullpath = output_key + output_filename

config = {
    # "credentials_profile_name": "techucapstone",  # E.g "default"
    "region_name": "us-west-2",  # E.g. "us-east-1"
    "model_id": "anthropic.claude-v2",  # E.g "anthropic.claude-v2"
    # "model_id": "anthropic.claude-v2:1",
    "model_kwargs": {"temperature": 0.4},
}

bedrock_model = BedrockChat(
    # credentials_profile_name=config["credentials_profile_name"],
    client=bedrock_runtime,
    region_name=config["region_name"],
    endpoint_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
    model_id=config["model_id"],
    model_kwargs=config["model_kwargs"],
)
# wrapper around bedrock_model
ragas_bedrock_model = LangchainLLM(bedrock_model)

# init and change the embeddings
# only for answer_relevancy
bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    region_name=config["region_name"],
)



@dataclass
class AnswerSimilarityInline(MetricWithLLM):
    """
    Scores the semantic similarity of ground truth with generated answer.
    cross encoder score is used to quantify semantic similarity.
    SAS paper: https://arxiv.org/pdf/2108.06130.pdf

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    embeddings:
        The cross-encoder model to be used.
        Defaults cross-encoder/stsb-TinyBERT-L-4
        Other good options https://huggingface.co/spaces/mteb/leaderboard
    threshold:
        The threshold if given used to map output to binary
        Default 0.5
    """

    name: str = "answer_similarity"
    evaluation_mode: EvaluationMode = EvaluationMode.ga
    batch_size: int = 15
    embeddings: str | None = None
    threshold: float | None = 0.5

    def __post_init__(self: t.Self):
        if self.embeddings is None:
            self.cross_encoder = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")

    def init_model(self: t.Self):
        pass

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        ground_truths, answers = dataset["ground_truths"], dataset["answer"]
        ground_truths = [item[0] for item in ground_truths]
        inputs = [list(item) for item in list(zip(ground_truths, answers))]
        scores = self.cross_encoder.predict(
            inputs, batch_size=self.batch_size, convert_to_numpy=True
        )

        assert isinstance(scores, np.ndarray), "Expects ndarray"
        if self.threshold:
            scores = scores >= self.threshold  # type: ignore

        return scores.tolist()


def similarity_score(ag: Dataset) -> list[float]:
    """
    Scores the semantic similarity of ground truth with generated answer.
    cross encoder score is used to quantify semantic similarity.

    Attributes
    ----------
    ag : Answer and Ground_truth datasets.
    """
    answer_similarity = AnswerSimilarityInline(threshold=None)
    answer_similarity.llm = ragas_bedrock_model
    # answer_similarity.embeddings = bedrock_embeddings

    results = answer_similarity.score(ag)
    return results

def plot_dynamic_lines(data_sets):
    # 自动生成颜色和标记
    plt.ylim(0, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_sets)))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'H', '+', 'x', 'D']

    # 循环处理每个数据集
    for i, data in enumerate(data_sets):
        x_values, y_values, label = data
        # 自动生成颜色和标记
        color = colors[i]
        marker = markers[i % len(markers)]

        # 绘制折线
        plt.plot(x_values, y_values, label=label, color=color, marker=marker)

    # 添加标签和标题
    plt.xlabel('Questions')
    plt.ylabel('Answer Score')
    plt.title('Answer Score per Question')

    # 添加图例
    plt.legend()

    # 保存图
    plt.savefig('/tmp/answer_score_per_question.png')
    plt.close

    # 显示图表
    # plt.show()

def draw_table(sample_result):
    sorted_res = sorted(sample_result, key=lambda x: x['mean'], reverse=True)
    df_data = {'Model': [],
            'Answer Score ↑': [],
            'Stability ↓': []}
    for i in sorted_res:
        df_data['Model'].append(i['model_id'])
        df_data['Answer Score ↑'].append(round(i['mean'], 4))
        df_data['Stability ↓'].append(round(i['variance'], 4))
    df_data = pd.DataFrame(df_data)
    df_data.insert(0, 'Rank', [i+1 for i in range(len(sorted_res))])
    df_data.at[0, 'Rank'] = "♛ 1"
    df_data.set_index('Rank', inplace=True)
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')
    tab = Table(df_data, row_dividers=False, odd_row_color="#f0f0f0", even_row_color="#e0f6ff")

    plt.savefig('/tmp/result_table.png')
    plt.close
    # plt.show()

def handler(event, context):
    # load file in S3
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        json_data = response['Body'].read().decode('utf-8')
        print("CONTENT TYPE: " + response['ContentType'])
        data = json.loads(json_data)
        # Data preprocessing
        ga_list = []
        model_list = []
        for i in data[0]["responses"]:
            model_list.append(i["model_id"])
        # Split data by model_id
        for i in model_list:
            mkv = {"model_id": i,
                "ground_truths": [],
                "answer": []
                }
            ga_list.append(mkv)
            
        for i in ga_list:
            for j in data:
                i["ground_truths"].append(j["user_given_answer"])
                for k in j["responses"]:
                    if k["model_id"] == i["model_id"]:
                        i["answer"].append(k["response"])
        # Evaluating by different model_id
        results = []
        for i in ga_list:
            model_id = i.pop("model_id")
            ag = Dataset.from_dict(i)
            model_sim_score = similarity_score(ag)
            # DEBUG
            print("model_sim_score['answer_similarity'] is: ", model_sim_score['answer_similarity'])
            res = {
                "model_id": model_id, 
                "similarity_all": model_sim_score['answer_similarity'],
                "mean": np.mean(model_sim_score['answer_similarity']),
                "variance": np.var(model_sim_score['answer_similarity'])
                }
            results.append(res)
        # Drawing
        dataset_list = []
        for i in results:
            dataset_list.append([[j for j in range(len(i['similarity_all']))], i['similarity_all'], i['model_id']])
        # 调用函数绘制图表
        plot_dynamic_lines(dataset_list)
        draw_table(results)
        # Output
        if output_bucket is not None:
            upload_stream = bytes(json.dumps(results), encoding='utf-8')
            s3.put_object(Bucket=output_bucket, Key=output_fullpath, Body=upload_stream)
            print('Result uploaded to S3:{}/{} successfully'.format(output_bucket, output_fullpath))
            s3.upload_file('/tmp/answer_score_per_question.png', output_bucket, output_key + 'answer_score_per_question.png')
            s3.upload_file('/tmp/result_table.png', output_bucket, output_key + 'result_table.png')
            print('Figure and Table are both uploaded to S3:{}/{} successfully'.format(output_bucket, output_key))
        else:
            raise Exception('output_bucket is not defined')
        return results
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
        

