import json
import urllib.parse
import boto3
import json
import os
import asyncio
import concurrent.futures
# from utils import read_qa, read_llm_options
from generate_response import get_response, get_contexts
from retreival import get_bedrock_agent_client, retrieve

print('Loading function with parallel')
s3 = boto3.client("s3")
# kb_id = os.environ['KB_ID']


# get get retrieval results async
def get_retrieval_results_wrapper(bedrock_client, query, user_given_answer, kb_id, numberOfResults=5):
    # print("in query")
    result = retrieve(bedrock_client, query, kb_id, numberOfResults)
    return (query, user_given_answer, result)
    
def get_response_wrapper(i, model_id, query, contexts, bedrock_client):
    # print("in response")
    response = get_response(model_id, query, contexts, bedrock_client)
    return (i, model_id, response)


async def generate_results(querys, user_given_answers, model_ids, kb_id, numberOfResults=5):
    bedrock_client = boto3.client('bedrock-runtime')
    bedrock_agent_client = get_bedrock_agent_client()

    retreival_tasks = []
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(os.environ['MAX_WORKERS']),
    )

    for i, (query, user_given_answer) in enumerate(zip(querys, user_given_answers)):
        # get retrieval results
        retreival_tasks.append(loop.run_in_executor(executor, get_retrieval_results_wrapper, bedrock_agent_client, query, user_given_answer, kb_id, numberOfResults))
        

    completed, pending = await asyncio.wait(retreival_tasks)
    results = [t.result() for t in completed]

    generate_tasks = []
    data = []
    for i, (query, user_given_answer, responses) in enumerate(results):
        # construct prompt
        retrievalResults = responses['retrievalResults']
        contexts = get_contexts(retrievalResults)

        # construct output
        data.append({"query": query, 
            "user_given_answer": user_given_answer, 
            "retrievalResults": retrievalResults, 
            "responses": []})

        for model_id in model_ids:
            # get_response(model_id, query, contexts, bedrock_client)
            generate_tasks.append(loop.run_in_executor(executor, get_response_wrapper, i, model_id, query, contexts, bedrock_client))

    completed, pending = await asyncio.wait(generate_tasks)
    results = [t.result() for t in completed]
    
    for i, model_id, response in results:
        data[i]["responses"].append({"model_id":model_id, "response":response})
    return data




def lambda_handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    s3 = boto3.client('s3')

    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')


    # 获取桶的标签
    response = s3.get_bucket_tagging(Bucket=bucket)

    # 获取特定标签键为 'kb_id' 的标签值
    tag_key = 'kb_id'  
    kb_id = next((tag['Value'] for tag in response['TagSet'] if tag['Key'] == tag_key), None)

    try:
        # get user uploaded qa
        response = s3.get_object(Bucket=bucket, Key=key)
        json_data = response['Body'].read().decode('utf-8')
        qa_data = json.loads(json_data)

        # get model selection
        model_ids_key = os.environ['MODEL_IDS_KEY']
        response = s3.get_object(Bucket=bucket, Key=model_ids_key)
        json_data = response['Body'].read().decode('utf-8')
        llm_options = json.loads(json_data)

        loop = asyncio.get_event_loop()

        results = loop.run_until_complete(generate_results(qa_data["question"], qa_data["answer"], llm_options, kb_id)) 

        if os.environ['OUTPUT_BUCKET'] is not None:
            output_bucket = os.environ['OUTPUT_BUCKET']
            output_key = os.environ['OUTPUT_KEY']
            output_filename = os.environ['OUTPUT_FILENAME']
            output_fullpath = output_key + output_filename
        if output_bucket is not None:
            upload_stream = bytes(json.dumps(results), encoding='utf-8')
            s3.put_object(Bucket=output_bucket, Key=output_fullpath, Body=upload_stream)
            print('Result uploaded to S3:{}/{} successfully'.format(output_bucket, output_fullpath))
        else:
            raise Exception('output_bucket is not defined')

    except Exception as e:
        print(e)
        # print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e