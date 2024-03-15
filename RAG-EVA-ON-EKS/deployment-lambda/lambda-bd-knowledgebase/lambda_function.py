import json
import urllib.parse
import boto3
import json
import os
# from utils import read_qa, read_llm_options
from generate_response import get_response, get_contexts
from retreival import get_bedrock_agent_client, retrieve

print('Loading function')

kb_id = os.environ['KB_ID']

def generate_results(querys, user_given_answers, model_ids, kb_id):
    data = []
    for i, (query, user_given_answer) in enumerate(zip(querys, user_given_answers)):
    # get retrieval result
        bedroc_agent_client = get_bedrock_agent_client()

        response = retrieve(bedroc_agent_client, query, kb_id, numberOfResults=5)

        retrievalResults = response['retrievalResults']

        # construct prompt
        contexts = get_contexts(retrievalResults)

        responses = []
        bedrock_client = boto3.client('bedrock-runtime')

        for model_id in model_ids:
            response = get_response(model_id, query, contexts, bedrock_client)
            # pp.pprint(response)
            responses.append({"model_id":model_id, "response":response})

        data.append({"query": query, 
                "user_given_answer": user_given_answer, 
                "retrievalResults": retrievalResults, 
                "responses": responses})
        print("qa", i, "is ready")
    return data



s3 = boto3.client('s3')

def lambda_handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
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

        results = generate_results(qa_data["question"], qa_data["answer"], llm_options, kb_id)

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