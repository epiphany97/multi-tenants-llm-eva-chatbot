import boto3
import pprint
from botocore.client import Config

def get_bedrock_agent_client(bedrock_config=None):
    if bedrock_config is None:
        bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
    bedrock_agent_client = boto3.client("bedrock-agent-runtime",
                              config=bedrock_config)
    return bedrock_agent_client

def retrieve(bedrock_agent_client, query, kbId, numberOfResults=5):
    return bedrock_agent_client.retrieve(
        retrievalQuery= {
            'text': query
        },
        knowledgeBaseId=kbId,
        retrievalConfiguration= {
            'vectorSearchConfiguration': {
                'numberOfResults': numberOfResults
            }
        }
    )

if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=2)

    query = "What can I do to help with the sustainability goals?"
    kb_id = "MYSZREMZOO"

    bedroc_agent_client = get_bedrock_agent_client()
    response = retrieve(bedroc_agent_client, query, kb_id, 5)

    retrievalResults = response['retrievalResults']
    pp.pprint(retrievalResults)
