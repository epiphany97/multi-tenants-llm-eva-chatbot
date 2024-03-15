from langchain.prompts import PromptTemplate
import boto3
import pprint
from botocore.client import Config
from langchain.llms.bedrock import Bedrock

from retreival import get_bedrock_agent_client, retrieve

pp = pprint.PrettyPrinter(indent=2)
# ["cohere.command-text-v14", "ai21.j2-ultra-v1", "amazon.titan-text-express-v1", "amazon.titan-text-lite-v1", "ai21.j2-mid-v1", "anthropic.claude-v2:1"]
parameters = {
    "amazon.titan-text-lite-v1":{},
    "anthropic.claude-v2:1":{},
    "ai21.j2-mid-v1":{},
    "ai21.j2-ultra-v1":{},
    "amazon.titan-text-express-v1":{},
    "cohere.command-text-v14":{}
}

titan_and_ai21_prompt_template = """
    You are an advisory AI system, and provides answers to questions by using fact based and statistical information when possible. 
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context_str}
    </context>

    <question>
    {query_str}
    </question>

    The response should be specific and stick to the context given when possible.
    """

claude_prompt_template = """\n\nHuman: 
    You are an advisory AI system, and provides answers to questions by using fact based and statistical information when possible. 
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context_str}
    </context>

    <question>
    {query_str}
    </question>

    The response should be specific and stick to the context given when possible.

    Assistant:"""

PROMPT_TEMPLATES = {
    "amazon.titan-text-lite-v1": titan_and_ai21_prompt_template,
    "amazon.titan-text-express-v1": titan_and_ai21_prompt_template,
    "ai21.j2-mid-v1": titan_and_ai21_prompt_template,
    "ai21.j2-ultra-v1": titan_and_ai21_prompt_template,
    "anthropic.claude-v2:1":claude_prompt_template,
    "anthropic.claude-3-sonnet-20240229-v1:0":claude_prompt_template,
    "anthropic.claude-instant-v1":claude_prompt_template,
    "cohere.command-text-v14": titan_and_ai21_prompt_template
}

# fetch context from the response
def get_contexts(retrievalResults):
    contexts = []
    for retrievedResult in retrievalResults: 
        contexts.append(retrievedResult['content']['text'])
    return contexts

def get_response(model_id, query, contexts, bedrock_client):

    llm = Bedrock(model_id = model_id,
              model_kwargs=parameters[model_id],
              client = bedrock_client)

    prompt_template = PromptTemplate(template=PROMPT_TEMPLATES[model_id],
                               input_variables=["context_str","query_str"])

    prompt = prompt_template.format(context_str=contexts, 
                                 query_str=query)
    
    # print("the prompt is: ")
    # print(prompt)

    # generate response
    response = llm(prompt)

    # print("the response is: ")
    # pp.pprint(response)

    return response


if __name__  == "__main__":
    bedrock_client = boto3.client('bedrock-runtime')
    model_id = "anthropic.claude-v2"
    # model_id = "amazon.titan-text-lite-v1"

    # get retrieval results
    query = "What can I do to help with the sustainability goals?"
    kb_id = "MYSZREMZOO"

    # get retrieval results
    bedroc_agent_client = get_bedrock_agent_client()
    response = retrieve(bedroc_agent_client, query, kb_id, numberOfResults=5)

    retrievalResults = response['retrievalResults']

    # construct prompt
    contexts = get_contexts(retrievalResults)

    response = get_response(model_id, query, contexts, bedrock_client)
    print(response)