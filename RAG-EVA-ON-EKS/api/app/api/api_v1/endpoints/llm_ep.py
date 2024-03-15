import os
import sys
import boto3
import logging
from typing import Any, Dict
from fastapi import APIRouter
from urllib.parse import urlparse
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate 
# from langchain import PromptTemplate
from .fastapi_request import (Request,
                              Text2TextModelName,
                              EmbeddingsModelName,
                              VectorDBType)
from .initialize import load_vector_db_faiss


logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

CHATHISTORY_TABLE=os.environ.get('CHATHISTORY_TABLE')

EMBEDDINGS_MODEL = os.environ.get('EMBEDDING_MODEL_ID')
TEXT2TEXT_MODEL_ID = os.environ.get('TEXT2TEXT_MODEL_ID')
BEDROCK_SERVICE = os.environ.get('BEDROCK_SERVICE')


VECTOR_DB_DIR = os.path.join("/tmp", "_vectordb")
_vector_db = None
vectordb_s3_path: str = f"s3://{os.environ.get('CONTEXTUAL_DATA_BUCKET')}/faiss_index/"

if _vector_db is None:
    _vector_db = load_vector_db_faiss(vectordb_s3_path,
                                      VECTOR_DB_DIR,
                                      EMBEDDINGS_MODEL,
                                      BEDROCK_SERVICE)
router = APIRouter()


@router.post("/rag") #接受客户端请求
def rag_handler(req: Request) -> Dict[str, Any]:
    # dump the received request for debugging purposes
    logger.info(f"req={req}")
    

    # Use the vector db to find similar documents to the query
    # the vector db call would automatically convert the query text
    # into embeddings
    docs = _vector_db.similarity_search(req.q, k=req.max_matching_docs)
    logger.info(f"here are the {req.max_matching_docs} closest matching docs to the query=\"{req.q}\"")
    for d in docs:
        logger.info(f"---------")
        logger.info(d)
        logger.info(f"---------")

    parameters = {
        "max_tokens_to_sample": req.maxTokenCount,
        "stop_sequences": req.stopSequences,
        "temperature": req.temperature,
         "top_k": req.top_k,
        "top_p": req.top_p
        }

    endpoint_name = req.text_generation_model
    # logger.info(f"ModelId: {TEXT2TEXT_MODEL_ID}, Bedrock Model: {BEDROCK_SERVICE}")
    logger.info(f"ModelId: {TEXT2TEXT_MODEL_ID}, Bedrock Model: {BEDROCK_SERVICE}")

    session_id = req.user_session_id
    boto3_bedrock = boto3.client(service_name=BEDROCK_SERVICE)
    bedrock_llm = Bedrock(model_id=TEXT2TEXT_MODEL_ID, client=boto3_bedrock)
    bedrock_llm.model_kwargs = parameters
    
    message_history = DynamoDBChatMessageHistory(table_name=CHATHISTORY_TABLE, session_id=session_id)
    memory_chain = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        input_key="question",
        ai_prefix="Assistant",
        return_messages=True
    )
    
    condense_prompt_claude = PromptTemplate.from_template("""
    Answer only with the new question.
    
    Human: How would you ask the question considering the previous conversation: {question}
    
    Assistant: Question:""")
    # claude_prompt_template = """
    # Answer only with the new question.
    
    # Human: How would you ask the question considering the previous conversation: {question}
    
    # Assistant: Question:"""
    
    
    # titan_and_ai21_prompt_template = """
    # You are an advisory AI system, and provides answers to questions by using fact based and statistical information when possible. 
    # Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # <context>
    # {context_str}
    # </context>

    # <question>
    # {query_str}
    # </question>

    # The response should be specific and stick to the context given when possible."""
#     PROMPT_TEMPLATES = {
#     "amazon.titan-text-lite-v1": titan_and_ai21_prompt_template,
#     "amazon.titan-text-express-v1": titan_and_ai21_prompt_template,
#     "ai21.j2-ultra-v1": claude_prompt_template,
#     "anthropic.claude-instant-v1":claude_prompt_template,
#     "cohere.command-text-v14": titan_and_ai21_prompt_template
# }
    qa = ConversationalRetrievalChain.from_llm(
        llm=bedrock_llm, 
        retriever=_vector_db.as_retriever(search_type='similarity', search_kwargs={"k": req.max_matching_docs}), 
        memory=memory_chain,
        condense_question_prompt=condense_prompt_claude,
        chain_type='stuff', # 'refine',
    )

    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template("""
    {context}

    Human: Answer the question inside the <q></q> XML tags.
    
    <q>{question}</q>
    
    Do not use any XML tags in the answer. If you don't know the answer or if the answer is not in the context say "Sorry, I don't know."

    Assistant:""")

    answer = ""
    answer = qa.run({'question': req.q })

    logger.info(f"answer received from llm,\nquestion: \"{req.q}\"\nanswer: \"{answer}\"")
    resp = {'question': req.q, 'answer': answer,'session_id': req.user_session_id,"top_k":req.top_k,"top_p":req.top_p,"max_length":req.max_length,"temperature":req.temperature}
    if req.verbose is True:
        resp['docs'] = docs

    return resp
    
# client = TestClient(router)
# def test_read_main():
#     response = client.get("/rag")
#     assert response.status_code == 200
#     assert response.json() == {"msg": "Hello World"}
