"""Create a ChatVectorDBChain for question/answering."""
#from langchain.callbacks.base import AsyncCallbackManager
#from langchain.callbacks.tracers import LangChainTracer
#from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
#from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
#from langchain.llms import OpenAI
from langchain.llms import Cohere
from langchain.vectorstores.base import VectorStore
from langchain import PromptTemplate

import boto3
from langchain.llms.bedrock import Bedrock

'''def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ChatVectorDBChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    qa = ChatVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
    )
    return qa'''

def get_cohere_chain(vectorstore: VectorStore) -> ConversationalRetrievalChain:
    prompt_template = """Text: {context}
    
    Question: {question}
    
    You are an intelligent Aerospace assistant! Create a concise and informative answer for a given question 
    based solely on the given documents. You must only use information from the given documents. Use an
    unbiased and journalistic tone. If the answer is not available in the documents, please say 'I don't know'
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    #chain = load_qa_chain(Cohere(model="command-xlarge-nightly", temperature=0),
                          #chain_type="stuff", prompt=PROMPT)
    #question_gen_llm = Cohere(temperature=0, verbose=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm = Cohere(model="command-xlarge-nightly", temperature=0),
        retriever=vectorstore.as_retriever(),
        condense_question_prompt = CONDENSE_QUESTION_PROMPT,
        condense_question_llm = Cohere(temperature=0, verbose=True),
        combine_docs_chain_kwargs = {"prompt": PROMPT},
        return_source_documents = True,
    )
    return qa


def get_bedrock_chain(vectorstore: VectorStore) -> ConversationalRetrievalChain:
    prompt_template = """Text: {context}

    Question: {question}

    You are an intelligent Aerospace assistant! Create a concise and informative answer for a given question 
    based solely on the given documents. You must only use information from the given documents. Use an
    unbiased and journalistic tone. If the answer is not available in the documents, please say 'I don't know'
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    bedrock_client = boto3.client(service_name='bedrock', region_name='us-east-1',
                                  endpoint_url='https://bedrock.us-east-1.amazonaws.com',
                                  aws_access_key_id = '',
                                  aws_secret_access_key='')

    bedrock_llm = Bedrock(
        model_id="amazon.titan-tg1-large",
        client = bedrock_client
    )

    bedrock_qa = ConversationalRetrievalChain.from_llm(
        llm=bedrock_llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        condense_question_llm=bedrock_llm,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return bedrock_qa