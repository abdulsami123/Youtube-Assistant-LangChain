import os
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI







model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)








def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, api_key ):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key= api_key # if you prefer to pass api key in directly instaed of using env vars
)

    


    

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
        """You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript.
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed."""
    ),
    HumanMessagePromptTemplate.from_template(
        """Answer the following question: {question}
        By searching the following video transcript: {docs}"""
    )
    ])


    chain =  prompt | llm

    response = chain.invoke({
        "question": query,
        "docs": docs_page_content
    })
    #response.content = response.content.replace('\n', '')
    return response.content, docs

