import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, project_root)

from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import Chroma
from FinalProject.data.dataingestion import load_all_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from models.embedding import get_embeddings

pdf_data = load_all_pdfs()
embedder = get_embeddings()
def get_rag_retriever():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=270)
    splits = text_splitter.split_documents(pdf_data)
    vectorstore = Chroma.from_documents(documents=splits,embedding=embedder)
    rag_retriever = vectorstore.as_retriever()
    return rag_retriever

def get_wiki_retriever():
    wikiretriever = WikipediaRetriever(top_k_results=2)
    return wikiretriever
    

