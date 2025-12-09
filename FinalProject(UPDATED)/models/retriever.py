
import os
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 
from .embedding import get_embeddings
from typing import List

embedder = get_embeddings()


def get_rag_retriever_from_paths(pdf_paths: List[str]):
    """Loads PDFs from a list of paths, splits them, and creates a Chroma retriever."""
    
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=270)
    splits = text_splitter.split_documents(all_docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedder)
    rag_retriever = vectorstore.as_retriever()
    
    return rag_retriever

def get_wiki_retriever():
    wikiretriever = WikipediaRetriever(top_k_results=2)
    return wikiretriever