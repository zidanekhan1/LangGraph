import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")

def load_all_pdfs() -> List[Document]:
    """
    Loads all PDF documents from the designated 'pdfs' folder
    using LangChain's optimized directory loader.
    """
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER, exist_ok=True)
        print(f"Created PDF ingestion directory: {PDF_FOLDER}")
        return []

    try:
        
        loader = PyPDFDirectoryLoader(PDF_FOLDER)
        
        all_docs = loader.load()
        
        print(f"Successfully loaded {len(all_docs)} document pages from the 'pdfs' folder.")
        
        for doc in all_docs:
            if 'source' in doc.metadata:
                doc.metadata['source_short'] = os.path.basename(doc.metadata['source'])

        return all_docs
        
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []