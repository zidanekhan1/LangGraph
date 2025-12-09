import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")

def load_all_pdfs():
    if not os.path.exists(PDF_FOLDER):
        return []

    pdf_paths = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs