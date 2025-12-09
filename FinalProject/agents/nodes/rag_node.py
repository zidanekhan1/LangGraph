import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

sys.path.insert(0, project_root)

from models.retriever import get_rag_retriever


retriever = get_rag_retriever()

def rag_node(state):
    question = state["messages"][-1].content
    document = retriever.invoke(question)

    return {"documents":document,"source":"final"}    
