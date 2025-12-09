import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

sys.path.insert(0, project_root)


from models.retriever import get_wiki_retriever


def wiki_node(state):
    retriever = get_wiki_retriever()
    question = state["messages"][-1].content
    document = retriever.invoke(question)

    return {"documents":document,"source":"final"}