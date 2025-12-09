import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

sys.path.insert(0, project_root)

from models.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

def answer_node(state):
    question = state["messages"][-1].content
    document = state["documents"]
    context = "\n\n----\n\n".join([docs.page_content for docs in document])
    model = get_llm(api="USE YOUR OWN")

    prompt = ChatPromptTemplate.from_messages([
        ("system","Your job is to provide a concise answer to the user query from the provided context: {context}"),
        ("user","{query}")
    ])

    answer_chain = prompt|model

    response = answer_chain.invoke({"query":question,"context":context})

    return {"messages":[response]}