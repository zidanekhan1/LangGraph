from models.embeddings import get_embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from agents.nodes import retriever as retriever_holder
from graph import app

docs = [
    Document(page_content="LangChain is a framework for building LLM-powered applications."),
    Document(page_content="LangGraph adds stateful multi-step agent workflows.")
]

embeddings = get_embeddings()
vectorstore = Chroma.from_documents(docs, embeddings)
retriever_holder = vectorstore.as_retriever()

import agents.nodes as nodes_module
nodes_module.retriever = retriever_holder

query = input("Ask something: ")
result = app.invoke({"messages": [HumanMessage(content=query)]})
print("\nFINAL ANSWER:\n", result["messages"][-1].content)