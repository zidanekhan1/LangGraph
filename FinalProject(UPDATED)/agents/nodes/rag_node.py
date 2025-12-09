
def rag_node(state):
    question = state["messages"][-1].content
    
    rag_retriever = state.get("rag_retriever") 
    
    if rag_retriever is None:
        print("RAG source is not available. Skipping retrieval.")
        return {"documents": []}

    documents = rag_retriever.invoke(question)

    return {"documents": documents}