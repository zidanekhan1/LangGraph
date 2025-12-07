from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from models.llm import get_llm

llm = get_llm()

retriever = None

def router_node(state):
    text = state["messages"][-1].content
    if "langchain" in text.lower():
        print("Routing → RAG")
        return {"route": "rag"}
    print("Routing → FINAL")
    return {"route": "final"}


def rag_node(state):
    print("Running RAG node...")
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    simple_docs = [d.page_content for d in docs]
    return {"documents": simple_docs, "route": "final"}


def answer_node(state):
    print("Producing final answer...")
    question = state["messages"][-1].content
    context = "\n".join(state["documents"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following context to answer:\n{context}"),
        ("user", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})

    return {"messages": [response]}