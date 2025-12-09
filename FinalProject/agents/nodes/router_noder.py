import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

sys.path.insert(0, project_root)


from FinalProject.models.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from models.llm import get_llm
from langchain_core.output_parsers import StrOutputParser
from data.dataingestion import load_all_pdfs
document = load_all_pdfs()


model = get_llm(api="USE YOUR OWN")
def route_node(state):
    question = state["messages"][-1].content

    
    prompt = ChatPromptTemplate.from_messages([
        ("system","""You are an expert router.
        Your task is to classify the user's question based on its content:
        1. 'rag': If the question is related to the topics provided in these documents : {documents}
        2. 'wikipedia': If the question is about general knowledge, history, people, or events.
        Return ONLY a single word string: 'rag' or 'wikipedia'.
        """),
        ("user","{question}")
    ])

    route_chain = prompt|model|StrOutputParser()

    route = route_chain.invoke({"question":question,"documents":document})
    if "rag" in route:
        decision = "rag"
        print("routing to rag")
    else:
        # FIX 2: Explicitly set the decision to 'wiki' to match path_map and state
        decision = "wiki" 
        print("routing to wikipedia")

    return {"source":decision}
def route_decision(state):
    return state["source"]
