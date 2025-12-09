from langgraph.graph import StateGraph
from langgraph.graph.message import TypedDict,Annotated,Literal
from typing import List
from operator import add
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class AgentGraph(TypedDict):
    messages:Annotated[List[BaseMessage],add]
    documents:List[Document]
    source:Literal["rag","wiki","final"]