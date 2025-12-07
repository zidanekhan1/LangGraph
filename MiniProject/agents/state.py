from typing import TypedDict, Annotated, List, Literal
from operator import add
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    route: Literal["rag", "final"]
    documents: List[str] 