import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.join(current_dir, os.pardir)

sys.path.insert(0, project_root)

from agents.nodes.rag_node import rag_node
from agents.nodes.wiki_node import wiki_node
from agents.nodes.answer_node import answer_node
from agents.nodes.router_noder import route_node,route_decision
from agents.state import AgentGraph
from langgraph.graph import StateGraph

graph = StateGraph(state_schema=AgentGraph)

graph.add_node("router",route_node)
graph.add_node("document",rag_node)
graph.add_node("wiki",wiki_node)
graph.add_node("answer",answer_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    source="router",
    path=route_decision,
    path_map={
        "rag":"document",
        "wiki":"wiki"
    }
)

graph.add_edge("document","answer")
graph.add_edge("wiki","answer")

graph.set_finish_point("answer")

app = graph.compile()
