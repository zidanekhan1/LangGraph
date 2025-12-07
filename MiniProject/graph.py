from langgraph.graph import StateGraph
from agents.state import AgentState
from agents.nodes import router_node, rag_node, answer_node

graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("rag", rag_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "rag": "rag",
        "final": "answer"
    }
)

graph.add_edge("rag", "answer")
graph.set_finish_point("answer")

app = graph.compile()