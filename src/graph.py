from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import AgentState
from src.nodes import (
    intent_classifier,
    retrieve_context,
    department_agent,
    rewrite_query,
    human_approval,
    supervisor_agent
)
import sqlite3

def route_query(state: AgentState):
    if state["intent"] == "Memory":
        return "supervisor"
    return "retrieve_context"

def evaluate_grounding(state: AgentState):
    """Cyclic decision: loop back to rewrite_query if ungrounded and retries remain, else check approval."""
    if not state.get("is_grounded", True) and state.get("retry_count", 0) < 1:
        return "rewrite_query"
    if state.get("approval_required"):
        return "human_approval"
    return "supervisor"

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classifier", intent_classifier)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("department", department_agent)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("human_approval", human_approval)
workflow.add_node("supervisor", supervisor_agent)

# Add edges
workflow.add_edge(START, "classifier")
workflow.add_conditional_edges("classifier", route_query, {
    "retrieve_context": "retrieve_context",
    "supervisor": "supervisor"
})
workflow.add_edge("retrieve_context", "department")

# Cyclic edge from department:
workflow.add_conditional_edges("department", evaluate_grounding, {
    "rewrite_query": "rewrite_query",
    "human_approval": "human_approval",
    "supervisor": "supervisor"
})
workflow.add_edge("rewrite_query", "retrieve_context")

workflow.add_edge("human_approval", "supervisor")
workflow.add_edge("supervisor", END)

# Set up SQLite memory
conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Compile the graph with interrupt before human approval
app = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])
