from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import AgentState
from src.nodes import intent_classifier, retrieve_context, department_agent, human_approval, supervisor_agent
import sqlite3

def route_query(state: AgentState):
    if state["intent"] == "Memory":
        return "supervisor"
    return "retrieve_context"

def check_approval(state: AgentState):
    if state.get("approval_required"):
        return "human_approval"
    return "supervisor"

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classifier", intent_classifier)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("department", department_agent)
workflow.add_node("human_approval", human_approval)
workflow.add_node("supervisor", supervisor_agent)

# Add edges
workflow.add_edge(START, "classifier")
workflow.add_conditional_edges("classifier", route_query, {"retrieve_context": "retrieve_context", "supervisor": "supervisor"})
workflow.add_edge("retrieve_context", "department")
workflow.add_conditional_edges("department", check_approval, {"human_approval": "human_approval", "supervisor": "supervisor"})
workflow.add_edge("human_approval", "supervisor")
workflow.add_edge("supervisor", END)

# Set up SQLite memory
conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Compile the graph with interrupt before human approval
app = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])
