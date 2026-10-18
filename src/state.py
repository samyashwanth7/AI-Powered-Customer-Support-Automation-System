import operator
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str
    department: str
    retrieved_context: str
    approval_required: bool
    final_response: str
    sentiment: str
    priority: str
    ticket_id: str
    total_tokens: Annotated[int, operator.add]
    search_query: str
    retry_count: int
    is_grounded: bool
