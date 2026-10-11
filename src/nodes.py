import os
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from src.state import AgentState
from src.rag import retriever

# Initialize LLM
llm = ChatGroq(model="qwen/qwen3.8-27b", api_key=os.getenv("GROQ_API_KEY"))

class IntentOutput(BaseModel):
    intent: str = Field(description="The category of the customer query: Sales, Technical, Billing, Account, or Memory")
    department: str = Field(description="The department to route to: Sales, Technical, Billing, Account, or None")
    sentiment: str = Field(description="The customer's emotional tone: Frustrated, Neutral, Positive, or Urgent")
    priority: str = Field(description="Priority level for ticket triage: High, Medium, or Low")

def intent_classifier(state: AgentState):
    """Categorizes query, extracts customer sentiment, and assigns triage priority."""
    last_message = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    
    system_prompt = """You are an AI Support Triage Agent for ABC Technologies.
    Analyze the user's query and output:
    1. intent: Sales, Technical, Billing, Account, or Memory.
       - Sales: Product info, pricing plans.
       - Technical: Application errors, installation, crashes.
       - Billing: Invoices, refunds, cancellations.
       - Account: Passwords, profile updates, activation.
       - Memory: Asking about previous interactions (e.g., 'What was my previous issue?').
    2. department: Sales, Technical, Billing, Account, or None (route Memory queries to None).
    3. sentiment: Frustrated, Neutral, Positive, or Urgent.
    4. priority:
       - High: System outages, crashes, refund demands, urgent escalations, strong frustration.
       - Medium: Account recovery, invoice requests, setup questions.
       - Low: General product information, pricing exploration, casual questions.
    """
    
    structured_llm = llm.with_structured_output(IntentOutput)
    response = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    
    ticket_id = state.get("ticket_id") or f"#TKT-{random.randint(1000, 9999)}"
    
    return {
        "intent": response.intent,
        "department": response.department,
        "sentiment": response.sentiment,
        "priority": response.priority,
        "ticket_id": ticket_id
    }

def retrieve_context(state: AgentState):
    """Retrieves context for the given query."""
    query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def department_agent(state: AgentState):
    """Generic department agent that uses retrieved context."""
    department = state.get("department", "General")
    context = state.get("retrieved_context", "")
    sentiment = state.get("sentiment", "Neutral")
    query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    
    system_prompt = f"""You are a customer support agent for the {department} department at ABC Technologies.
    Detected customer tone: {sentiment}.
    Answer the user's query using ONLY the provided context. If the context does not contain the answer, politely say you don't know but will escalate.
    Context: {context}
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    
    # Check for human-in-the-loop triggers
    high_risk_keywords = ["refund", "cancel", "closure", "compensation", "escalate"]
    approval_required = any(keyword in query.lower() for keyword in high_risk_keywords)
    
    return {"messages": [response], "approval_required": approval_required}

def human_approval(state: AgentState):
    """Node executed after a human has intervened to approve."""
    approval_msg = AIMessage(content="System: The requested action has been APPROVED by the human supervisor.")
    return {"messages": [approval_msg]}

def supervisor_agent(state: AgentState):
    """Validates and improves responses before sending, tailored to sentiment and priority."""
    messages = state["messages"]
    intent = state.get("intent", "")
    approval_required = state.get("approval_required", False)
    sentiment = state.get("sentiment", "Neutral")
    priority = state.get("priority", "Medium")
    
    # For memory queries, read the conversation history (use recent window to stay within token budget)
    if intent == "Memory":
        system_prompt = "You are a helpful customer support supervisor. Answer the user's question based strictly on the conversation history."
        recent_history = messages[-8:] if len(messages) > 8 else messages
        response = llm.invoke([SystemMessage(content=system_prompt)] + recent_history)
        return {"final_response": response.content, "messages": [response]}
        
    draft_response = messages[-1].content
    approved = False
    
    if approval_required:
        approved = any("APPROVED" in msg.content for msg in messages if isinstance(msg, AIMessage))
        if not approved:
            draft_response = "Your request requires management approval. We will update you once it is reviewed."
        else:
            draft_response += "\n\nNote: This request has been officially approved by management."

    system_prompt = f"""You are a Customer Support Supervisor at ABC Technologies.
    Review the draft response below and rewrite it to be highly professional, empathetic, and clear.
    Customer Sentiment: {sentiment}
    Triage Priority: {priority}
    If the customer is Frustrated or Urgent, acknowledge their frustration with extra care and de-escalate effectively.
    Draft Response: {draft_response}
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please provide the finalized response for the customer.")
    ])
    return {"final_response": response.content, "messages": [response]}
