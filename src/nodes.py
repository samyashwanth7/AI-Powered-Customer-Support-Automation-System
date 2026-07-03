import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from src.state import AgentState
from src.rag import retriever

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

class IntentOutput(BaseModel):
    intent: str = Field(description="The category of the customer query: Sales, Technical, Billing, Account, or Memory")
    department: str = Field(description="The department to route to: Sales, Technical, Billing, Account, or None")

def intent_classifier(state: AgentState):
    """Categorizes the query and determines routing."""
    # Find the latest HumanMessage
    last_message = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    
    system_prompt = """You are an intent classifier for ABC Technologies.
    Categorize the user's query into one of the following: Sales, Technical, Billing, Account, or Memory.
    - Sales: Product info, pricing plans.
    - Technical: Application errors, installation, crashes.
    - Billing: Invoices, refunds, cancellations.
    - Account: Passwords, profile updates, activation.
    - Memory: Asking about previous interactions (e.g., 'What was my previous issue?').
    Route Memory queries to 'None' department so they can be handled directly by the Supervisor based on history.
    """
    
    structured_llm = llm.with_structured_output(IntentOutput)
    response = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    
    return {"intent": response.intent, "department": response.department}

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
    query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    
    system_prompt = f"""You are a customer support agent for the {department} department at ABC Technologies.
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
    # When this node runs, it means the human has approved the action.
    approval_msg = AIMessage(content="System: The requested action has been APPROVED by the human supervisor.")
    return {"messages": [approval_msg]}

def supervisor_agent(state: AgentState):
    """Validates and improves responses before sending."""
    messages = state["messages"]
    intent = state.get("intent", "")
    approval_required = state.get("approval_required", False)
    
    # For memory queries, read the conversation history
    if intent == "Memory":
        system_prompt = "You are a helpful customer support supervisor. Answer the user's question based strictly on the conversation history."
        response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
        return {"final_response": response.content, "messages": [response]}
        
    draft_response = messages[-1].content
    approved = False
    
    if approval_required:
        approved = any("APPROVED" in msg.content for msg in messages if isinstance(msg, AIMessage))
        if not approved:
            draft_response = "Your request requires management approval. We will update you once it is reviewed."
        else:
            draft_response += "\n\nNote: This request has been officially approved by management."

    system_prompt = f"""You are a Customer Support Supervisor.
    Review the draft response below and rewrite it to be highly professional, empathetic, and clear.
    Ensure the tone is supportive and represents ABC Technologies well.
    Draft Response: {draft_response}
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt)])
    return {"final_response": response.content, "messages": [response]}
