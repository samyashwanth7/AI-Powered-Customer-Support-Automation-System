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

def extract_tokens(response) -> int:
    """Extracts token count from LLM response metadata or estimates safely."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens = response.usage_metadata.get("total_tokens", 0)
        if tokens:
            return int(tokens)
    if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
        tokens = response.response_metadata["token_usage"].get("total_tokens", 0)
        if tokens:
            return int(tokens)
    content = getattr(response, "content", "")
    return max(len(str(content)) // 4, 20)

class IntentOutput(BaseModel):
    intent: str = Field(
        description="The category of the customer query: Sales, Technical, Billing, Account, or Memory. "
                    "Sales: pricing, plans, tiers, discounts, features. "
                    "Technical: errors (404, 500), app crashes, uploads, SSO login, IT settings. "
                    "Billing: invoices, payment methods, cards, refunds, subscription cancellations, compensation. "
                    "Account: passwords, profile settings, account activation, account closure, data deletion, manager escalation. "
                    "Memory: asking about past issues, previous conversation, earlier discussion."
    )
    department: str = Field(description="The department to route to: Sales, Technical, Billing, Account, or None")
    sentiment: str = Field(description="The customer's emotional tone: Frustrated, Neutral, Positive, or Urgent")
    priority: str = Field(description="Priority level for ticket triage: High, Medium, or Low")
    approval_required: bool = Field(
        description="Set to True ONLY IF the customer is explicitly commanding, requesting, or demanding an irreversible administrative action or management escalation: demanding a refund / money back, demanding financial compensation, immediately canceling an active subscription, permanently closing an account or deleting data, or demanding to speak to a manager. Set to False for informational or how-to queries (e.g. 'What is your refund policy?', 'How do I cancel?')."
    )

def intent_classifier(state: AgentState):
    """Categorizes query, extracts customer sentiment, and assigns triage priority and approval gating."""
    last_message = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    
    system_prompt = """You are an AI Support Triage Agent for ABC Technologies.
    Analyze the user's query and output:
    1. intent: Sales, Technical, Billing, Account, or Memory.
       - Sales: Product info, pricing plans, storage limits, discounts, tier differences.
       - Technical: Application errors (Error 404, Error 500), installation, app crashes, uploads, SSO login, browser cache.
       - Billing: Invoices, invoice history, payment methods, credit cards, refund requests, subscription cancellations, billing compensation.
       - Account: Passwords, profile updates, account activation/deactivation, permanent closure, data deletion, speaking with account manager.
       - Memory: Asking about previous interactions, past problems reported, earlier discussion (e.g., 'What was my previous issue?').
    2. department: Sales, Technical, Billing, Account, or None (route Memory queries to None).
    3. sentiment: Frustrated, Neutral, Positive, or Urgent.
    4. priority:
       - High: System outages, crashes, refund demands, urgent escalations, strong frustration.
       - Medium: Account recovery, invoice requests, setup questions.
       - Low: General product information, pricing exploration, casual questions.
    5. approval_required:
       - True ONLY IF customer is explicitly requesting an irreversible action or escalation:
         e.g., 'I need a refund', 'Can I get my money back?', 'Please cancel my subscription right now', 'I want compensation', 'I demand a refund', 'I want to close my account permanently', 'Please delete all my data', 'I want to speak to your manager'.
       - False for informational, general, or how-to queries:
         e.g., 'What is your refund policy?', 'Are monthly subscriptions refundable?', 'How do I cancel my free trial?', 'How do I deactivate my account?', 'How long do you keep my data?'.
    """
    
    structured_llm = llm.with_structured_output(IntentOutput, include_raw=True)
    raw_res = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    response = raw_res["parsed"]
    tokens = extract_tokens(raw_res.get("raw"))
    
    ticket_id = state.get("ticket_id") or f"#TKT-{random.randint(1000, 9999)}"
    
    # Deterministic safety rule reinforcement on approval_required
    query_lower = last_message.lower().strip()
    approval = response.approval_required
    
    action_triggers = [
        "money back", "refund for my", "demand a refund", "cancel my subscription right now",
        "please cancel my subscription", "want compensation", "demand compensation",
        "close my account permanently", "delete all my data", "speak to your manager",
        "talk to your manager", "speak to a manager"
    ]
    info_triggers = [
        "what is your refund policy", "are monthly subscriptions refundable",
        "how do i cancel", "how do i deactivate", "how long do you keep my data",
        "why was my card declined"
    ]
    if any(it in query_lower for it in info_triggers):
        approval = False
    elif any(at in query_lower for at in action_triggers):
        approval = True

    return {
        "intent": response.intent,
        "department": response.department,
        "sentiment": response.sentiment,
        "priority": response.priority,
        "approval_required": approval,
        "ticket_id": ticket_id,
        "total_tokens": tokens,
        "search_query": last_message,
        "retry_count": 0,
        "is_grounded": True
    }

def retrieve_context(state: AgentState):
    """Retrieves context for the given query or reformulated search query."""
    query = state.get("search_query")
    if not query:
        query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def department_agent(state: AgentState):
    """Generic department agent that uses retrieved context with strict grounding."""
    department = state.get("department", "General")
    context = state.get("retrieved_context", "")
    sentiment = state.get("sentiment", "Neutral")
    query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    
    system_prompt = f"""You are a customer support agent for the {department} department at ABC Technologies.
    Detected customer tone: {sentiment}.
    Answer the user's query using ONLY the provided documentation context below.
    - If the context contains relevant information, answer directly, factually, and accurately, citing relevant plan names, prices, versions, limits, or steps.
    - If the context DOES NOT contain the answer, explicitly state that you do not have this information in your documentation and offer to escalate to support.
    Context:
    {context}
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    tokens = extract_tokens(response)
    
    # Grounding evaluation: check if response indicates missing knowledge
    content_lower = response.content.lower()
    refusal_keywords = [
        "do not have", "don't have", "no information", "not covered",
        "cannot find", "unable to find", "not specified", "not available in"
    ]
    is_grounded = not any(rk in content_lower for rk in refusal_keywords)
    
    return {
        "messages": [response],
        "total_tokens": tokens,
        "is_grounded": is_grounded
    }

def rewrite_query(state: AgentState):
    """Cyclic recovery node: reformulates search query when initial retrieval was insufficient."""
    original_query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    current_retry = state.get("retry_count", 0) + 1
    
    system_prompt = """You are a search query optimizer for customer support documentation.
    The previous search was insufficient.
    Rewrite the user's query into 2 to 4 essential search keywords that capture the core subject (e.g. 'crashes upload' -> 'upload file crash desktop client').
    Output ONLY the search keywords without punctuation or quotes.
    """
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=original_query)])
    tokens = extract_tokens(response)
    optimized_query = response.content.strip().replace('"', '').replace("'", "")
    
    return {
        "search_query": optimized_query,
        "retry_count": current_retry,
        "total_tokens": tokens
    }

def human_approval(state: AgentState):
    """Node executed after a human has intervened to approve."""
    approval_msg = AIMessage(content="System: The requested action has been APPROVED by the human supervisor.")
    return {"messages": [approval_msg]}

def supervisor_agent(state: AgentState):
    """Validates, polishes, and finalizes responses before sending."""
    messages = state["messages"]
    intent = state.get("intent", "")
    approval_required = state.get("approval_required", False)
    sentiment = state.get("sentiment", "Neutral")
    priority = state.get("priority", "Medium")
    
    # For memory queries, read the conversation history
    if intent == "Memory":
        system_prompt = "You are a helpful customer support supervisor. Answer the user's question based strictly on the conversation history."
        recent_history = messages[-8:] if len(messages) > 8 else messages
        response = llm.invoke([SystemMessage(content=system_prompt)] + recent_history)
        tokens = extract_tokens(response)
        return {"final_response": response.content, "messages": [response], "total_tokens": tokens}
        
    draft_response = messages[-1].content
    approved = False
    
    if approval_required:
        approved = any("APPROVED" in msg.content for msg in messages if isinstance(msg, AIMessage))
        if not approved:
            draft_response = "Your request requires management approval. We will update you once it is reviewed."
        else:
            draft_response += "\n\nNote: This request has been officially approved by management."

    system_prompt = f"""You are a Customer Support Supervisor at ABC Technologies.
    Review the draft response below and finalize it to be highly professional, empathetic, and clear.
    IMPORTANT: Preserve all specific facts, prices, numbers, limits, versions, and steps from the draft response.
    Customer Sentiment: {sentiment}
    Triage Priority: {priority}
    Draft Response: {draft_response}
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please provide the finalized response for the customer.")
    ])
    tokens = extract_tokens(response)
    return {"final_response": response.content, "messages": [response], "total_tokens": tokens}
