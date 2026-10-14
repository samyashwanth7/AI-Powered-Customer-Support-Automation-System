import sys
# Global compatibility shim for langchain_core 0.3+/1.0+ pydantic_v1 deprecation
try:
    import pydantic.v1 as pydantic_v1
    sys.modules['langchain_core.pydantic_v1'] = pydantic_v1
except ImportError:
    pass

import streamlit as st
import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Enterprise Support AI | Autonomous Multi-Agent Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Refined SaaS Minimalist Theme (Linear / Vercel Aesthetic)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background canvas */
.stApp {
    background-color: #080B11;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.06) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(14, 165, 233, 0.05) 0px, transparent 50%);
}

/* Hide default streamlit decorations */
header[data-testid="stHeader"] {
    background: transparent;
}
#MainMenu, footer {visibility: hidden;}

/* Custom Top Nav / Header Card */
.header-container {
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.header-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 10px;
}

.header-subtitle {
    color: #94A3B8;
    font-size: 0.88rem;
    margin-top: 4px;
}

/* Session info bar */
.session-bar {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Badge styling */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-ticket {
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #A5B4FC;
    font-family: 'JetBrains Mono', monospace;
}

.badge-dept {
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #7DD3FC;
}

.badge-sent-pos {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #86EFAC;
}

.badge-sent-neut {
    background: rgba(234, 179, 8, 0.1);
    border: 1px solid rgba(234, 179, 8, 0.3);
    color: #FDE047;
}

.badge-sent-urg {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
    color: #FCA5A5;
}

.badge-prio-high {
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #FDA4AF;
}

.badge-prio-med {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #FCD34D;
}

.badge-prio-low {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #6EE7B7;
}

/* Status indicator pulse */
.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-ready {
    background: #10B981;
    box-shadow: 0 0 10px #10B981;
}
.status-pending {
    background: #F59E0B;
    box-shadow: 0 0 10px #F59E0B;
}

/* Chat container polish */
.stChatMessage {
    background: rgba(17, 24, 39, 0.55) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18) !important;
}

/* User message accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.75)) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
}

/* KPI Card styling for Dashboard */
div[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 6px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94A3B8;
    padding: 8px 18px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.2) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
}

/* Quick prompt pills */
.quick-prompt-btn button {
    border-radius: 9999px !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    background: rgba(30, 41, 59, 0.5) !important;
    color: #CBD5E1 !important;
    font-size: 0.78rem !important;
    padding: 4px 12px !important;
    transition: all 0.2s ease !important;
}
.quick-prompt-btn button:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
    background: rgba(99, 102, 241, 0.15) !important;
    color: #F8FAFC !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize analytics DB
from src.analytics import init_analytics_db, log_ticket_event, log_feedback, get_analytics_metrics
init_analytics_db()

# Initialize session states
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "ticket_id" not in st.session_state:
    st.session_state.ticket_id = f"#TKT-{uuid.uuid4().hex[:6].upper()}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = False

if "last_metadata" not in st.session_state:
    st.session_state.last_metadata = {}

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None

# Import the LangGraph app
from src.graph import app as graph_app

# Header Banner
st.markdown("""
<div class="header-container">
    <div class="header-title">
        <span>🛡️</span>
        <span>Enterprise Support AI & Operational Triage</span>
    </div>
    <div class="header-subtitle">
        Autonomous multi-agent LangGraph architecture with real-time sentiment triage, RAG knowledge grounding, Human-in-the-Loop policy guardrails & live operations observability.
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_chat, tab_dashboard = st.tabs(["💬 Customer Support Portal", "📊 Supervisor & Operations Dashboard"])

with tab_chat:
    # Top Session Bar
    status_dot = '<span class="status-indicator status-pending"></span>Review Required' if st.session_state.pending_approval else '<span class="status-indicator status-ready"></span>Live'
    
    col_bar1, col_bar2 = st.columns([3, 1])
    with col_bar1:
        st.markdown(f"""
        <div class="session-bar">
            <div>
                <span class="badge-pill badge-ticket">{st.session_state.ticket_id}</span>
                <span style="color:#64748B; margin: 0 8px;">|</span>
                <span style="font-size:0.82rem; color:#94A3B8;">Thread: <code style="color:#A5B4FC; background:transparent;">{st.session_state.thread_id[:8]}...</code></span>
                <span style="color:#64748B; margin: 0 8px;">|</span>
                <span style="font-size:0.82rem; color:#E2E8F0;">{status_dot}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_bar2:
        if st.button("🔄 New Ticket Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.ticket_id = f"#TKT-{uuid.uuid4().hex[:6].upper()}"
            st.session_state.pending_approval = False
            st.session_state.last_metadata = {}
            st.session_state.selected_prompt = None
            st.rerun()

    # One-Click Quick Test Chips for Recruiters & Visitors
    st.markdown("<p style='font-size:0.78rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>⚡ One-Click Scenarios</p>", unsafe_allow_html=True)
    chip_col1, chip_col2, chip_col3, chip_col4, chip_col5 = st.columns(5)
    
    with chip_col1:
        if st.button("💰 Pricing Plans", key="chip_pricing", use_container_width=True):
            st.session_state.selected_prompt = "What are the pricing plans available for your software?"
    with chip_col2:
        if st.button("🔑 Password Recovery", key="chip_pwd", use_container_width=True):
            st.session_state.selected_prompt = "I forgot my account password."
    with chip_col3:
        if st.button("⚠️ App Crash Bug", key="chip_crash", use_container_width=True):
            st.session_state.selected_prompt = "My application crashes whenever I upload a file."
    with chip_col4:
        if st.button("💳 Refund (HITL Demo)", key="chip_refund", use_container_width=True):
            st.session_state.selected_prompt = "I need a refund for my annual subscription."
    with chip_col5:
        if st.button("🧠 Recall Past Issue", key="chip_memory", use_container_width=True):
            st.session_state.selected_prompt = "What was my previous support issue?"

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Display Chat History
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show triage badges on assistant messages
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]
                dept = meta.get('department', 'General')
                sentiment = meta.get('sentiment', 'Neutral')
                priority = meta.get('priority', 'Medium')
                
                s_class = "badge-sent-urg" if sentiment in ["Frustrated", "Urgent"] else ("badge-sent-pos" if sentiment == "Positive" else "badge-sent-neut")
                p_class = "badge-prio-high" if priority == "High" else ("badge-prio-med" if priority == "Medium" else "badge-prio-low")
                
                st.markdown(f"""
                <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; align-items:center;">
                    <span class="badge-pill badge-dept">🏢 {dept}</span>
                    <span class="badge-pill {s_class}">🎭 {sentiment}</span>
                    <span class="badge-pill {p_class}">⚡ {priority} Priority</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback loop
                fb_col1, fb_col2, _ = st.columns([1, 1, 10])
                with fb_col1:
                    if st.button("👍", key=f"thumb_up_{i}"):
                        log_feedback(st.session_state.ticket_id, 1, "Helpful response")
                        st.toast("Thank you for your feedback! (Recorded in SQLite audit)", icon="✅")
                with fb_col2:
                    if st.button("👎", key=f"thumb_down_{i}"):
                        log_feedback(st.session_state.ticket_id, 0, "Unhelpful response")
                        st.toast("Feedback logged into model evaluation queue.", icon="📝")

    # Handle Human-in-the-Loop approval button
    if st.session_state.pending_approval:
        st.markdown("""
        <div style="background:rgba(239, 68, 68, 0.08); border:1px solid rgba(239, 68, 68, 0.3); border-radius:12px; padding:16px 20px; margin: 16px 0;">
            <div style="font-weight:600; color:#FCA5A5; font-size:0.95rem; display:flex; align-items:center; gap:8px;">
                <span>⚠️</span> <span>Management Clearance Required (Human-in-the-Loop)</span>
            </div>
            <div style="font-size:0.84rem; color:#CBD5E1; margin-top:6px;">
                The LangGraph state machine detected a policy-restricted action (e.g., refund, termination, or compensation). Workflow execution is suspended awaiting supervisor authorization.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_app1, col_app2 = st.columns([2, 3])
        with col_app1:
            if st.button("✅ Authorize Request (Simulate Supervisor)", type="primary", use_container_width=True):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                with st.spinner("Authorizing request through supervisor node..."):
                    final_response = None
                    for event in graph_app.stream(None, config=config):
                        for node_name, state_update in event.items():
                            if "final_response" in state_update:
                                final_response = state_update["final_response"]
                    
                    if final_response:
                        meta = st.session_state.last_metadata
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_response,
                            "metadata": meta
                        })
                        # Log to analytics DB
                        log_ticket_event(
                            st.session_state.ticket_id,
                            st.session_state.thread_id,
                            "Refund / Policy Exception Approval",
                            meta.get("department", "Billing"),
                            "Billing",
                            meta.get("sentiment", "Urgent"),
                            "High",
                            True,
                            final_response
                        )
                        st.session_state.pending_approval = False
                        st.rerun()

    # Determine user query (either typed or clicked from one-click chips)
    user_query = st.chat_input("Ask a question, describe an error, or test policies...")
    if st.session_state.selected_prompt:
        user_query = st.session_state.selected_prompt
        st.session_state.selected_prompt = None

    if user_query:
        if st.session_state.pending_approval:
            st.error("Please authorize or resolve the pending supervisor action above before entering new queries.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            inputs = {"messages": [HumanMessage(content=user_query)]}

            with st.chat_message("assistant"):
                with st.spinner("Triage classifying, retrieving knowledge, and synthesizing response..."):
                    final_response = None
                    extracted_meta = {}

                    for event in graph_app.stream(inputs, config=config):
                        for node_name, state_update in event.items():
                            if "department" in state_update:
                                extracted_meta["department"] = state_update.get("department")
                            if "sentiment" in state_update:
                                extracted_meta["sentiment"] = state_update.get("sentiment")
                            if "priority" in state_update:
                                extracted_meta["priority"] = state_update.get("priority")
                            if "final_response" in state_update:
                                final_response = state_update["final_response"]

                    st.session_state.last_metadata = extracted_meta
                    current_state = graph_app.get_state(config)

                    if current_state.next:
                        st.session_state.pending_approval = True
                        st.rerun()
                    elif final_response:
                        st.markdown(final_response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_response,
                            "metadata": extracted_meta
                        })
                        
                        # Log event in SQLite analytics table
                        log_ticket_event(
                            st.session_state.ticket_id,
                            st.session_state.thread_id,
                            user_query,
                            extracted_meta.get("department", "General"),
                            extracted_meta.get("intent", "General"),
                            extracted_meta.get("sentiment", "Neutral"),
                            extracted_meta.get("priority", "Medium"),
                            False,
                            final_response
                        )
                        st.rerun()

with tab_dashboard:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h3 style="margin: 0; color: #F8FAFC; font-weight: 700;">📊 Operational Intelligence & Telemetry</h3>
        <p style="color: #94A3B8; font-size: 0.86rem; margin-top: 4px;">Live triage metrics, sentiment telemetry, and audit trail extracted directly from the persistent SQLite database.</p>
    </div>
    """, unsafe_allow_html=True)
    
    analytics = get_analytics_metrics()
    
    # KPI Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Total Inbound Queries", value=analytics["total_tickets"])
    with m2:
        st.metric(label="HITL Escalations Paused", value=analytics["escalated_count"])
    with m3:
        st.metric(label="Customer Satisfaction (👍)", value=f"{analytics['satisfaction_rate']}%")
    with m4:
        st.metric(label="User Feedback Collected", value=analytics["total_feedback"])

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Visual Breakdown Charts
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("##### 🏢 Volume by Department")
        if analytics["departments"]:
            df_dept = pd.DataFrame(list(analytics["departments"].items()), columns=["Department", "Tickets"])
            st.bar_chart(df_dept.set_index("Department"), color="#6366F1")
        else:
            st.info("No department tickets logged yet.")

    with c2:
        st.markdown("##### 🎭 Customer Sentiment")
        if analytics["sentiments"]:
            df_sent = pd.DataFrame(list(analytics["sentiments"].items()), columns=["Sentiment", "Count"])
            st.bar_chart(df_sent.set_index("Sentiment"), color="#EC4899")
        else:
            st.info("No sentiment data logged yet.")

    with c3:
        st.markdown("##### ⚡ Priority Distribution")
        if analytics["priorities"]:
            df_prio = pd.DataFrame(list(analytics["priorities"].items()), columns=["Priority", "Count"])
            st.bar_chart(df_prio.set_index("Priority"), color="#06B6D4")
        else:
            st.info("No priority data logged yet.")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Audit Log Table
    st.markdown("##### 📋 Live Session Audit Trail")
    if analytics["recent_tickets"]:
        df_audit = pd.DataFrame(
            analytics["recent_tickets"],
            columns=["Timestamp", "Ticket ID", "Department", "Sentiment", "Priority", "HITL Escalated", "Customer Query"]
        )
        st.dataframe(df_audit, use_container_width=True)
    else:
        st.info("No ticket events recorded yet. Start interacting in the portal to populate telemetry data!")
