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
    page_title="Enterprise AI Support & Triage",
    page_icon="🛡️",
    layout="wide"
)

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

# Import the LangGraph app
from src.graph import app as graph_app

# Header
st.title("🛡️ Enterprise AI Support Automation & Triage")
st.caption("Multi-Agent LangGraph System with Real-Time Sentiment Triage, RAG Knowledge Base, Human-in-the-Loop Guardrails & Operations Analytics.")

# Tabs
tab_chat, tab_dashboard = st.tabs(["💬 Customer Support Portal", "📊 Supervisor & Operations Dashboard"])

with tab_chat:
    col_info1, col_info2, col_info3 = st.columns([2, 1, 1])
    with col_info1:
        st.markdown(f"**Active Session:** `{st.session_state.ticket_id}` | Thread: `{st.session_state.thread_id[:8]}...`")
    with col_info2:
        if st.session_state.pending_approval:
            st.error("Status: ⚠️ Action Pending Supervisor Approval")
        else:
            st.success("Status: 🟢 AI Agent Ready")
    with col_info3:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.ticket_id = f"#TKT-{uuid.uuid4().hex[:6].upper()}"
            st.session_state.pending_approval = False
            st.session_state.last_metadata = {}
            st.rerun()

    st.divider()

    # Display Chat History
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show triage badges on assistant messages
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.caption(f"🏢 **Dept:** {meta.get('department', 'General')}")
                with col_m2:
                    sentiment = meta.get('sentiment', 'Neutral')
                    s_icon = "🔴" if sentiment in ["Frustrated", "Urgent"] else ("🟢" if sentiment == "Positive" else "🟡")
                    st.caption(f"{s_icon} **Sentiment:** {sentiment}")
                with col_m3:
                    priority = meta.get('priority', 'Medium')
                    p_icon = "🔴" if priority == "High" else ("🟡" if priority == "Medium" else "🟢")
                    st.caption(f"{p_icon} **Priority:** {priority}")
                
                # Feedback loop
                fb_col1, fb_col2, _ = st.columns([1, 1, 8])
                with fb_col1:
                    if st.button("👍", key=f"thumb_up_{i}"):
                        log_feedback(st.session_state.ticket_id, 1, "Helpful response")
                        st.toast("Thank you for your feedback! (Recorded in DB)", icon="✅")
                with fb_col2:
                    if st.button("👎", key=f"thumb_down_{i}"):
                        log_feedback(st.session_state.ticket_id, 0, "Unhelpful response")
                        st.toast("Feedback recorded. Escalating to training data.", icon="📝")

    # Handle Human-in-the-Loop approval button
    if st.session_state.pending_approval:
        st.warning("⚠️ **Management Approval Required**\n\nThe AI system has detected a high-risk policy event (e.g., refund, subscription cancellation, or compensation request). It is held in state suspension awaiting supervisor clearance.")
        
        col_app1, col_app2 = st.columns([2, 3])
        with col_app1:
            if st.button("✅ Approve Request (Simulate Supervisor)", type="primary", use_container_width=True):
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
                            "Refund / Escalation Approval",
                            meta.get("department", "Billing"),
                            "Billing",
                            meta.get("sentiment", "Urgent"),
                            "High",
                            True,
                            final_response
                        )
                        st.session_state.pending_approval = False
                        st.rerun()

    # User Input
    if prompt := st.chat_input("Type your support issue here..."):
        if st.session_state.pending_approval:
            st.error("Please resolve the pending supervisor approval above before submitting new queries.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            inputs = {"messages": [HumanMessage(content=prompt)]}

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
                            prompt,
                            extracted_meta.get("department", "General"),
                            extracted_meta.get("intent", "General"),
                            extracted_meta.get("sentiment", "Neutral"),
                            extracted_meta.get("priority", "Medium"),
                            False,
                            final_response
                        )
                        st.rerun()

with tab_dashboard:
    st.subheader("📊 Operational Intelligence & Triage Analytics")
    st.caption("Live metrics pulled directly from SQLite checkpointer and audit tables.")
    
    analytics = get_analytics_metrics()
    
    # KPI Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Total Queries Processed", value=analytics["total_tickets"])
    with m2:
        st.metric(label="Supervisor Escalations (HITL)", value=analytics["escalated_count"])
    with m3:
        st.metric(label="Satisfaction Rating (👍)", value=f"{analytics['satisfaction_rate']}%")
    with m4:
        st.metric(label="Total Feedback Collected", value=analytics["total_feedback"])

    st.divider()

    # Visual Breakdown Charts
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("#### 🏢 Queries by Department")
        if analytics["departments"]:
            df_dept = pd.DataFrame(list(analytics["departments"].items()), columns=["Department", "Count"])
            st.bar_chart(df_dept.set_index("Department"))
        else:
            st.info("No department data logged yet.")

    with c2:
        st.markdown("#### 🎭 Customer Sentiment Triage")
        if analytics["sentiments"]:
            df_sent = pd.DataFrame(list(analytics["sentiments"].items()), columns=["Sentiment", "Count"])
            st.bar_chart(df_sent.set_index("Sentiment"))
        else:
            st.info("No sentiment data logged yet.")

    with c3:
        st.markdown("#### ⚡ Priority Distribution")
        if analytics["priorities"]:
            df_prio = pd.DataFrame(list(analytics["priorities"].items()), columns=["Priority", "Count"])
            st.bar_chart(df_prio.set_index("Priority"))
        else:
            st.info("No priority data logged yet.")

    st.divider()

    # Audit Log Table
    st.markdown("#### 📋 Recent Triage Audit Trail")
    if analytics["recent_tickets"]:
        df_audit = pd.DataFrame(
            analytics["recent_tickets"],
            columns=["Timestamp", "Ticket ID", "Department", "Sentiment", "Priority", "HITL Escalated", "Customer Query"]
        )
        st.dataframe(df_audit, use_container_width=True)
    else:
        st.info("No ticket audit events recorded yet. Start chatting to populate real-time metrics!")
