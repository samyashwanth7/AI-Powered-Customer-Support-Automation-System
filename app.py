import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Initialize Streamlit page config
st.set_page_config(page_title="AI Support Automation", page_icon="🤖", layout="centered")
st.title("🤖 AI-Powered Customer Support System")
st.markdown("Welcome! How can we assist you today? Our system can help with **Sales**, **Technical**, **Billing**, or **Account** issues.")

# Initialize session state for memory and thread ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = False

# Import the LangGraph app
# We do this after page config to avoid any potential issues
from src.graph import app as graph_app

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Human-in-the-Loop approval button
if st.session_state.pending_approval:
    st.warning("⚠️ **Management Approval Required**\n\nThe system has paused because your request requires management approval (e.g., refund, cancellation).")
    if st.button("Approve Request (Simulate Supervisor)"):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        with st.spinner("Processing approval..."):
            # Resume graph execution by passing None
            final_response = None
            for event in graph_app.stream(None, config=config):
                for node_name, state_update in event.items():
                    if "final_response" in state_update:
                        final_response = state_update["final_response"]
            
            if final_response:
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                # Clear approval state
                st.session_state.pending_approval = False
                st.rerun()

# Handle new user input
if prompt := st.chat_input("Type your support issue here..."):
    if st.session_state.pending_approval:
        st.error("Please approve or resolve the pending request first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run LangGraph workflow
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        inputs = {"messages": [HumanMessage(content=prompt)]}
        
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                final_response = None
                
                # Stream events from the graph
                for event in graph_app.stream(inputs, config=config):
                    for node_name, state_update in event.items():
                        # We only care about displaying the final response
                        if "final_response" in state_update:
                            final_response = state_update["final_response"]
                
                # Check if graph interrupted for approval
                current_state = graph_app.get_state(config)
                if current_state.next:
                    st.session_state.pending_approval = True
                    st.rerun()
                elif final_response:
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})

# Sidebar options
with st.sidebar:
    st.header("Session Management")
    if st.button("Clear Chat & Restart"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending_approval = False
        st.rerun()
