import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from src.graph import app

def main():
    queries = [
        "What are the pricing plans available for your software?",
        "I forgot my account password.",
        "My application crashes whenever I upload a file.",
        "I need a refund for my annual subscription.",
        "What was my previous support issue?"
    ]

    # Use a single thread_id to persist memory across queries
    config = {"configurable": {"thread_id": "customer_123"}}

    for i, query in enumerate(queries, 1):
        print(f"========== Query {i} ==========")
        print(f"Customer: {query}")
        print("-" * 30)
        
        inputs = {"messages": [HumanMessage(content=query)]}
        
        for event in app.stream(inputs, config=config):
            for node_name, state_update in event.items():
                print(f"[{node_name}] executed.")
                if "intent" in state_update:
                    print(f"  -> Intent Classified: {state_update.get('intent')} | Department: {state_update.get('department')}")
                if "final_response" in state_update:
                    print(f"\nFINAL RESPONSE:\n{state_update['final_response']}\n")
        
        # Check if the graph execution was interrupted (Human-in-the-loop)
        current_state = app.get_state(config)
        if current_state.next:
            print("\n*** HUMAN IN THE LOOP TRIGGERED ***")
            print(f"Execution interrupted before node: {current_state.next[0]}")
            print("Supervisor, do you approve this request? (Simulating approval...)")
            
            # Resume graph execution by passing None
            print("Resuming graph execution...")
            for event in app.stream(None, config=config):
                for node_name, state_update in event.items():
                    print(f"[{node_name}] executed.")
                    if "final_response" in state_update:
                        print(f"\nFINAL RESPONSE (After Approval):\n{state_update['final_response']}\n")
        
        print("\n")

    # Generate Workflow Diagram
    try:
        print("Generating Workflow Diagram...")
        png_data = app.get_graph().draw_mermaid_png()
        with open("workflow_diagram.png", "wb") as f:
            f.write(png_data)
        print("Workflow diagram saved successfully as 'workflow_diagram.png'.")
    except Exception as e:
        print(f"Failed to generate workflow diagram: {e}")

if __name__ == "__main__":
    main()
