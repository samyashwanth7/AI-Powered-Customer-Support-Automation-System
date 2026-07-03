# AI-Powered Customer Support Automation System

## Overview
This project implements an AI-Powered Customer Support Automation System using LangGraph and LangChain. It categorizes customer queries, routes them to specialized departmental agents (Sales, Technical, Billing, Account), retrieves relevant information using a RAG pipeline from company documents, and maintains interaction history in a SQLite database. It also incorporates a human-in-the-loop workflow for high-risk requests.

## Features
- **Intent Classification**: Categorizes user requests to route to the appropriate department.
- **RAG Pipeline**: Retrieves context from local documents (Company Policy, FAQ, Pricing, Technical Manual) using ChromaDB and HuggingFace embeddings.
- **Memory**: Uses SQLite to persist conversation history across interactions.
- **Human-in-the-Loop**: Automatically pauses execution for refunds, cancellations, and escalations, waiting for a human supervisor's approval.
- **Supervisor Agent**: Validates and improves the final response before sending it to the user.

## Project Structure
- `src/state.py`: Defines the LangGraph State (`AgentState`).
- `src/rag.py`: Implements the Document Loaders and Vectorstore Retriever.
- `src/nodes.py`: Contains the intent classifier, department agents, and supervisor node logic.
- `src/graph.py`: Connects the nodes and edges, compiling the LangGraph workflow with a SQLite checkpointer.
- `docs/`: Contains the dummy knowledge base text documents.
- `demo.py`: Main execution script to demonstrate the 5 sample queries.
- `workflow_diagram.png`: Visual representation of the LangGraph architecture.
- `memory.db`: SQLite database storing interaction states.

## Setup Instructions
1. Ensure you have Python 3.9+ installed.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install langgraph langchain langchain-groq langchain-community chromadb sentence-transformers pydantic python-dotenv httpx langgraph-checkpoint-sqlite
   ```
4. Configure your API key in a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

## Run Instructions
Run the demonstration script:
```bash
python demo.py
```
This script will sequentially process the five required test queries, demonstrate the RAG retrieval, trigger the human-in-the-loop interruption, and demonstrate memory recall.

## Submission Checklist Notes
- **Workflow Diagram**: Located at `workflow_diagram.png`.
- **SQLite Memory File**: Located at `memory.db`.
- **Source Code**: Packaged in `submission.zip`.
- **Task Output Screenshots**: Please run `python demo.py` and take screenshots of your terminal showing the output for each query, as requested by the assignment guidelines.
