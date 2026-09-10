# RAG QA Tool

A Retrieval-Augmented Generation (RAG) Question-Answering system powered by Google Gemini and ChromaDB.

## Features
- Document loading and chunking (PDF, etc.)
- Vector embeddings and persistent vector search using ChromaDB
- Context-grounded Q&A with Gemini models

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/netaasree/rag-qa-tool.git
   cd rag-qa-tool
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # On Windows PowerShell
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment:**
   Copy `.env.example` to `.env` and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ```
