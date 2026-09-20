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

## Usage

1. **Extract and Chunk Document (Standard 500 chars / 50 overlap):**
   ```bash
   python load_and_chunk.py
   ```

2. **Generate Embeddings and Store in ChromaDB (`rag_documents`):**
   ```bash
   python embed_and_store.py
   ```

3. **Run Chunk Size Experiment (800 chars / 100 overlap into `rag_documents_v2`):**
   ```bash
   python experiment_chunk_size.py
   ```

4. **Ask Questions via Grounded RAG Pipeline:**
   ```bash
   # Query default baseline collection (rag_documents)
   python ask.py "What is Retrieval-Augmented Generation?"

   # Query experimental collection (rag_documents_v2)
   python ask.py "What is Retrieval-Augmented Generation?" --collection rag_documents_v2
   ```

## Debugging & Findings

### 2026-09-20: Retrieval Miss on Keyword-Dense Queries (PIER-QA vs. RAPTOR)

- **Failed Question**:
  `"How does PIER-QA index and cluster chunks for retrieval?"` (or questions asking about PIER-QA's chunk indexing/clustering enhancements).
- **What Was Retrieved**:
  When `n_results=5` was used, the retrieval step returned chunks (such as Chunk 15, 16, 17, 0, and 9) that repeatedly contained the system name `"PIER-QA"` and general high-level overview text. Chunk 18, which actually described how **RAPTOR** is used to enhance retrieval by indexing and clustering preprocessed chunks based on semantics, was ranked outside the top 5 (at ranks 6–8). As a result, the model was forced to trigger the fallback phrase (`"I don't know based on the provided documents"`) because the factual answer was absent from the retrieved context.
- **Root Cause**:
  Dense semantic embedding with `gemini-embedding-001` assigned higher similarity scores to chunks with multiple matches and thematic proximity to the specific query tokens (`"PIER-QA"`). Because Chunk 18 focused on the technical implementation of RAPTOR without repeating the token `"PIER-QA"` within its 500-character boundary, it was ranked just below the top 5 cut-off.
- **Remediation**:
  Updated the retrieval depth standard (`n_results`) from `5` to `8` in `ask.py` and `.antigravity/skills/retrieval-and-generation/SKILL.md`. With `n_results=8`, Chunk 18 is reliably included in the prompt context, allowing the generator model (`gemini-3.8-flash`) to locate the RAPTOR indexing details and formulate the correct answer while staying comfortably within prompt token constraints.
