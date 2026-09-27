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

### 2026-09-27: Chunk Size Comparison (500/50 vs 800/100)

- **What Was Compared**:
  Two ChromaDB collections were evaluated:
  - Baseline (`rag_documents`): chunk size 500 characters, overlap 50 characters.
  - Experimental (`rag_documents_v2`): chunk size 800 characters, overlap 100 characters.
- **Methodology**:
  A 9-question retrieval evaluation set (`eval_set.py`) consisting of `(question, expected_keyword)` pairs was executed against both collections via `run_eval.py`. Questions were embedded using `gemini-embedding-001`, retrieving the top-8 chunks (`n_results=8`) per question. Each query was scored as a hit if the expected keyword appeared within any of the top-8 retrieved chunks.
- **Results**:
  - `500/50` (Baseline): scored **8/9** (88.9% hit rate).
  - `800/100` (Experimental): scored **9/9** (100.0% hit rate).
- **The Retrieval Miss**:
  On question 7 (`"What model is fine-tuned as the RAG-aware language model?"`), the 500/50 collection failed to retrieve the chunk containing `"Llama3-70B-Instruct"` within the top-8 results. In contrast, the 800/100 collection retrieved the chunk containing `"Llama3-70B-Instruct"` at rank 5.
- **Explanation**:
  The likely cause is chunk fragmentation. With smaller 500-character chunks, the model name and its surrounding context (the LoRA fine-tuning details) were split across adjacent chunk boundaries, diluting the semantic match of each individual chunk against the question embedding. The larger 800-character chunks preserved more surrounding context per chunk, allowing the dense retriever to recognize the full semantic relationship between the fine-tuning query and the target chunk.
- **Tradeoffs & Caveat**:
  This benchmark relies on a targeted 9-question evaluation set. While it demonstrates the mechanism by which larger chunk windows prevent semantic boundary splits, it is not a definitive claim that 800/100 is universally superior. Larger chunks carry more tokens per document, which can introduce noise or reduce retrieval precision for granular, localized facts in larger corpora.
