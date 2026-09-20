---
name: embedding-and-storage
description: Mandatory standard for embedding generation and vector storage in the RAG QA tool. Enforces google-genai SDK, gemini-embedding-001 model, persistent ChromaDB local storage, and structured metadata.
---

# Embedding Generation & Vector Storage Standard

## Mandatory Project Directive

> [!IMPORTANT]
> **All future embedding generation and vector database storage code in this project MUST adhere to this standard unless explicitly told otherwise.**
> - Use the modern **`google-genai`** SDK (`from google import genai`). **DO NOT** use the deprecated `google-generativeai` package.
> - Use **`gemini-embedding-001`** as the standard embedding model.
> - Use **ChromaDB** with a **persistent local client** (`chromadb.PersistentClient(path=...)`), never an in-memory client, so embeddings survive between runs.
> - Every stored chunk **must include metadata** with at least `source` (source filename) and `chunk_index` (integer offset/index).

---

## Standard Specifications

| Component | Standard Specification | Notes |
| :--- | :--- | :--- |
| **SDK** | `google-genai` (`genai.Client`) | Deprecated legacy package `google-generativeai` is strictly disallowed. |
| **Embedding Model** | `gemini-embedding-001` | Used with `client.models.embed_content()`. |
| **Vector Database** | `chromadb.PersistentClient` | Must persist to a local folder (e.g., `./chroma_db`). In-memory storage is prohibited. |
| **Chunk Metadata** | `source` (str), `chunk_index` (int) | Enables citation tracking, source attribution, and ordered chunk retrieval. |
| **Chunk ID Convention** | `{source_basename}_chunk_{index}` | Unique, deterministic identifier for each stored chunk. |

---

## Key Rationale & Principles

1. **Modern `google-genai` SDK**:
   - The original `google-generativeai` SDK is deprecated and replaced by `google-genai`. The new SDK provides unified API surfaces, improved typing, and official support for latest Gemini multimodal and embedding models.

2. **`gemini-embedding-001`**:
   - High-dimensional vector representation optimized for semantic search and retrieval-augmented generation.

3. **Persistent ChromaDB (`PersistentClient`)**:
   - In-memory databases (`chromadb.EphemeralClient` or `chromadb.Client()`) discard all embeddings when the Python process exits.
   - `chromadb.PersistentClient(path="chroma_db")` writes indices and SQLite metadata to disk, ensuring data survives across restarts, avoids costly re-embedding calls, and facilitates reproducible testing.

4. **Required Metadata (`source` & `chunk_index`)**:
   - In production RAG systems, grounding answers in verifiable citations requires tracking the origin document name (`source`).
   - Tracking `chunk_index` allows sorting retrieved results, reconstructing neighboring text for broader context, and debugging retrieval quality.

---

## Reference Implementation

The following reference implementation shows the standard pattern for embedding text chunks and persisting them into ChromaDB:

```python
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
import chromadb

# 1. Load environment variables (API keys)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Check your .env file.")

# 2. Initialize the modern Google GenAI client
genai_client = genai.Client(api_key=api_key)

# 3. Initialize Persistent ChromaDB Client (survives between process runs)
DB_PATH = Path(__file__).resolve().parent / "chroma_db"
chroma_client = chromadb.PersistentClient(path=str(DB_PATH))

# 4. Get or create collection
collection = chroma_client.get_or_create_collection(
    name="rag_documents",
    metadata={"description": "PDF chunks with gemini-embedding-001 vectors"}
)


def generate_embedding(text: str) -> List[float]:
    """
    Generates an embedding vector using gemini-embedding-001 and google-genai SDK.
    """
    response = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    # response.embeddings contains ContentEmbedding objects with .values
    return response.embeddings[0].values


def store_chunks_in_chromadb(
    chunks: List[str],
    source_filename: str,
    collection: chromadb.Collection,
    batch_size: int = 20
) -> None:
    """
    Embeds and persists chunks into ChromaDB with required metadata.

    Required Metadata:
        - source: filename (e.g. 'rag_research_paper.pdf')
        - chunk_index: integer index of the chunk
    """
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{source_filename}_chunk_{idx}"
        embedding = generate_embedding(chunk)
        
        metadata: Dict[str, Any] = {
            "source": source_filename,
            "chunk_index": idx,
            "char_count": len(chunk)
        }
        
        collection.add(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[metadata]
        )
```

---

## Guidelines for Development

- **Import check**: Always use `from google import genai` — never `import google.generativeai`.
- **Client check**: Always instantiate `chromadb.PersistentClient(path=...)`.
- **Batching**: When embedding large volumes of chunks, consider batching requests to optimize API throughput and observe rate limits.
- **Git ignore**: The persistent directory (e.g., `chroma_db/`) must remain excluded in `.gitignore` to prevent committing binary vector databases to version control.
