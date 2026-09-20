---
name: retrieval-and-generation
description: Mandatory standard for query embedding, ChromaDB vector retrieval, and Gemini-powered answer generation in the RAG QA tool. Enforces gemini-embedding-001 query embeddings, top 8 chunk retrieval from persistent ChromaDB rag_documents collection, strictly grounded prompt construction, and gemini-3.8-flash generation.
---

# Retrieval & Generation Standard

## Mandatory Project Directive

> [!IMPORTANT]
> **All future retrieval and generation code in this project MUST adhere to this standard unless explicitly told otherwise.**
> - Use **`gemini-embedding-001`** to embed the user's query (matching the exact model used for storing chunk embeddings).
> - Query the persistent ChromaDB **`rag_documents`** collection for the **top 8 most similar chunks** (`n_results=8`) using `collection.query()`.
> - Construct a prompt that includes the retrieved chunks as context and strictly instructs the model to answer **only** from that context.
> - Mandate the exact fallback response: if the answer is not present in the provided context, the model must say:  
>   `"I don't know based on the provided documents"`
> - Generate the answer using **`gemini-3.8-flash`** via the modern SDK method **`client.models.generate_content()`**.

---

## Standard Specifications

| Component | Standard Specification | Notes |
| :--- | :--- | :--- |
| **SDK** | `google-genai` (`from google import genai`) | Modern SDK client; `google-generativeai` is strictly prohibited. |
| **Query Embedding Model** | `gemini-embedding-001` | Must match the storage embedding model to ensure vector space symmetry. |
| **Vector Database** | `chromadb.PersistentClient` | Connects to local persistent storage (e.g., `./chroma_db`). |
| **Collection Name** | `rag_documents` | Standard collection where document chunks and embeddings reside. |
| **Retrieval Depth** | `n_results=8` | Retrieves the top 8 most semantically relevant text chunks. |
| **Retrieval Method** | `collection.query(query_embeddings=[...], n_results=8)` | Vector similarity query against indexed embeddings. |
| **Grounding Rule** | Strict context boundary | Must not use external knowledge or fabricate information. |
| **Fallback Response** | `"I don't know based on the provided documents"` | Exact response when the context does not contain the answer. |
| **Generation Model** | `gemini-3.8-flash` | Fast, high-accuracy model invoked via `client.models.generate_content()`. |

---

## Key Rationale & Principles

1. **Vector Space Symmetry (`gemini-embedding-001`)**:
   - Query vectors and stored document vectors must share identical dimensionality and embedding space geometry. Using `gemini-embedding-001` for both query embedding and chunk storage guarantees accurate cosine distance scoring.

2. **Top-8 Chunk Retrieval (`n_results=8`) from `rag_documents`**:
   - Standard retrieval depth is configured to 8 chunks. This was intentionally increased from 5 to 8 after observing a retrieval miss during testing: a question containing "PIER-QA" retrieved chunks that repeated that keyword over the chunk that actually contained the answer (about RAPTOR), which was ranked outside the top 5.
   - Expanding retrieval depth to `n_results=8` ensures that vital answering passages are not pushed out of the prompt context by chunks with high superficial keyword frequency, preserving context breadth while remaining well within prompt token limits.

3. **Strict Grounding & Anti-Hallucination Guardrail**:
   - RAG applications must be truthful and grounded in the source documentation.
   - Instructing the model to state `"I don't know based on the provided documents"` prevents speculative hallucinations when users ask out-of-scope or ungrounded questions.

4. **Modern Generation with `gemini-3.8-flash`**:
   - `gemini-3.8-flash` offers state-of-the-art inference speed, strong reasoning, and strict instruction-following capabilities.
   - Calling `client.models.generate_content()` complies with the modern `google-genai` SDK architecture.

---

## Reference Implementation

The following reference pattern demonstrates the standard workflow for query embedding, ChromaDB retrieval, prompt assembly, and response generation:

```python
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
import chromadb

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

# 2. Initialize the modern Google GenAI Client
genai_client = genai.Client(api_key=api_key)

# 3. Connect to Persistent ChromaDB and target collection
DB_PATH = Path(__file__).resolve().parent / "chroma_db"
chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
collection = chroma_client.get_collection(name="rag_documents")


def embed_query(client: genai.Client, query: str) -> List[float]:
    """
    Embeds the user's query using gemini-embedding-001 (same model as storage).
    """
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    )
    return response.embeddings[0].values


def retrieve_top_chunks(
    collection: chromadb.Collection,
    query_embedding: List[float],
    n_results: int = 8
) -> List[str]:
    """
    Queries ChromaDB rag_documents for the top 8 most similar chunks.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # results["documents"] is a list of lists (one list per query embedding)
    documents = results.get("documents", [[]])
    return documents[0] if documents else []


def construct_prompt(query: str, retrieved_chunks: List[str]) -> str:
    """
    Constructs a strictly grounded prompt instructing the model to answer
    only from the provided context or output the mandatory fallback string.
    """
    context_text = "\n\n---\n\n".join(retrieved_chunks)
    
    prompt = f"""You are a helpful assistant that answers questions strictly using the provided context.

Context:
{context_text}

Question:
{query}

Instructions:
1. Answer the question using ONLY the information provided in the Context above.
2. If the answer cannot be determined directly from the Context, you MUST respond with:
   "I don't know based on the provided documents"
3. Do not make assumptions, extrapolate, or use outside knowledge.
"""
    return prompt


def generate_answer(client: genai.Client, prompt: str) -> str:
    """
    Generates the final response using gemini-3.8-flash via client.models.generate_content().
    """
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )
    return response.text


def rag_query(query: str) -> str:
    """
    End-to-end standard retrieval and generation pipeline.
    """
    # Step A: Embed query with gemini-embedding-001
    query_vec = embed_query(genai_client, query)
    
    # Step B: Retrieve top 8 chunks from rag_documents
    chunks = retrieve_top_chunks(collection, query_vec, n_results=8)
    
    # Step C: Construct grounded prompt
    prompt = construct_prompt(query, chunks)
    
    # Step D: Generate answer with gemini-3.8-flash
    return generate_answer(genai_client, prompt)
```

---

## Guidelines for Development

- **Consistent Embedding Model**: Never mix embedding models (e.g., embedding chunks with `gemini-embedding-001` and queries with another model). Both must use `gemini-embedding-001`.
- **Exact Fallback Phrasing**: The prompt must strictly enforce `"I don't know based on the provided documents"` when context is insufficient.
- **Top 8 Limit**: Maintain `n_results=8` as the default chunk retrieval count.
- **Modern SDK Invocation**: Always use `client.models.generate_content(model="gemini-3.8-flash", contents=...)`. Do not use legacy methods or deprecated packages.
- **Metadata Awareness**: While the prompt uses chunk text, retain metadata (`source`, `chunk_index`) from `results["metadatas"]` whenever source attribution or citations are required in user-facing output.
