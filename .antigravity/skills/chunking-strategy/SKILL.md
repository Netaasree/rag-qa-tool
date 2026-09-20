---
name: chunking-strategy
description: Project standard for text chunking in the RAG QA tool. Enforces 500-character chunks with 50-character overlap using pure Python string slicing (no LangChain).
---

# Text Chunking Strategy & Standard

## Mandatory Project Directive

> [!IMPORTANT]
> **All future chunking code in this project MUST follow this standard unless explicitly told otherwise.**
> Do not introduce external text splitting or chunking libraries (such as LangChain) for document chunking.

---

## Standard Specifications

| Parameter | Specification |
| :--- | :--- |
| **Chunk Size** | `500` characters |
| **Chunk Overlap** | `50` characters |
| **Implementation** | Pure Python string slicing (`text[start:end]`) |
| **External Dependencies** | **None** (No LangChain, LlamaIndex, or third-party splitters) |

---

## Rationale & Design Decisions

1. **Fixed-Size Chunking for Simplicity & Predictability**:
   - Fixed-size character chunking provides deterministic chunk boundaries and consistent payload sizes.
   - Avoids hidden heuristics, tokenization discrepancies, and dependency bloat at this stage of the project.

2. **50-Character Overlap for Context Preservation**:
   - Overlap prevents sentences, key phrases, and semantic meaning from being abruptly cut across chunk boundaries.
   - Ensures boundary context is preserved so that embeddings and retrieval steps do not miss split entities or concepts.

3. **Pure Python String Slicing**:
   - Keeps the core pipeline lightweight, transparent, and easy to test and debug without heavy external abstractions.

---

## Reference Implementation

All chunking logic in the codebase should follow this reference pattern:

```python
from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into fixed-size chunks with overlap using Python string slicing.
    
    Standard:
        - chunk_size: 500 characters
        - overlap: 50 characters
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")

    chunks: List[str] = []
    step = chunk_size - overlap
    
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
            
    return chunks
```

---

## Guidelines for Development

- **Do not import** `langchain.text_splitter` or similar modules.
- **Maintain parameters**: Keep defaults at `500` characters and `50` characters overlap.
- **Preserve metadata**: When attaching metadata (e.g., chunk index, document name), ensure each slice corresponds directly to the source text offsets.
