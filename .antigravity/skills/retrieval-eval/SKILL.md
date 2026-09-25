---
name: retrieval-eval
description: Standard for evaluating and benchmarking RAG retrieval quality using (question, expected_keyword) hit rates across ChromaDB collections.
---

# Retrieval Evaluation Standard & Benchmark Methodology

## Overview

In Retrieval-Augmented Generation (RAG), answer generation quality is strictly bounded by retrieval quality: if the retrieved context misses the essential facts, the generator model cannot produce a grounded answer. 

This skill defines the project standard for evaluating and comparing retrieval quality across different collections (e.g., comparing baseline chunking `rag_documents` [500 chars / 50 overlap] against experimental chunking `rag_documents_v2` [800 chars / 100 overlap]) using an objective, reproducible metric rather than manual spot-checking.

---

## Core Standard: Hit Rate Evaluation

### 1. The Evaluation Set Format

An evaluation set is defined as a list of `(question, expected_keyword)` pairs:

```python
EVAL_SET = [
    ("What are the two primary components of a RAG model?", "parametric"),
    ("Which pre-trained sequence-to-sequence model is used as the generator in RAG?", "BART"),
    ("How is the retriever in RAG initialized?", "DPR"),
    # ...
]
```

### 2. Definition of `expected_keyword`

- **Distinctive & Unambiguous**: An `expected_keyword` is a distinctive word or phrase that must appear somewhere within the retrieved chunks for that question to be counted as a **"hit"**.
- **Ground Truth Grounding**: The keyword must be an indispensable factual anchor required to answer the question accurately based on the source document.
- **Avoid Ambiguous / Ubiquitous Words**: Do not use generic words that appear across almost all chunks (e.g., `"model"`, `"paper"`, `"data"`, `"system"`). Choose specific technical terms, named entities, metrics, or model architectures (e.g., `"Dense Passage Retrieval"`, `"BART"`, `"Exact Match"`, `"parametric"`).

---

## Retrieval Evaluation Workflow

For any given ChromaDB collection and evaluation set, the evaluation script executes the following deterministic procedure:

```
[For Each (question, expected_keyword) in EVAL_SET]
  │
  ├── 1. Embed question using `gemini-embedding-001`
  │
  ├── 2. Query target ChromaDB collection for top-k chunks (default k=8)
  │
  ├── 3. Keyword Check: Does `expected_keyword.lower()` exist in ANY retrieved chunk text?
  │       ├── YES ──> Record as HIT  (PASS)
  │       └── NO  ──> Record as MISS (FAIL)
  │
[Calculate Metric]
  └── Hit Rate (%) = (Total Hits / Total Questions) * 100
```

---

## The Hit Rate Metric

$$\text{Hit Rate (\%)} = \left(\frac{\text{Hits}}{\text{Total Questions}}\right) \times 100$$

### Why Hit Rate?
- **Quantitative Comparison**: Provides an exact numerical score (e.g., `85.7%` vs `71.4%`) to objectively measure whether a change in chunk size, overlap, embedding model, or retrieval `k` improves passage recall.
- **Fast & Cost-Effective**: Keyword checking runs entirely in-memory on the retrieved chunk strings, avoiding expensive LLM evaluation calls (LLM-as-a-judge) while maintaining strong alignment with downstream answerability.
- **Repeatable & Deterministic**: Running the evaluation multiple times yields reproducible metrics without stochastic variance from generation prompts.

---

## Reference Implementation

The following reference script illustrates how to run retrieval evaluation against any target collection using the project's standard components (`chromadb.PersistentClient`, `gemini-embedding-001`):

```python
"""
Reference implementation for RAG retrieval hit rate evaluation.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from google import genai
import chromadb


def evaluate_retrieval(
    collection_name: str,
    eval_set: List[Tuple[str, str]],
    chroma_db_dir: Path,
    k: int = 8
) -> Dict[str, Any]:
    """
    Evaluates retrieval quality against a target collection using a keyword hit rate benchmark.

    Parameters:
        collection_name (str): Target ChromaDB collection name (e.g. 'rag_documents', 'rag_documents_v2').
        eval_set (List[Tuple[str, str]]): List of (question, expected_keyword) tuples.
        chroma_db_dir (Path): Path to the persistent ChromaDB directory.
        k (int): Number of top chunks to retrieve (project standard is 8).

    Returns:
        Dict[str, Any]: Evaluation summary including hits, total, hit_rate, and per-question details.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment.")

    genai_client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(chroma_db_dir))
    collection = chroma_client.get_collection(name=collection_name)

    total_questions = len(eval_set)
    hits = 0
    results_detail = []

    print(f"\n=======================================================")
    print(f" Evaluating Collection: '{collection_name}' (k={k})")
    print(f" Total Evaluation Questions: {total_questions}")
    print(f"=======================================================\n")

    for idx, (question, expected_keyword) in enumerate(eval_set, start=1):
        # 1. Embed question (ensuring vector symmetry)
        embed_resp = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )
        query_vector = embed_resp.embeddings[0].values

        # 2. Retrieve top-k chunks from ChromaDB
        query_result = collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        retrieved_docs = query_result.get("documents", [[]])[0]
        retrieved_metas = query_result.get("metadatas", [[]])[0]

        # 3. Check if expected keyword appears in any retrieved chunk
        keyword_lower = expected_keyword.lower()
        matched_chunk_idx = None

        for chunk_pos, doc_text in enumerate(retrieved_docs):
            if keyword_lower in doc_text.lower():
                matched_chunk_idx = chunk_pos + 1  # 1-based rank
                break

        is_hit = matched_chunk_idx is not None
        if is_hit:
            hits += 1
            status = f"HIT  (Rank #{matched_chunk_idx})"
        else:
            status = "MISS"

        results_detail.append({
            "index": idx,
            "question": question,
            "expected_keyword": expected_keyword,
            "is_hit": is_hit,
            "hit_rank": matched_chunk_idx
        })

        print(f"[{idx}/{total_questions}] {status} | Q: {question[:50]}... | Key: '{expected_keyword}'")

    hit_rate = (hits / total_questions) * 100.0 if total_questions > 0 else 0.0

    print(f"\n-------------------------------------------------------")
    print(f" Results for '{collection_name}':")
    print(f" Hits: {hits}/{total_questions}")
    print(f" Hit Rate: {hit_rate:.1f}%")
    print(f"-------------------------------------------------------\n")

    return {
        "collection_name": collection_name,
        "hits": hits,
        "total_questions": total_questions,
        "hit_rate": hit_rate,
        "details": results_detail
    }
```

---

## Comparing Collections (e.g., `rag_documents` vs `rag_documents_v2`)

To compare chunking configurations or collection variations:

1. **Keep the Evaluation Set Constant**: Run the exact same `EVAL_SET` across all candidate collections.
2. **Keep Retrieval `k` Constant**: Use the project standard `k=8` unless explicitly tuning the retrieval budget.
3. **Compare Hit Rates Side-by-Side**:
   - Compare overall percentage scores.
   - Inspect rank distributions (e.g., does collection A place the relevant keyword at Rank #1 while collection B places it at Rank #7?).
   - Identify questions that fail on both collections to detect vocabulary gaps or missing coverage in the source document.

---

## Guidelines for Authors & Evaluators

- **Strict Vector Symmetry**: Always embed questions using `gemini-embedding-001` to match the vector space of the stored chunks.
- **Read-Only Operation**: Retrieval evaluation scripts must never modify or re-index the collection during an evaluation run.
- **Case-Insensitive Matching**: Match `expected_keyword.lower() in chunk_text.lower()` to avoid false negatives caused by capitalization differences.
- **Keyword Specificity**: Choose keywords that are specific enough that finding them indicates true relevance to the query, rather than coincidental mention.
