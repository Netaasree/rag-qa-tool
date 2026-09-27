"""
run_eval.py
-----------
Retrieval evaluation script for benchmarking and comparing RAG retrieval quality
across multiple ChromaDB collections using (question, expected_keyword) hit rates.

Implements the standard defined in .antigravity/skills/retrieval-eval/SKILL.md:
1. Imports EVAL_SET from eval_set.py.
2. Runs evaluate_retrieval() against:
   - "rag_documents" (baseline: chunk_size 500 / overlap 50)
   - "rag_documents_v2" (experimental: chunk_size 800 / overlap 100)
3. Uses n_results=8 (k=8) for retrieval, consistent with ask.py.
4. Prints per-question results (✅ HIT / ❌ MISS) for each collection.
5. Prints a side-by-side summary comparing hit rates and per-question ranks.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import errors
import chromadb

# Ensure Windows terminal handles UTF-8 emojis (e.g. ✅, ❌) without charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Step 1: Import the ground-truth evaluation set
from eval_set import EVAL_SET


def get_genai_client() -> genai.Client:
    """
    Initializes the Google GenAI client using the GEMINI_API_KEY environment variable.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please ensure it is defined in your .env file."
        )
    return genai.Client(api_key=api_key)


def embed_question(
    client: genai.Client,
    question: str,
    max_retries: int = 5
) -> List[float]:
    """
    Generates an embedding vector for a question using gemini-embedding-001.
    Includes exponential backoff retry for transient API spikes (503 / 429).
    """
    backoff_delays = [2, 4, 7, 10, 15]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=question,
            )
            return response.embeddings[0].values
        except errors.APIError as err:
            if attempt < max_retries and getattr(err, "code", None) in (503, 429):
                wait_time = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                print(f"  [Notice] Gemini API busy ({err.code}). Retrying in {wait_time}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise


def evaluate_retrieval(
    collection_name: str,
    eval_set: List[Tuple[str, str]],
    chroma_db_dir: Path,
    k: int = 8,
    genai_client: Optional[genai.Client] = None,
    query_embeddings: Optional[Dict[str, List[float]]] = None,
    chunk_config_desc: str = ""
) -> Dict[str, Any]:
    """
    Evaluates retrieval quality against a target ChromaDB collection using a keyword hit rate benchmark.

    Parameters:
        collection_name (str): Target ChromaDB collection name (e.g., 'rag_documents', 'rag_documents_v2').
        eval_set (List[Tuple[str, str]]): List of (question, expected_keyword) tuples.
        chroma_db_dir (Path): Path to the persistent ChromaDB directory.
        k (int): Number of top chunks to retrieve (project standard is 8).
        genai_client (Optional[genai.Client]): Initialized GenAI client instance.
        query_embeddings (Optional[Dict[str, List[float]]]): Optional precomputed cache of question embeddings.
        chunk_config_desc (str): Descriptive label for chunking parameters (e.g. '500/50').

    Returns:
        Dict[str, Any]: Evaluation summary including hits, total, hit_rate, and per-question details.
    """
    if genai_client is None:
        genai_client = get_genai_client()

    chroma_client = chromadb.PersistentClient(path=str(chroma_db_dir))
    collection = chroma_client.get_collection(name=collection_name)

    total_questions = len(eval_set)
    hits = 0
    results_detail: List[Dict[str, Any]] = []

    config_str = f" [{chunk_config_desc}]" if chunk_config_desc else ""
    print(f"\n=======================================================")
    print(f" Evaluating Collection: '{collection_name}'{config_str}")
    print(f" Top-k Chunks (n_results): {k}")
    print(f" Total Evaluation Questions: {total_questions}")
    print(f"=======================================================\n")

    for idx, (question, expected_keyword) in enumerate(eval_set, start=1):
        # 1. Obtain query embedding (from cache if available, else embed with gemini-embedding-001)
        if query_embeddings and question in query_embeddings:
            query_vector = query_embeddings[question]
        else:
            query_vector = embed_question(genai_client, question)
            if query_embeddings is not None:
                query_embeddings[question] = query_vector

        # 2. Query target ChromaDB collection for top-k chunks
        query_result = collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        retrieved_docs = query_result.get("documents", [[]])[0]
        retrieved_metas = query_result.get("metadatas", [[]])[0]

        # 3. Keyword Check: Does expected_keyword.lower() exist in any retrieved chunk?
        keyword_lower = expected_keyword.lower()
        matched_chunk_idx: Optional[int] = None

        for chunk_pos, doc_text in enumerate(retrieved_docs):
            if keyword_lower in doc_text.lower():
                matched_chunk_idx = chunk_pos + 1  # 1-based rank
                break

        is_hit = matched_chunk_idx is not None
        if is_hit:
            hits += 1
            status_tag = f"✅ HIT  (Rank #{matched_chunk_idx})"
        else:
            status_tag = "❌ MISS"

        results_detail.append({
            "index": idx,
            "question": question,
            "expected_keyword": expected_keyword,
            "is_hit": is_hit,
            "hit_rank": matched_chunk_idx,
            "status_tag": status_tag
        })

        # 4. Print per-question result
        short_q = question[:50] + ("..." if len(question) > 50 else "")
        print(f"[{idx}/{total_questions}] {status_tag:<20} | Q: {short_q:<53} | Key: '{expected_keyword}'")

    hit_rate = (hits / total_questions) * 100.0 if total_questions > 0 else 0.0

    print(f"\n-------------------------------------------------------")
    print(f" Results for '{collection_name}':")
    print(f" Hits: {hits}/{total_questions}")
    print(f" Hit Rate: {hit_rate:.1f}%")
    print(f"-------------------------------------------------------\n")

    return {
        "collection_name": collection_name,
        "chunk_config": chunk_config_desc,
        "k": k,
        "hits": hits,
        "total_questions": total_questions,
        "hit_rate": hit_rate,
        "details": results_detail
    }


def print_comparison_summary(
    results_a: Dict[str, Any],
    results_b: Dict[str, Any]
) -> None:
    """
    Step 5: Prints a side-by-side summary table comparing hit rate % and per-question ranks.
    """
    col_a_name = f"{results_a['collection_name']} ({results_a.get('chunk_config', '')})"
    col_b_name = f"{results_b['collection_name']} ({results_b.get('chunk_config', '')})"

    hits_a_str = f"{results_a['hits']}/{results_a['total_questions']}"
    hits_b_str = f"{results_b['hits']}/{results_b['total_questions']}"
    misses_a = results_a['total_questions'] - results_a['hits']
    misses_b = results_b['total_questions'] - results_b['hits']
    rate_a_str = f"{results_a['hit_rate']:.1f}%"
    rate_b_str = f"{results_b['hit_rate']:.1f}%"

    print("\n" + "=" * 92)
    print("                 RETRIEVAL EVALUATION: SIDE-BY-SIDE COMPARISON SUMMARY")
    print("=" * 92)
    print(f"{'Metric / Attribute':<30} | {col_a_name:<28} | {col_b_name:<28}")
    print("-" * 92)
    print(f"{'Top-k Retrieved (n_results)':<30} | {str(results_a['k']):<28} | {str(results_b['k']):<28}")
    print(f"{'Total Questions':<30} | {str(results_a['total_questions']):<28} | {str(results_b['total_questions']):<28}")
    print(f"{'Successful Hits':<30} | {hits_a_str:<28} | {hits_b_str:<28}")
    print(f"{'Misses':<30} | {str(misses_a):<28} | {str(misses_b):<28}")
    print(f"{'Hit Rate (%)':<30} | {rate_a_str:<28} | {rate_b_str:<28}")
    print("-" * 92)

    delta = results_b["hit_rate"] - results_a["hit_rate"]
    sign = "+" if delta > 0 else ""
    delta_str = f"{sign}{delta:.1f}%"
    print(f"{'Hit Rate Delta (v2 - baseline)':<30} | {delta_str:<28} |")
    print("=" * 92)

    # Per-Question breakdown table
    print("\n" + "-" * 92)
    print(f"{'#':<3} {'Expected Keyword':<22} | {col_a_name:<30} | {col_b_name:<30}")
    print("-" * 92)

    details_a = results_a["details"]
    details_b = results_b["details"]

    for item_a, item_b in zip(details_a, details_b):
        idx = item_a["index"]
        key = item_a["expected_keyword"]
        tag_a = item_a["status_tag"]
        tag_b = item_b["status_tag"]
        print(f"{idx:<3} {key:<22} | {tag_a:<30} | {tag_b:<30}")

    print("-" * 92 + "\n")


def main() -> None:
    """
    Main evaluation pipeline orchestrator.
    """
    project_root = Path(__file__).resolve().parent
    chroma_db_dir = project_root / "chroma_db"

    print("=" * 70)
    print(" RAG Retrieval Quality Evaluation & Benchmark")
    print(f" Benchmark Set: {len(EVAL_SET)} questions from eval_set.py")
    print(" Embedding Model: 'gemini-embedding-001'")
    print(" Retrieval Budget: n_results=8")
    print("=" * 70)

    # Initialize client once
    genai_client = get_genai_client()

    # Pre-embed questions once into a shared cache to avoid redundant API calls
    print("\nPre-embedding evaluation questions with 'gemini-embedding-001'...")
    query_embeddings: Dict[str, List[float]] = {}
    for i, (q, _) in enumerate(EVAL_SET, start=1):
        query_embeddings[q] = embed_question(genai_client, q)
        print(f"  Embedded question {i}/{len(EVAL_SET)}")
    print("Embeddings ready.\n")

    # Evaluate Baseline Collection: rag_documents (500 chars / 50 overlap)
    baseline_results = evaluate_retrieval(
        collection_name="rag_documents",
        eval_set=EVAL_SET,
        chroma_db_dir=chroma_db_dir,
        k=8,
        genai_client=genai_client,
        query_embeddings=query_embeddings,
        chunk_config_desc="500/50"
    )

    # Evaluate Experimental Collection: rag_documents_v2 (800 chars / 100 overlap)
    v2_results = evaluate_retrieval(
        collection_name="rag_documents_v2",
        eval_set=EVAL_SET,
        chroma_db_dir=chroma_db_dir,
        k=8,
        genai_client=genai_client,
        query_embeddings=query_embeddings,
        chunk_config_desc="800/100"
    )

    # Step 5: Side-by-side comparative summary
    print_comparison_summary(baseline_results, v2_results)


if __name__ == "__main__":
    main()
