"""
ask.py
------
This script implements the retrieval and generation stage of the RAG pipeline.

What it does:
1. Accepts a user question as a command-line argument.
2. Initializes the modern Google GenAI client (`from google import genai`) using GEMINI_API_KEY.
3. Connects to the local persistent ChromaDB database (`./chroma_db`) and loads the `rag_documents` collection.
4. Generates an embedding vector for the question using `gemini-embedding-001` (guaranteeing vector symmetry with storage).
5. Queries ChromaDB for the top 8 most semantically similar document chunks.
6. Displays the retrieved chunks (chunk_index, source, and preview) so the user can inspect retrieval quality.
7. Builds a strictly grounded prompt with the chunks as context and the mandatory anti-hallucination instruction:
   "I don't know based on the provided documents" if the answer cannot be determined.
8. Generates the final answer using `gemini-3.8-flash` via `client.models.generate_content()`.
9. Prints the final grounded answer clearly.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import errors
import chromadb


def get_genai_client() -> genai.Client:
    """
    Step 1: Load environment variables and initialize the Google GenAI client.

    Why this matters:
    - Never hardcode API keys. We use python-dotenv to securely read from .env.
    - We strictly use the modern 'google-genai' SDK (`from google import genai`).
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please ensure it is defined in your .env file."
        )

    return genai.Client(api_key=api_key)


def get_persistent_collection(
    db_path: Path,
    collection_name: str = "rag_documents"
) -> chromadb.Collection:
    """
    Step 2: Connect to the existing persistent ChromaDB client and retrieve the collection.

    Why PersistentClient?
    - An in-memory client resets between executions.
    - PersistentClient connects to our SQLite/index storage in `chroma_db/`,
      allowing us to query previously indexed vectors without re-embedding the source documents.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at '{db_path}'. "
            "Please run 'embed_and_store.py' first to create and populate the database."
        )

    chroma_client = chromadb.PersistentClient(path=str(db_path))
    
    # Verify the collection exists
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if collection_name not in existing_collections:
        raise ValueError(
            f"Collection '{collection_name}' not found in ChromaDB at '{db_path}'. "
            "Please run 'embed_and_store.py' first to embed and store chunks."
        )

    return chroma_client.get_collection(name=collection_name)


def embed_question(client: genai.Client, question: str, max_retries: int = 5) -> List[float]:
    """
    Step 3: Embed the user's question using gemini-embedding-001.

    Why vector symmetry is essential:
    - Vector search relies on cosine similarity / Euclidean distance in a shared vector space.
    - We MUST use the exact same embedding model (`gemini-embedding-001`) that was used
      during ingestion in `embed_and_store.py`.
    - Mixing different embedding models leads to nonsensical distance metrics.
    - Includes automatic retry handling for temporary API spikes (503 / 429).
    """
    backoff_delays = [2, 4, 7, 10, 15]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=question,
            )
            # Extract the vector values from the first embedding object
            return response.embeddings[0].values
        except errors.APIError as err:
            if attempt < max_retries and getattr(err, "code", None) in (503, 429):
                wait_time = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                print(f"  [Notice] Gemini API busy (503/429). Retrying in {wait_time}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise


def retrieve_relevant_chunks(
    collection: chromadb.Collection,
    query_embedding: List[float],
    n_results: int = 8
) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
    """
    Step 4: Query the persistent ChromaDB collection for the top 8 most similar chunks.

    Parameters:
        collection (chromadb.Collection): Target ChromaDB collection.
        query_embedding (List[float]): The 3072-dimensional embedding vector of the question.
        n_results (int): Number of chunks to retrieve (project standard is 8).

    Returns:
        Tuple containing:
            - documents (List[str]): Retrieved chunk texts.
            - metadatas (List[Dict[str, Any]]): Metadata for each chunk (e.g., chunk_index, source).
            - distances (List[float]): Similarity distances.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    return documents, metadatas, distances


def print_retrieved_chunks(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    preview_length: int = 120
) -> None:
    """
    Step 5: Display inspection details for retrieved chunks.

    Why inspect chunks?
    - Transparency: In RAG, inspecting retrieved chunks helps verify whether the retrieval
      layer actually located the relevant passage before passing it to the generator model.
    """
    print("\n" + "=" * 70)
    print(f"[Step 5] Retrieved Chunks ({len(documents)} most relevant passages):")
    print("=" * 70)

    for rank, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        chunk_idx = meta.get("chunk_index", "Unknown")
        source = meta.get("source", "Unknown")
        
        # Clean preview text for single-line display
        cleaned_doc = " ".join(doc.split())
        preview = cleaned_doc[:preview_length] + ("..." if len(cleaned_doc) > preview_length else "")

        print(f"  Rank #{rank} | Chunk Index: {chunk_idx} | Source: {source}")
        print(f"    Preview: \"{preview}\"\n")


def construct_grounded_prompt(question: str, context_chunks: List[str]) -> str:
    """
    Step 6a: Build a prompt that grounds the model strictly in retrieved context.

    Why this design:
    - RAG solves hallucination only if the prompt strictly enforces boundary conditions.
    - We provide the retrieved chunks joined together as Context.
    - We instruct the model to answer ONLY from that Context.
    - We explicitly define a fallback phrase:
      "I don't know based on the provided documents"
      if the context does not contain sufficient information to answer the question.
    """
    joined_context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are an accurate, reliable question-answering assistant. Your answers must be grounded strictly in the provided context documents.

Context:
{joined_context}

Question:
{question}

Instructions:
1. Answer the question using ONLY the facts directly mentioned in the Context above.
2. If the answer cannot be determined strictly from the provided Context, you MUST respond with the exact phrase:
   "I don't know based on the provided documents"
3. Do not speculate, extrapolate, or use any outside knowledge not present in the Context.
4. Keep the answer clear, concise, and factual.
"""
    return prompt


def generate_grounded_answer(client: genai.Client, prompt: str, max_retries: int = 5) -> str:
    """
    Step 6b: Generate the answer using gemini-3.8-flash.

    Why gemini-3.8-flash?
    - Fast response times and efficient token usage.
    - Excellent instruction following and contextual synthesis capabilities.
    - Modern SDK call: `client.models.generate_content()`.
    - Includes automatic retry handling for temporary API spikes (503 / 429).
    """
    backoff_delays = [2, 4, 7, 10, 15]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )
            return response.text.strip() if response.text else "I don't know based on the provided documents"
        except errors.APIError as err:
            if attempt < max_retries and getattr(err, "code", None) in (503, 429):
                wait_time = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                print(f"  [Notice] Gemini API busy (503/429). Retrying in {wait_time}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise


def ask(question: str, collection_name: str = "rag_documents") -> str:
    """
    Full pipeline function: Question -> Embedding -> Retrieval -> Grounding -> Generation.
    """
    project_root = Path(__file__).resolve().parent
    chroma_db_dir = project_root / "chroma_db"

    print(f"\n[Query] \"{question}\"")
    print(f"[Target Collection] '{collection_name}'")

    # Step 1: Initialize Google GenAI client
    print("[Step 1] Initializing Google GenAI Client...")
    genai_client = get_genai_client()

    # Step 2: Connect to Persistent ChromaDB
    print(f"[Step 2] Connecting to persistent ChromaDB collection '{collection_name}'...")
    collection = get_persistent_collection(chroma_db_dir, collection_name=collection_name)

    # Step 3: Embed the question
    print("[Step 3] Embedding question with 'gemini-embedding-001'...")
    query_vector = embed_question(genai_client, question)

    # Step 4: Retrieve top 8 chunks
    print(f"[Step 4] Querying ChromaDB collection '{collection_name}' for top 8 most similar chunks...")
    documents, metadatas, _ = retrieve_relevant_chunks(
        collection=collection,
        query_embedding=query_vector,
        n_results=8
    )

    if not documents:
        print("[Warning] No chunks were retrieved from ChromaDB.")
        return "I don't know based on the provided documents"

    # Step 5: Print retrieved chunk previews
    print_retrieved_chunks(documents, metadatas)

    # Step 6: Construct grounded prompt & generate response
    print("[Step 6] Constructing grounded prompt and querying 'gemini-3.8-flash'...")
    prompt = construct_grounded_prompt(question, documents)
    answer = generate_grounded_answer(genai_client, prompt)

    # Step 7: Print final answer clearly
    print("=" * 70)
    print("[Step 7] Answer:")
    print("=" * 70)
    print(answer)
    print("=" * 70 + "\n")

    return answer


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the question and optional collection.
    """
    parser = argparse.ArgumentParser(
        description="Ask a question against the indexed RAG documents using Google Gemini & ChromaDB."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="The question to ask (wrap in quotes if it contains spaces)."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="rag_documents",
        help="ChromaDB collection to query (default: 'rag_documents')."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.question:
        # If no argument is passed, provide a helpful prompt or instructions
        print("Usage: python ask.py \"<your question here>\" [--collection COLLECTION_NAME]")
        print("Example: python ask.py \"What is Retrieval-Augmented Generation?\"")
        print("         python ask.py \"What is Retrieval-Augmented Generation?\" --collection rag_documents_v2")
        sys.exit(1)

    ask(args.question, collection_name=args.collection)


if __name__ == "__main__":
    main()
