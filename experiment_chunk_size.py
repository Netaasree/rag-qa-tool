"""
experiment_chunk_size.py
-------------------------
Experimentation pipeline to test alternative chunking parameters.

What it does:
1. Loads the source PDF ('documents/rag_research_paper.pdf') using load_pdf_text.
2. Chunks the document using chunk_size=800 and overlap=100 (compared to the baseline 500/50).
3. Connects to persistent ChromaDB ('./chroma_db').
4. Stores the generated embeddings and chunks into a new, separate collection:
   'rag_documents_v2' (leaving 'rag_documents' intact for direct comparison).
5. Reports chunking statistics, embedding progress, and final storage counts.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import errors
import chromadb

# Re-use document loading and chunking functions from load_and_chunk.py
from load_and_chunk import load_pdf_text, chunk_text


def get_genai_client() -> genai.Client:
    """
    Initializes the Google GenAI client from the environment GEMINI_API_KEY.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")
    return genai.Client(api_key=api_key)


def get_persistent_collection(
    db_path: Path,
    collection_name: str = "rag_documents_v2"
) -> chromadb.Collection:
    """
    Retrieves or creates the experimental persistent ChromaDB collection.
    """
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": "PDF chunks with chunk_size=800 and overlap=100",
            "chunk_size": 800,
            "overlap": 100,
            "embedding_model": "gemini-embedding-001"
        }
    )
    return collection


def generate_embedding(
    client: genai.Client,
    text: str,
    max_retries: int = 5
) -> List[float]:
    """
    Generates embedding using gemini-embedding-001 with transient error retry.
    """
    backoff_delays = [2, 4, 7, 10, 15]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
            )
            return response.embeddings[0].values
        except errors.APIError as err:
            if attempt < max_retries and getattr(err, "code", None) in (503, 429):
                wait_time = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                print(f"  [Notice] Gemini API busy (503/429). Retrying in {wait_time}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise


def embed_and_store_experimental_chunks(
    chunks: List[str],
    source_filename: str,
    genai_client: genai.Client,
    collection: chromadb.Collection
) -> None:
    """
    Embeds and stores the 800-char chunks into the experimental collection.
    """
    total_chunks = len(chunks)
    print(f"\n[Storage] Storing {total_chunks} chunks into experimental collection '{collection.name}'...")
    print("          Parameters: chunk_size=800, overlap=100")
    print("          Embedding Model: gemini-embedding-001")

    for i, chunk in enumerate(chunks, start=1):
        chunk_idx = i - 1
        embedding = generate_embedding(genai_client, chunk)
        chunk_id = f"{source_filename}_v2_chunk_{chunk_idx}"

        metadata: Dict[str, Any] = {
            "source": source_filename,
            "chunk_index": chunk_idx,
            "chunk_size": 800,
            "overlap": 100,
            "char_count": len(chunk)
        }

        collection.upsert(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[metadata]
        )

        if i % 10 == 0 or i == total_chunks:
            print(f"  [Progress] Processed and stored {i}/{total_chunks} chunks...")

    total_stored = collection.count()
    print("\n" + "=" * 65)
    print(f"[Confirmation] Successfully populated '{collection.name}'!")
    print(f"               Total chunks stored: {total_stored}")
    print("=" * 65 + "\n")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / "documents" / "rag_research_paper.pdf"
    chroma_db_dir = project_root / "chroma_db"

    print("=" * 65)
    print(" RAG QA Experiment: Evaluating Chunk Size 800 (Overlap 100)")
    print(" Target Collection: 'rag_documents_v2'")
    print("=" * 65)

    # Step 1: Extract text from PDF
    print("\n[Step 1] Loading document text...")
    full_text = load_pdf_text(pdf_path)

    # Step 2: Chunk with experimental parameters (800 / 100)
    print("\n[Step 2] Chunking with experimental parameters (chunk_size=800, overlap=100)...")
    chunks = chunk_text(full_text, chunk_size=800, overlap=100)
    print(f"         Created {len(chunks)} chunks (vs 94 chunks in original 500/50 setup).")

    # Step 3: Initialize GenAI Client
    print("\n[Step 3] Initializing Google GenAI Client...")
    genai_client = get_genai_client()

    # Step 4: Connect to separate collection 'rag_documents_v2'
    print("\n[Step 4] Accessing ChromaDB collection 'rag_documents_v2'...")
    collection = get_persistent_collection(chroma_db_dir, collection_name="rag_documents_v2")

    # Step 5: Embed and store chunks
    embed_and_store_experimental_chunks(
        chunks=chunks,
        source_filename=pdf_path.name,
        genai_client=genai_client,
        collection=collection
    )


if __name__ == "__main__":
    main()
