"""
embed_and_store.py
------------------
This script performs the embedding generation and vector storage stage of a RAG pipeline.

What it does:
1. Reuses `load_pdf_text` and `chunk_text` from `load_and_chunk.py` to ingest and slice the PDF.
2. Loads the Google Gemini API key from `.env` using `python-dotenv`.
3. Connects to Google GenAI using the modern `google-genai` SDK (`from google import genai`).
4. Generates high-dimensional vector embeddings for each chunk using the `gemini-embedding-001` model.
5. Persists chunks, vectors, and metadata (`source` filename, `chunk_index`) into a local ChromaDB
   database (`./chroma_db`) under the collection name `rag_documents`.
6. Prints progress every 10 chunks and displays a final confirmation with the total chunk count.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
import chromadb

# Step 1: Import functions from load_and_chunk.py to reuse document processing logic
from load_and_chunk import load_pdf_text, chunk_text


def get_genai_client() -> genai.Client:
    """
    Step 2: Load environment variables from .env and initialize the Google GenAI client.

    Why this matters:
    - Never hardcode API keys directly into source code.
    - We use python-dotenv to read variables from .env.
    - We use the modern 'google-genai' SDK (not deprecated google-generativeai).
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add it to your .env file."
        )

    print("[Step 2] Initializing Google GenAI client with Gemini API key...")
    # Initialize the modern Gemini API client
    return genai.Client(api_key=api_key)


def get_persistent_collection(db_path: Path, collection_name: str = "rag_documents") -> chromadb.Collection:
    """
    Step 4a: Initialize a persistent ChromaDB client and retrieve or create the collection.

    Why PersistentClient?
    - An in-memory client loses all data when the Python process finishes.
    - A PersistentClient saves embeddings, documents, and indices to disk (SQLite + binary files).
    - This ensures our data survives across script runs, restarts, and web server lifecycles.
    """
    print(f"[Step 4] Connecting to persistent ChromaDB at: {db_path}")
    # PersistentClient persists data locally to the given directory path
    chroma_client = chromadb.PersistentClient(path=str(db_path))

    # get_or_create_collection avoids throwing an error if the collection already exists
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "PDF chunks embedded with gemini-embedding-001"}
    )
    return collection


def generate_embedding(client: genai.Client, text: str) -> List[float]:
    """
    Step 3: Generate an embedding vector for a single text chunk.

    What is an embedding?
    - An embedding is a list of floating-point numbers (a vector) representing the
      semantic meaning of the text.
    - Texts with similar meanings end up closer together in vector space (using cosine distance).
    - We use 'gemini-embedding-001', which generates high-quality 3072-dimensional vectors.
    """
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    # The SDK returns EmbedContentResponse with .embeddings containing ContentEmbedding objects
    return response.embeddings[0].values


def embed_and_store_chunks(
    chunks: List[str],
    source_filename: str,
    genai_client: genai.Client,
    collection: chromadb.Collection
) -> None:
    """
    Step 3, 4 & 5: Generate embeddings and store each chunk in ChromaDB with metadata.

    Why metadata is crucial:
    - In RAG applications, the model must cite its sources (e.g. 'From page X of rag_research_paper.pdf').
    - Storing `source` and `chunk_index` metadata enables tracking the exact origin of retrieved chunks.
    - We use collection.upsert() so that rerunning the script safely updates existing chunks without error.
    """
    total_chunks = len(chunks)
    print(f"\n[Step 3 & 4] Generating embeddings and storing {total_chunks} chunks in ChromaDB...")
    print(f"             Target Collection: '{collection.name}'")
    print(f"             Embedding Model:   'gemini-embedding-001'")

    for i, chunk in enumerate(chunks, start=1):
        chunk_idx = i - 1  # 0-based index for metadata

        # Generate embedding vector for the current chunk
        embedding = generate_embedding(genai_client, chunk)

        # Build unique chunk ID and metadata dictionary
        chunk_id = f"{source_filename}_chunk_{chunk_idx}"
        metadata: Dict[str, Any] = {
            "source": source_filename,
            "chunk_index": chunk_idx,
        }

        # Upsert into ChromaDB (adds new or updates if chunk_id already exists)
        collection.upsert(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[metadata]
        )

        # Step 5: Print progress every 10 chunks or on the final chunk
        if i % 10 == 0:
            print(f"  [Progress] Processed and stored {i}/{total_chunks} chunks...")

    # Final confirmation
    total_stored = collection.count()
    print("\n" + "=" * 60)
    print(f"[Step 5] Confirmation: Successfully stored all {total_chunks} chunks!")
    print(f"         Total records now in '{collection.name}': {total_stored}")
    print("=" * 60)


def main() -> None:
    # Define file paths
    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / "documents" / "rag_research_paper.pdf"
    chroma_db_dir = project_root / "chroma_db"

    # Step 1: Ingest PDF and create chunks using load_and_chunk functions
    print("[Step 1] Loading document and chunking text...")
    full_text = load_pdf_text(pdf_path)
    chunks = chunk_text(full_text, chunk_size=500, overlap=50)
    print(f"         Created {len(chunks)} chunks according to project standards.")

    # Step 2: Initialize modern Google GenAI Client
    genai_client = get_genai_client()

    # Step 4a: Get persistent ChromaDB collection
    collection = get_persistent_collection(chroma_db_dir, collection_name="rag_documents")

    # Step 3, 4 & 5: Embed, store with metadata, and report progress
    embed_and_store_chunks(
        chunks=chunks,
        source_filename=pdf_path.name,
        genai_client=genai_client,
        collection=collection
    )


if __name__ == "__main__":
    main()
