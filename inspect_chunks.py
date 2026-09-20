"""
inspect_chunks.py
-----------------
Small one-off utility script to inspect specific chunks stored in the persistent
ChromaDB 'rag_documents' collection.

Retrieves and prints the full text and metadata (source filename and chunk_index)
for chunks 16, 17, and 18.
"""

from pathlib import Path
from typing import List, Dict, Any
import chromadb


def inspect_specific_chunks(
    target_indices: List[int] = [16, 17, 18],
    collection_name: str = "rag_documents"
) -> None:
    """
    Connects to persistent ChromaDB and prints the full text and source of target chunks.
    """
    project_root = Path(__file__).resolve().parent
    db_path = project_root / "chroma_db"

    if not db_path.exists():
        raise FileNotFoundError(f"ChromaDB directory not found at: {db_path}")

    print(f"Connecting to persistent ChromaDB at: {db_path}")
    chroma_client = chromadb.PersistentClient(path=str(db_path))

    # Retrieve existing collection
    collection = chroma_client.get_collection(name=collection_name)
    total_chunks = collection.count()
    print(f"Connected to collection '{collection_name}' (Total stored chunks: {total_chunks})")
    print(f"Searching for chunk indices: {target_indices}\n")

    # Query by chunk_index metadata using ChromaDB's $in operator
    results = collection.get(
        where={"chunk_index": {"$in": target_indices}},
        include=["documents", "metadatas"]
    )

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    if not ids:
        print("No chunks found matching the specified indices.")
        return

    # Combine and sort by chunk_index so they appear in sequence
    chunk_records = []
    for chunk_id, doc, meta in zip(ids, documents, metadatas):
        idx = meta.get("chunk_index", -1) if meta else -1
        source = meta.get("source", "Unknown") if meta else "Unknown"
        chunk_records.append({
            "id": chunk_id,
            "chunk_index": idx,
            "source": source,
            "document": doc
        })

    # Sort sequentially by chunk_index
    chunk_records.sort(key=lambda x: x["chunk_index"])

    # Print full text and details for each chunk
    for item in chunk_records:
        print("=" * 80)
        print(f"CHUNK INDEX: {item['chunk_index']} | SOURCE: {item['source']} | ID: {item['id']}")
        print(f"CHARACTER COUNT: {len(item['document'])}")
        print("-" * 80)
        print(item["document"])
        print("=" * 80 + "\n")


if __name__ == "__main__":
    inspect_specific_chunks(target_indices=[16, 17, 18])
