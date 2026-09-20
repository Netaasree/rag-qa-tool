"""
load_and_chunk.py
-----------------
This script demonstrates the document loading and chunking stage of a RAG pipeline.

What it does:
1. Loads a PDF document ('documents/rag_research_paper.pdf') using `pypdf`.
2. Extracts raw text from every page of the PDF.
3. Splits the full extracted text into smaller, overlapping chunks following our
   project standard (Chunk Size = 500 characters, Overlap = 50 characters)
   using pure Python string slicing (no external chunking libraries).
4. Prints the summary statistics and the first 2 chunks for manual inspection.
"""

from pathlib import Path
from typing import List
from pypdf import PdfReader


def load_pdf_text(pdf_path: Path) -> str:
    """
    Step 1 & 2: Load the PDF file and extract text from every page.

    Parameters:
        pdf_path (Path): Path to the PDF file to read.

    Returns:
        str: Concatenated text from all pages of the document.
    """
    # Verify that the PDF file exists before attempting to read
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"[Step 1] Loading PDF from: {pdf_path}")
    
    # Initialize the PDF reader from pypdf
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    print(f"         Total pages found: {total_pages}")

    # Extract text page by page
    extracted_text_list: List[str] = []
    
    print("[Step 2] Extracting text from all pages...")
    for index, page in enumerate(reader.pages, start=1):
        # page.extract_text() retrieves readable text from the current page
        page_text = page.extract_text()
        if page_text:
            extracted_text_list.append(page_text)
        else:
            print(f"         Warning: Page {index} yielded empty text.")

    # Join the pages together into one continuous text block
    full_text = "\n".join(extracted_text_list)
    print(f"         Extraction complete. Total characters: {len(full_text):,}")
    
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Step 3: Split text into fixed-size chunks with overlap using Python string slicing.

    Standard for this project (defined in .antigravity/skills/chunking-strategy/SKILL.md):
    - chunk_size: 500 characters
    - overlap: 50 characters
    - method: Pure Python string slicing [start:end]

    Why fixed-size chunks?
    - Provides predictable chunk sizes and simplicity without third-party dependencies.

    Why overlap?
    - Prevents sentences, words, and semantic context from getting cut in half
      across chunk boundaries, ensuring retrieval quality remains high.

    Parameters:
        text (str): The continuous document text.
        chunk_size (int): Maximum length of each chunk in characters.
        overlap (int): Number of characters to carry over from the previous chunk.

    Returns:
        List[str]: List of chunked text strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size.")

    chunks: List[str] = []
    
    # Step size determines how far forward we jump after each chunk.
    # For example, with chunk_size=500 and overlap=50, step = 450.
    # Chunk 0: index 0 to 500
    # Chunk 1: index 450 to 950 (sharing characters 450-500 with Chunk 0)
    # Chunk 2: index 900 to 1400 (sharing characters 900-950 with Chunk 1)
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only keep non-empty chunks
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def main() -> None:
    # Resolve the project root and locate the target PDF
    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / "documents" / "rag_research_paper.pdf"

    # Step 1 & 2: Load and extract text
    full_text = load_pdf_text(pdf_path)

    # Step 3: Chunk the extracted text per our standard
    print("\n[Step 3] Chunking text according to project standards...")
    print("         Standards: Chunk Size = 500 characters, Overlap = 50 characters")
    chunks = chunk_text(full_text, chunk_size=500, overlap=50)

    # Step 4: Verification and displaying results
    print("\n[Step 4] Results & Verification:")
    print("=" * 60)
    print(f"Total Chunks Created: {len(chunks)}")
    print("=" * 60)

    # Display the first 2 chunks for verification
    num_to_display = min(2, len(chunks))
    for i in range(num_to_display):
        print(f"\n--- Chunk {i + 1} (Length: {len(chunks[i])} chars) ---")
        print(chunks[i])
        print("-" * 60)


if __name__ == "__main__":
    main()
