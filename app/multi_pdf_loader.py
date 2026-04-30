import os
from app.chunker import chunk_pdf

def load_all_pdfs(folder="./app/data"):
    all_chunks = []  # Unified chunk pool
    metadata = []
    chunk_id = 0  # Global counter across all PDFs to ensure unique IDs

    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)

            # chunk_pdf handles both loading and chunking internally
            # since unstructured works best when partitioning and chunking together
            chunks = chunk_pdf(path)

            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({
                    "content": chunk,
                    "source": filename,
                    "chunk_id": str(chunk_id)  # string ID, unique across all PDFs 
                })
                chunk_id += 1  # increment globally, not per PDF
    #All these adjustments are for ChromaDB
    return all_chunks, metadata