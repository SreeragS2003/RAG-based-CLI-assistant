import os
from app.loader import load_pdf
from app.chunker import chunk_text

def load_all_pdfs(folder="./app/data"):
    all_chunks = [] #Unified chunk pool
    metadata = []

    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)

            text = load_pdf(path)
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "content": chunk,
                    "source": filename,
                    "chunk_id": i
                })

    return all_chunks, metadata