def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()

    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i+chunk_size])
        chunks.append(chunk)
    
    print(f"Total chunks created: {len(chunks)}") #Print the total number of chunks created from the original text, which can be useful for debugging and understanding how the text has been segmented.

    return chunks