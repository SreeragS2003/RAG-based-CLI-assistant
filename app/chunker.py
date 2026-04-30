from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

def chunk_pdf(path, max_characters=800, overlap=50): #800 characters - earlier it was word-size which was 200
    # Partition first to get structured elements
    elements = partition_pdf(
        filename=path,
        strategy="fast",
        infer_table_structure=True
    )

    # Chunk by title — keeps related content together under the same section heading
    # Much smarter than word-count chunking which splits mid-sentence or mid-table
    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        overlap=overlap,
        include_orig_elements=False
    )

    print(f"Total chunks created: {len(chunks)}")
    return [str(c) for c in chunks]