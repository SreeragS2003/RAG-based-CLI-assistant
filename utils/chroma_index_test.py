# run this once in a python shell
from app.vector_store import VectorStore
store = VectorStore()
print("Total chunks:", store.collection.count())

# peek at all documents
results = store.collection.get(include=["documents", "metadatas"])
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(meta["source"], "|", doc[:80])