from app.vector_store import VectorStore
from pathlib import Path
from app.rag import RAG
from app.loader import load_pdf
from app.chunker import chunk_text

store = VectorStore() #Initialize empty vector store
# Try loading existing index
if store.load():
    print("Loaded existing index")
else:
    print("Building index...")

    text = load_pdf("./app/data/aurora.pdf")
    chunks = chunk_text(text)

    store.add_texts(chunks, source="aurora.pdf") #Add the text chunks to the vector store, which will generate embeddings and build the FAISS index
    store.save()

    print("Index built and saved successfully")

#We add RAG here:
rag = RAG(store) #Initialize the RAG system with the vector store

while True:
    query = input("Enter your question (or 'exit' to quit): ") #Prompt the user to enter a question
    if query.lower() == "exit": #Check if the user wants to exit the program
        break
    answer = rag.answer(query) #Get the answer from the RAG system based on the user's query
    print("Answer:", answer) #Print the answer to the console