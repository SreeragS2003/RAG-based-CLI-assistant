from app.vector_store import VectorStore
from app.rag import RAG
from app.multi_pdf_loader import load_all_pdfs

store = VectorStore() #Initialize empty vector store
# Try loading existing index
if store.load():
    print("Loaded existing index")
else:
    print("Building index...")

    all_chunks, metadata = load_all_pdfs() #Load all PDFs from the specified folder, extract their text, and chunk it into manageable pieces. This function returns a list of all text chunks and their associated metadata (such as source and chunk ID).

    store.add_texts(all_chunks, metadata) #Add the text chunks to the vector store, which will generate embeddings and build the FAISS index
    store.save()

    print("Index built and saved successfully")

#We add RAG here:
rag = RAG(store) #Initialize the RAG system with the vector store

while True:
    query = input("Enter your question (or 'exit' to quit): ") #Prompt the user to enter a question
    if query.lower() == "exit": #Check if the user wants to exit the program
        break
    answer, sources = rag.answer(query) #Get the answer from the RAG system based on the user's query
    print("Answer:", answer) #Print the answer to the console
    print("Sources:", sources) #Print the sources that were used to generate the answer, which can help the user understand where the information is coming from and verify its credibility if needed