from app.vector_store import VectorStore
from app.rag import RAG
from app.multi_pdf_loader import load_all_pdfs
from app.agent import run_agent

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

while True:
    query = input("Enter your question (or 'exit' to quit): ") #Prompt the user to enter a question
    if query.lower() == "exit": #Check if the user wants to exit the program
        break
    result = run_agent(query, store)
    print("Answer:", end=" ", flush=True) #Print "Answer:" without a newline and flush the output buffer to ensure it appears immediately
    for chunk in result:
        print(chunk, end="", flush=True)
    print()