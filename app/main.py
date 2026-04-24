from app.vector_store import VectorStore
from pathlib import Path
from app.rag import RAG

store = VectorStore() #Initialize empty vector store

BASE_DIR = Path(__file__).resolve().parent #Get the directory of the current file (main.py) and resolve it to an absolute path
file_path = BASE_DIR / "data" / "sample.txt" #Construct the full path to the sample.txt file located in the data subdirectory of the base directory

with open(file_path, "r") as f:
    texts = f.readlines() #Read the contents of the sample.txt file into a list of lines
texts = [t.strip() for t in texts if t.strip()] #Remove leading/trailing whitespace and filter out empty lines

store.add_texts(texts) #Add the texts to the vector store, which will generate embeddings and build the FAISS index

#We add RAG here:
rag = RAG(store) #Initialize the RAG system with the vector store

while True:
    query = input("Enter your question (or 'exit' to quit): ") #Prompt the user to enter a question
    if query.lower() == "exit": #Check if the user wants to exit the program
        break
    answer = rag.answer(query) #Get the answer from the RAG system based on the user's query
    print("Answer:", answer) #Print the answer to the console