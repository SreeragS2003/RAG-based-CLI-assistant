from fastapi import FastAPI
from pydantic import BaseModel, Field #For request body validation
from contextlib import asynccontextmanager
from app.agent import run_agent
from app.vector_store import VectorStore
from app.memory_store import Memory
from app.multi_pdf_loader import load_all_pdfs
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

store = None #Global variable to hold the vector store instance, which will be initialized in the lifespan function to ensure it is ready before handling any requests.
memory_store = {} #In-memory store for user conversations, mapping user_id to their respective Memory instance

class Query(BaseModel):
    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)

@asynccontextmanager
async def lifespan(app: FastAPI): #Lifespan handler to manage startup and shutdown events of the FastAPI app, ensuring that the vector store is properly initialized and loaded before the app starts handling requests.
    global store
    store = VectorStore() #Initialize vector store
    if store.load():
        print("Loaded existing index")
    else:
        print("Building index...")

        all_chunks, metadata = load_all_pdfs() #Load all PDFs from the specified folder, extract their text, and chunk it into manageable pieces. This function returns a list of all text chunks and their associated metadata (such as source and chunk ID).

        store.add_texts(all_chunks, metadata) #Add the text chunks to the vector store, which will generate embeddings and build the FAISS index
        store.save()

        print("Index built and saved successfully")
    yield #App runs here

    print("Shutting down...") #Cleanup code can go here if needed

app = FastAPI(lifespan=lifespan) #Create FastAPI app instance with the defined lifespan function to handle startup and shutdown events, ensuring that the vector store is properly initialized and loaded before the app starts handling requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/chat")
async def chat(q: Query):
    try:
        if q.user_id not in memory_store:
            memory_store[q.user_id] = Memory() #Separate memory for each user based on user_id, allowing the assistant to maintain context across multiple interactions with the same user while keeping different users' conversations isolated from each other.

        user_memory = memory_store[q.user_id]
        return StreamingResponse(run_agent(q.query, store, user_memory), media_type="text/plain")
    except Exception as e:
        return {"error": str(e)}