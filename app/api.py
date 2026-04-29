from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field #For request body validation
from contextlib import asynccontextmanager
from app.vector_store import VectorStore
from app.memory_store import Memory
from app.multi_pdf_loader import load_all_pdfs
from fastapi.middleware.cors import CORSMiddleware
from app.agent import initialize_agent
from app.tools import create_search_tool, calculator
from app.strip_markdown import strip_markdown
from langchain_core.messages import HumanMessage

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
     # Create tools AFTER store
    search_tool = create_search_tool(store)
    tools = [search_tool, calculator]

    # Create agent AFTER tools
    agent = initialize_agent(tools)
    print("Visualizing the agent graph created",agent.get_graph().draw_ascii())

    # Store in app.state
    app.state.store = store #Not utilized right now
    app.state.agent = agent
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
async def chat(q: Query, request: Request):
    try:
        agent = request.app.state.agent

        if q.user_id not in memory_store:
            memory_store[q.user_id] = Memory()

        user_memory = memory_store[q.user_id]

        history = user_memory.get()

        # Build query with history as context, not as system message
        query_with_context = q.query
        if history:
            query_with_context = f"Previous context:\n{history}\n\nCurrent question: {q.query}"

        result = await agent.ainvoke({ #This goes to entry point of graph which is set as call_llm
            "messages": [HumanMessage(content=query_with_context)]
        })

        answer = strip_markdown(result["messages"][-1].content)

        user_memory.add(q.query, answer)

        async def generator():
            for word in answer.split():
                yield word + " "

        return StreamingResponse(generator(), media_type="text/plain")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": repr(e)} #For showing hidden exceptions