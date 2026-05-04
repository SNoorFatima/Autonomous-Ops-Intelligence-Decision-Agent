import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from schema import ChatRequest, ChatResponse
from graph import build_graph, SYSTEM_PROMPT

# Global instance of graph app
app_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_graph
    # Ensure checkpointer is initialized at global app level
    memory = MemorySaver()
    app_graph = build_graph(memory=memory)
    print("✅ ABIDA Agent Web Service ready.")
    yield
    print("Shutting down ABIDA Agent Web Service.")

app = FastAPI(title="ABIDA LangGraph API", lifespan=lifespan)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Standard REST endpoint that waits for the entire response.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Check if thread is new
    existing_state = app_graph.get_state(config)
    if not existing_state.values:
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=request.message)]
    else:
        # If thread exists, just add the new message
        messages = [HumanMessage(content=request.message)]
        
    state = {"messages": messages}
    
    # Run the graph synchronously
    # To run sync in async def, we can just use the sync stream, or ainvoke
    # ainvoke is ideal here to prevent blocking event loop, but .invoke() works.
    final_state = await app_graph.ainvoke(state, config)
    
    last_message = final_state["messages"][-1]
    
    return ChatResponse(
        final_answer=last_message.content,
        status="completed"
    )

@app.post("/stream")
async def stream_endpoint(request: ChatRequest):
    """
    Streaming endpoint emitting Server-Sent Events (SSE).
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    
    async def event_generator():
        existing_state = app_graph.get_state(config)
        if not existing_state.values:
            messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=request.message)]
        else:
            messages = [HumanMessage(content=request.message)]
            
        state = {"messages": messages}
        
        try:
            # Using stream_mode="updates" yields chunks node-by-node
            async for event in app_graph.astream(state, config, stream_mode="updates"):
                for node_name, node_data in event.items():
                    yield f"event: successful\ndata: {json.dumps({'type': 'node', 'content': f'Executed node: {node_name}'})}\n\n"
                    
                    if "messages" in node_data and node_data["messages"]:
                        last_msg = node_data["messages"][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            yield f"event: final\ndata: {json.dumps({'type': 'message', 'content': last_msg.content})}\n\n"
                            
            yield f"event: complete\ndata: {json.dumps({'type': 'status', 'content': 'completed'})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")
