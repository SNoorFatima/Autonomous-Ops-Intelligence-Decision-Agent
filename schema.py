from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    final_answer: str
    status: str
