from pydantic import BaseModel
from typing import Optional, Literal

class ChatRequest(BaseModel):
    question: str
    thread_id: str = "1"

class TenQMetadata(BaseModel):
    ticker: Optional[str]
    year: Optional[int]
    period: Optional[Literal["Q1", "Q2", "Q3"]]
