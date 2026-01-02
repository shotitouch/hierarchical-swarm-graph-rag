from typing import List, TypedDict, Annotated, Optional
from langchain_core.documents import Document
from langgraph.graph.message import add_messages, AnyMessage

class DocumentContext(TypedDict):
    content: str      # The full, sorted, reconstructed text/summaries
    source: str       # Filename (e.g., CV_aug_eng.pdf)
    pages: List[int]  # All page numbers involved in this context
    doc_id: str       # The parent_id for tracking

class AgentState(TypedDict):
    question: str
    intent: str
    # Annotated with add_messages makes this a "living" history list
    messages: Annotated[list[AnyMessage], add_messages]
    documents: List[DocumentContext]
    fiscal_info: Optional[dict]  # e.g., {"ticker": "AAPL", "year": 2025, "period": "Q3"}
    generation: str
    retry_count: int
    is_grounded: str  # 'yes' or 'no'
    is_useful: str    # 'yes' or 'no'