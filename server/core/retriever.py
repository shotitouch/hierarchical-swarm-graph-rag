from langchain_chroma import Chroma
from core.embeddings import embeddings
from config import PERSIST_DIR
from core.reranker import MiniLMReranker

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

# Base retriever for initial broad search
retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 
reranker = MiniLMReranker()

async def get_reranked_full_context(q: str):
    """
    Retrieves, reranks, and then reconstructs full documents in order.
    """
    # 1. Initial Retrieval (Child Chunks)
    docs = await retriever.ainvoke(q)
    
    # 2. Rerank the chunks to find the most relevant document parts
    reranked_docs = await reranker.rerank(q, docs)
    
    # Limit to top 3 parents to stay within LLM context limits
    top_picks = reranked_docs[:3]
    
    seen_parents = set()
    structured_results = []

    for doc in top_picks:
        parent_id = doc.metadata.get("parent_id")
        
        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            
            # Pull ALL siblings
            full_doc_elements = vectorstore.get(where={"parent_id": parent_id}, include=["documents", "metadatas"])

            # Create element list and sort by element_index
            elements = sorted(
                [{"text": t, "meta": m} for t, m in zip(full_doc_elements['documents'], full_doc_elements['metadatas'])],
                key=lambda x: x['meta'].get('element_index', 0)
            )
            
            # --- START ELITE LOGIC: INLINE PAGE ANCHORS ---
            content_parts = []
            current_page = None
            all_pages = set()

            for e in elements:
                page_num = e['meta'].get("page_number", 1)
                all_pages.add(page_num)

                # Insert a marker ONLY when the page changes
                if page_num != current_page:
                    content_parts.append(f"\n<<< PAGE {page_num} >>>\n")
                    current_page = page_num
                
                content_parts.append(e['text'])
            
            full_text_with_anchors = "\n".join(content_parts)
            # --- END ELITE LOGIC ---
            
            structured_results.append({
                "content": full_text_with_anchors,
                "source": elements[0]['meta'].get("source", "Unknown"),
                "pages": sorted(list(all_pages)),
                "doc_id": parent_id
            })

    # Memory Cleanup
    del docs
    del reranked_docs

    return structured_results