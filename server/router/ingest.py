# router/ingest.py
import os
import uuid
from fastapi import APIRouter, UploadFile
from unstructured.partition.pdf import partition_pdf
from core.llm import llm
from core.retriever import vectorstore

router = APIRouter(prefix="/ingest")

async def summarize_image(base64_str):
    """Cleans and sends image to Vision LLM."""
    # Clean the string to avoid 400 errors from extra whitespace
    clean_base64 = base64_str.replace("\n", "").strip()
    
    prompt = "Describe this professional profile picture or document image for a RAG system."
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_base64}"}}
    ]
    
    res = await llm.ainvoke([{"role": "user", "content": content}])
    return res.content

@router.post("/")
async def ingest_pro_multimodal(file: UploadFile):
    # 1. Partition with NO chunking to prevent image 'swallowing'
    raw_elements = partition_pdf(
        file=file.file,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy=None # Keep elements separate first
    )

    # 2. Assign a shared Parent ID to maintain the relationship
    parent_id = str(uuid.uuid4())
    processed_texts = []
    metadatas = []

    for index, el in enumerate(raw_elements):
        category = el.category
        metadata = {
            "source": file.filename,
            "parent_id": parent_id, # Link all elements to this document
            "type": category,
            "page_number": el.metadata.page_number or 1,
            "element_index": index
        }

        if category == "Image":
            # Process image as its own searchable node
            summary = await summarize_image(el.metadata.image_base64)
            processed_texts.append(f"IMAGE SUMMARY: {summary}")
            metadatas.append(metadata)
        
        elif category == "Table":
            # Process table html as its own node
            table_html = el.metadata.text_as_html or str(el)
            processed_texts.append(f"TABLE DATA: {table_html}")
            metadatas.append(metadata)
            
        else:
            # Standard text elements (Title, NarrativeText)
            processed_texts.append(str(el))
            metadatas.append(metadata)

    # 3. Add to vector store
    # Note: For even better 'Pro' results, apply a TextSplitter to the text 
    # elements here while keeping the shared parent_id.
    await vectorstore.aadd_texts(texts=processed_texts, metadatas=metadatas)
    
    return {"status": "success", "parent_id": parent_id, "elements": len(raw_elements)}