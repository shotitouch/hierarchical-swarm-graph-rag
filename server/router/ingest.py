# router/ingest.py
import uuid
import tempfile
from fastapi import APIRouter, UploadFile
from unstructured.partition.pdf import partition_pdf
from core.retriever import vectorstore
from utils.extractors.tenq_metadata import regex_extract_tenq_metadata, llm_extract_tenq_metadata
from utils.vision.financial_image import summarize_financial_image
router = APIRouter(prefix="/ingest")

@router.post("/")
async def ingest_10q_multimodal(file: UploadFile):
    """
    Modern multimodal 10-Q ingestion:
    - deterministic first
    - structured LLM fallback
    - modality-aware storage
    """

    # --------------------------------------------------
    # 1. Save PDF to temp file (required by unstructured)
    # --------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    elements = partition_pdf(
        filename=pdf_path,
        strategy="auto",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy=None,
    )

    # --------------------------------------------------
    # 2. Extract cover text safely
    # --------------------------------------------------
    cover_text = "\n".join(
        el.text for el in elements[:20]
        if hasattr(el, "text") and el.text
    )[:3000]

    # --------------------------------------------------
    # 3. Regex-first metadata extraction
    # --------------------------------------------------
    regex_metadata = regex_extract_tenq_metadata(cover_text)
    ticker = regex_metadata.ticker
    year = regex_metadata.year
    period = regex_metadata.period
    used_llm = False

    # --------------------------------------------------
    # 4. Structured LLM fallback (ONLY if needed)
    # --------------------------------------------------
    if ticker is None or year is None or period is None:
        used_llm = True
        llm_metadata = await llm_extract_tenq_metadata(cover_text)
        
        if ticker is None:
            ticker = llm_metadata.ticker
        if year is None:
            year = llm_metadata.year
        if period is None:
            period = llm_metadata.period


    ticker = ticker.strip().upper() if isinstance(ticker, str) else "UNKNOWN"

    metadata_complete = (
        ticker != "UNKNOWN" and
        year is not None and
        period is not None
    )

    # --------------------------------------------------
    # 5. Build vectorstore payload (modality-aware)
    # --------------------------------------------------
    parent_id = str(uuid.uuid4())
    texts = []
    metadatas = []

    for idx, el in enumerate(elements):
        base_meta = {
            "parent_id": parent_id,
            "ticker": ticker,
            "year": year,
            "period": period,
            "metadata_complete": metadata_complete,
            "source": file.filename,
            "page_number": el.metadata.page_number or 1,
            "element_index": idx,
        }

        # -------- IMAGE --------
        if el.category == "Image" and el.metadata.image_base64:
            summary = await summarize_financial_image(
                el.metadata.image_base64
            )
            texts.append(summary)
            metadatas.append({
                **base_meta,
                "modality": "image",
                "type": "chart",
            })

        # -------- TABLE --------
        elif el.category == "Table":
            table_html = el.metadata.text_as_html or el.text
            texts.append(table_html)
            metadatas.append({
                **base_meta,
                "modality": "table",
                "type": "financial_table",
            })

        # -------- TEXT --------
        elif hasattr(el, "text") and el.text:
            texts.append(el.text)
            metadatas.append({
                **base_meta,
                "modality": "text",
                "type": "narrative",
            })

    if not texts:
        raise ValueError("No ingestable content extracted from 10-Q")

    # --------------------------------------------------
    # 6. Persist
    # --------------------------------------------------
    await vectorstore.aadd_texts(
        texts=texts,
        metadatas=metadatas,
    )

    return {
        "status": "success",
        "parent_id": parent_id,
        "ticker": ticker,
        "year": year,
        "period": period,
        "used_llm_fallback": used_llm,
        "chunks": len(texts),
    }
