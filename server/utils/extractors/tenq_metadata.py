import re
from schemas import TenQMetadata
from core.llm import llm

TRADING_SYMBOL_RE = re.compile(
    r"\bTrading\s+Symbol(?:s)?\b\s*[:\-]?\s*([A-Z]{1,6})\b",
    re.IGNORECASE,
)

EXCHANGE_TICKER_RE = re.compile(
    r"\(\s*(NASDAQ|NYSE|AMEX|NYSEAMERICAN)\s*:\s*([A-Z]{1,6})\s*\)",
    re.IGNORECASE,
)

ENDED_DATE_RE = re.compile(
    r"\b(?:quarterly\s+period|quarter|three\s+months)\s+ended\s+"
    r"(March|June|September)\s+\d{1,2},\s+(20\d{2})\b",
    re.IGNORECASE,
)

MONTH_TO_Q = {
    "march": "Q1",
    "june": "Q2",
    "september": "Q3",
}


def regex_extract_tenq_metadata(
    cover_text: str
) -> TenQMetadata:
    """
    Deterministic, cheap extraction.
    Returns TenQMetadata (may contain None fields).
    """
    text = cover_text or ""

    ticker = None
    m = TRADING_SYMBOL_RE.search(text)
    if m:
        ticker = m.group(1).upper()
    else:
        m = EXCHANGE_TICKER_RE.search(text)
        if m:
            ticker = m.group(2).upper()

    year = None
    period = None
    m = ENDED_DATE_RE.search(text)
    if m:
        month = m.group(1).lower()
        year = int(m.group(2))
        period = MONTH_TO_Q.get(month)

    return TenQMetadata(
        ticker=ticker,
        year=year,
        period=period,
    )


async def llm_extract_tenq_metadata(cover_text: str) -> TenQMetadata:
    structured_llm = llm.with_structured_output(TenQMetadata)

    prompt = f"""
        Extract ticker, year, and quarter from this SEC Form 10-Q cover text.

        TEXT:
        {cover_text}
    """

    result: TenQMetadata = await structured_llm.ainvoke(prompt)

    return result