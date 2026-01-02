import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app import app
from schemas import TenQMetadata

class FakeMetadata:
    def __init__(
        self,
        page_number=1,
        image_base64=None,
        text_as_html=None,
    ):
        self.page_number = page_number
        self.image_base64 = image_base64
        self.text_as_html = text_as_html


class FakeElement:
    def __init__(self, category, text=None, metadata=None):
        self.category = category
        self.text = text
        self.metadata = metadata or FakeMetadata()

client = TestClient(app)

# case: regex captures all metadata
def test_ingest_10q_regex_only():
    fake_elements = [
        FakeElement(
            category="NarrativeText",
            text="Trading Symbol: AAPL\nFor the quarterly period ended June 30, 2024",
            metadata=FakeMetadata(page_number=1),
        )
    ]

    with patch(
        "router.ingest.partition_pdf",
        return_value=fake_elements,
    ), patch(
        "router.ingest.vectorstore.aadd_texts",
        new_callable=AsyncMock,
    ) as mock_add, patch(
        "router.ingest.llm_extract_tenq_metadata",
        new_callable=AsyncMock,
    ) as mock_llm:

        response = client.post(
            "/ingest",
            files={
                "file": (
                    "test.pdf",
                    b"%PDF-1.4 fake pdf",
                    "application/pdf",
                )
            },
        )

        assert response.status_code == 200

        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["year"] == 2024
        assert data["period"] == "Q2"
        assert data["used_llm_fallback"] is False

        mock_llm.assert_not_called()
        mock_add.assert_awaited_once()

# case: fallback to llm metadata extraction
def test_ingest_10q_llm_fallback():
    fake_elements = [
        FakeElement(
            category="NarrativeText",
            text="For the quarterly period ended September 30, 2023",
            metadata=FakeMetadata(page_number=1),
        )
    ]

    llm_result = TenQMetadata(
        ticker="MSFT",
        year=2023,
        period="Q3",
    )

    with patch(
        "router.ingest.partition_pdf",
        return_value=fake_elements,
    ), patch(
        "router.ingest.vectorstore.aadd_texts",
        new_callable=AsyncMock,
    ), patch(
        "router.ingest.llm_extract_tenq_metadata",
        new=AsyncMock(return_value=llm_result),
    ) as mock_llm:

        response = client.post(
            "/ingest",
            files={
                "file": (
                    "test.pdf",
                    b"%PDF-1.4 fake pdf",
                    "application/pdf",
                )
            },
        )

        assert response.status_code == 200

        data = response.json()
        assert data["ticker"] == "MSFT"
        assert data["year"] == 2023
        assert data["period"] == "Q3"
        assert data["used_llm_fallback"] is True

        mock_llm.assert_awaited_once()

# check metadata modality inserted into vector db
def test_ingest_modality_metadata():
    fake_elements = [
        FakeElement(
            category="Table",
            metadata=FakeMetadata(text_as_html="<table>Revenue</table>"),
        ),
        FakeElement(
            category="NarrativeText",
            text="Trading Symbol: AAPL\nFor the quarterly period ended June 30, 2024",
        ),
    ]

    with patch(
        "router.ingest.partition_pdf",
        return_value=fake_elements,
    ), patch(
        "router.ingest.regex_extract_tenq_metadata",
        return_value=TenQMetadata(
            ticker="AAPL",
            year=2024,
            period="Q2",
        ),
    ), patch(
        "router.ingest.llm_extract_tenq_metadata",
        new=AsyncMock(
            return_value=TenQMetadata(
                ticker="AAPL",
                year=2024,
                period="Q2",
            )
        ),
    ), patch(
        "router.ingest.vectorstore.aadd_texts",
        new_callable=AsyncMock,
    ) as mock_add:


        client.post(
            "/ingest",
            files={
                "file": (
                    "test.pdf",
                    b"%PDF-1.4 fake pdf",
                    "application/pdf",
                )
            },
        )

        _, kwargs = mock_add.await_args
        metadatas = kwargs["metadatas"]

        assert metadatas[0]["modality"] == "table"
        assert metadatas[1]["modality"] == "text"
