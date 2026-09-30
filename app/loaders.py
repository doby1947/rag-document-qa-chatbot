"""Multi-format document ingestion (PDF, DOCX, TXT) with page-level metadata."""
from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED = {".pdf", ".docx", ".txt"}


def load_file(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    name = path.name
    if ext == ".pdf":
        docs = []
        for i, page in enumerate(PdfReader(str(path)).pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": name, "page": i + 1}))
        return docs
    if ext == ".docx":
        text = "\n".join(p.text for p in DocxDocument(str(path)).paragraphs if p.text.strip())
        return [Document(page_content=text, metadata={"source": name, "page": None})]
    if ext == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": name, "page": None})]
    raise ValueError(f"Unsupported file type: {ext}")
