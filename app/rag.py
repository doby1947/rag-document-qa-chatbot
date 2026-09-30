"""RAG engine: semantic chunking -> Sentence-Transformer embeddings -> FAISS -> Gemini."""
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config as C
from app.loaders import SUPPORTED, load_file

ANSWER_PROMPT = """You are a careful document assistant. Answer the question using ONLY the context below.
Cite sources inline with their numbers like [1] or [2]. If the answer is not in the context, say
"I couldn't find that in the provided documents." Do not invent facts.

Context:
{context}

Conversation so far:
{history}

Question: {question}

Answer:"""

CONDENSE_PROMPT = """Given the conversation and a follow-up question, rewrite the follow-up as a single
standalone question. Return only the rewritten question.

Conversation:
{history}

Follow-up: {question}

Standalone question:"""


class RagEngine:
    def __init__(self):
        C.INDEX_DIR.mkdir(exist_ok=True)
        C.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=C.EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=C.CHUNK_SIZE,
            chunk_overlap=C.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.store = None
        if (C.INDEX_DIR / "index.faiss").exists():
            self.store = FAISS.load_local(
                str(C.INDEX_DIR), self.embeddings, allow_dangerous_deserialization=True
            )
        self._llm = None
        self.sessions: Dict[str, List[tuple]] = defaultdict(list)

    @property
    def llm(self):
        if self._llm is None:  # reads GOOGLE_API_KEY from the environment
            self._llm = ChatGoogleGenerativeAI(model=C.GEMINI_MODEL, temperature=0.2)
        return self._llm

    # ---- ingestion ---------------------------------------------------------
    def add_files(self, paths: List[Path]) -> int:
        docs = []
        for p in paths:
            if p.suffix.lower() in SUPPORTED:
                docs.extend(load_file(p))
        chunks = self.splitter.split_documents(docs)
        if not chunks:
            return 0
        if self.store is None:
            self.store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.store.add_documents(chunks)
        self.store.save_local(str(C.INDEX_DIR))
        return len(chunks)

    def stats(self) -> dict:
        return {"chunks": self.store.index.ntotal if self.store else 0}

    # ---- question answering ------------------------------------------------
    def _history_text(self, session_id: str) -> str:
        turns = self.sessions[session_id][-C.MAX_HISTORY_TURNS:]
        return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in turns) or "(none)"

    def ask(self, question: str, session_id: str = "default") -> dict:
        if self.store is None:
            return {"answer": "No documents indexed yet. Upload a document first.", "sources": [], "latency_s": 0}
        start = time.perf_counter()
        history = self._history_text(session_id)

        standalone = question
        if self.sessions[session_id]:  # resolve follow-ups like "what about its cost?"
            standalone = self.llm.invoke(
                CONDENSE_PROMPT.format(history=history, question=question)
            ).content.strip()

        hits = self.store.similarity_search_with_score(standalone, k=C.TOP_K)
        context_parts, sources = [], []
        for i, (doc, score) in enumerate(hits, 1):
            page = doc.metadata.get("page")
            context_parts.append(
                f"[{i}] ({doc.metadata['source']}, page {page or 'n/a'})\n{doc.page_content}"
            )
            sources.append(
                {
                    "id": i,
                    "source": doc.metadata["source"],
                    "page": page,
                    "snippet": doc.page_content[:250].replace("\n", " "),
                    "distance": round(float(score), 4),
                }
            )

        answer = self.llm.invoke(
            ANSWER_PROMPT.format(context="\n\n".join(context_parts), history=history, question=question)
        ).content.strip()

        self.sessions[session_id].append((question, answer))
        return {
            "answer": answer,
            "sources": sources,
            "standalone_question": standalone,
            "latency_s": round(time.perf_counter() - start, 2),
        }

    def reset(self, session_id: str):
        self.sessions.pop(session_id, None)
