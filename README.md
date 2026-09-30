# Intelligent Document Q&A Chatbot (RAG + Gemini)

Python · LangChain · Google Gemini API · FAISS · Sentence Transformers · FastAPI · Streamlit

## Features
- Multi-format ingestion: PDF (page-level), DOCX, TXT
- Semantic chunking (recursive splitter) + `all-MiniLM-L6-v2` embeddings + FAISS index (persisted to disk)
- Grounded answers from Gemini with inline citations `[1]` and source file / page attribution
- Multi-turn memory: follow-up questions are rewritten into standalone queries before retrieval
- FastAPI backend + Streamlit chat UI

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your GOOGLE_API_KEY (https://aistudio.google.com/apikey)
```

## Run (two terminals)
```bash
uvicorn app.api:app --port 8000
streamlit run streamlit_app.py
```
Open http://localhost:8501, upload documents, click **Index documents**, then chat.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status + chunk count |
| POST | `/upload` | multipart files -> index |
| POST | `/ask` | `{question, session_id}` -> answer + sources |
| DELETE | `/session/{id}` | clear conversation memory |

## Evaluate
Build a small Q&A test set of your own documents (question + expected answer) and record the share answered correctly and the latency (returned as `latency_s`). Put your measured numbers here.

## Notes
- `.env` and the FAISS index are git-ignored. Never commit your API key.
- FAISS `load_local` uses pickle: only load indexes you created yourself.
