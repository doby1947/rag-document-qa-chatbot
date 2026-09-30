import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
UPLOAD_DIR = Path("data/uploads")
INDEX_DIR = Path("vectorstore")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_HISTORY_TURNS = 6
