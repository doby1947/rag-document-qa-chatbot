"""Streamlit chat UI.  Run: streamlit run streamlit_app.py"""
import os
import uuid

import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
st.title("📄 Intelligent Document Q&A")

if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("Index documents", disabled=not files):
        with st.spinner("Indexing..."):
            r = requests.post(
                f"{BACKEND}/upload",
                files=[("files", (f.name, f.getvalue())) for f in files],
                timeout=600,
            )
        if r.ok:
            st.success(f"Added {r.json()['chunks_added']} chunks")
        else:
            st.error(r.text)
    if st.button("New conversation"):
        requests.delete(f"{BACKEND}/session/{st.session_state.sid}", timeout=30)
        st.session_state.messages = []
        st.rerun()


def render_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            page = f", p.{s['page']}" if s["page"] else ""
            st.markdown(f"**[{s['id']}] {s['source']}{page}**  \n{s['snippet']}...")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"])

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = requests.post(
                f"{BACKEND}/ask", json={"question": prompt, "session_id": st.session_state.sid}, timeout=120
            )
        if r.ok:
            data = r.json()
            st.markdown(data["answer"])
            st.caption(f"Answered in {data['latency_s']}s")
            render_sources(data["sources"])
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
            )
        else:
            st.error(r.text)
