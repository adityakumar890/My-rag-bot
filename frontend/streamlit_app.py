"""Streamlit chat interface for the RAG-Based AI Chatbot.

Run with:  streamlit run frontend/streamlit_app.py
"""

import os
from urllib.parse import quote

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="RAG-Based AI Chatbot", page_icon="🤖", layout="wide")


def api_error(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def call_api(method: str, path: str, **kwargs):
    """Call the backend. Returns (data, error_message)."""
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=120, **kwargs)
    except requests.ConnectionError:
        return None, f"Cannot reach the backend at {BACKEND_URL}. Is it running?"
    except requests.Timeout:
        return None, "The backend took too long to respond."
    if not response.ok:
        return None, str(api_error(response))
    return response.json(), None


# ---------------- Sidebar: documents ----------------
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        for f in uploaded_files:
            with st.spinner(f"Processing {f.name}..."):
                data, error = call_api(
                    "POST", "/upload", files={"file": (f.name, f.getvalue(), f.type)}
                )
            if error:
                st.error(f"{f.name}: {error}")
            else:
                st.success(data["message"])

    st.divider()
    st.subheader("Indexed documents")
    docs, error = call_api("GET", "/documents")
    if error:
        st.warning(error)
    elif not docs["documents"]:
        st.caption("No documents yet.")
    else:
        for doc in docs["documents"]:
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{doc['filename']}**  \n{doc['chunks']} chunks")
            if col2.button("🗑️", key=f"del-{doc['filename']}", help="Delete"):
                _, error = call_api("DELETE", f"/documents/{quote(doc['filename'], safe='')}")
                if error:
                    st.error(error)
                else:
                    st.rerun()

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    health, _ = call_api("GET", "/health")
    if health:
        st.caption(f"Backend: online · LLM: {health['llm_provider']}")


# ---------------- Main area: chat ----------------
st.title("🤖 RAG-Based AI Chatbot")
st.caption("Ask questions about your uploaded documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            label = s["filename"] + (f" · page {s['page']}" if s.get("page") else "")
            st.markdown(f"**{label}** (chunk {s['chunk_index']})")
            st.caption(s["content"][:500] + ("..." if len(s["content"]) > 500 else ""))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data, error = call_api("POST", "/chat", json={"question": question})
        if error:
            answer, sources = f"⚠️ {error}", []
        else:
            answer, sources = data["answer"], data["sources"]
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
