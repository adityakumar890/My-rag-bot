# RAG-Based AI Chatbot

An AI chatbot that answers questions about **your own documents**. You upload PDF/TXT files, and the bot uses **Retrieval-Augmented Generation (RAG)** to find the relevant parts and asks a Large Language Model (Gemini by default) to answer from them.

**Stack:** Python · FastAPI · Streamlit · LangChain · ChromaDB · Sentence Transformers · Gemini (or OpenAI)

---

## Architecture

```
┌──────────────────────┐   HTTP (JSON)   ┌──────────────────────────────────────────────┐
│  Streamlit frontend  │ ──────────────> │               FastAPI backend                │
│  frontend/           │                 │               app/main.py                    │
│  streamlit_app.py    │                 │                                              │
│  - upload sidebar    │ <────────────── │                                              │
│  - chat interface    │                 │  POST /upload                                │
└──────────────────────┘                 │    1. validate file (type, size, content)    │
                                         │    2. extract text (pypdf / txt decode)      │
                                         │    3. split into chunks (LangChain)          │
                                         │    4. embed chunks (Sentence Transformers)   │
                                         │    5. store vectors in ChromaDB (on disk)    │
                                         │                                              │
                                         │  POST /chat                                  │
                                         │    1. embed the question                     │
                                         │    2. similarity search in ChromaDB (top-k)  │
                                         │    3. prompt = context chunks + question     │
                                         │    4. LLM (Gemini / OpenAI) generates answer │
                                         │    5. return answer + source chunks          │
                                         └──────────────────────────────────────────────┘
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Backend status and active LLM provider |
| POST | `/upload` | Upload a PDF/TXT file (multipart field `file`) |
| POST | `/chat` | Ask a question: `{"question": "...", "top_k": 4}` |
| GET | `/documents` | List indexed documents and chunk counts |
| DELETE | `/documents/{filename}` | Delete one document |
| DELETE | `/documents` | Delete all documents |

Interactive API docs: http://localhost:8000/docs

## Project structure

```
.
├── app/
│   ├── main.py                 # FastAPI app, routes and request/response models
│   ├── chatbot.py              # LLM provider (gemini/openai/fake) + RAG pipeline (ingest, answer)
│   ├── document_loader.py      # File validation, text extraction and chunking
│   ├── embeddings.py           # Sentence Transformers embedding model
│   ├── vector_store.py         # ChromaDB storage and similarity search
│   └── config.py               # Settings loaded from .env
├── frontend/
│   └── streamlit_app.py        # Streamlit chat UI
├── tests/                      # Pytest tests (run offline)
├── data/                       # Uploaded files + ChromaDB (created at runtime, gitignored)
├── requirements.txt
├── .env.example                # Template for your .env (no real secrets)
└── README.md
```

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/adityakumar890/My-rag-bot.git
cd My-rag-bot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> Tip: to avoid downloading the large GPU build of PyTorch, first run
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

### Configure

```bash
cp .env.example .env             # Windows: copy .env.example .env
```

Edit `.env` and set your key:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-from-https://aistudio.google.com/app/apikey
```

- Use `LLM_PROVIDER=openai` + `OPENAI_API_KEY` to use OpenAI instead.
- Use `LLM_PROVIDER=fake` to try the app without any API key (placeholder answers).
- `ANSWER_MODE=documents` (default) answers only from your uploaded files. `ANSWER_MODE=hybrid` uses your files when they contain the answer and otherwise answers from the LLM's general knowledge, labelled "Not from your documents" and shown without sources. Restart the backend after changing it.
- `.env` is listed in `.gitignore` — **never commit your real API keys.**

## Running

Open two terminals (with the virtual environment activated in both):

```bash
# Terminal 1 - backend (http://localhost:8000)
uvicorn app.main:app --reload

# Terminal 2 - frontend (http://localhost:8501)
streamlit run frontend/streamlit_app.py
```

Then open http://localhost:8501, upload documents in the sidebar, click **Process documents**, and start asking questions.

The first upload downloads the embedding model (~90 MB) from Hugging Face, so it can take a minute.

## Running tests

```bash
pytest
```

Tests use a fake LLM, fake embeddings and a temporary database, so they need no API key or internet connection.

## Error handling & validation

- Only `.pdf` and `.txt` files are accepted; size is limited by `MAX_FILE_SIZE_MB` (default 10 MB).
- Empty files, fake/corrupted/encrypted PDFs, and PDFs without extractable text (scanned images) are rejected with a clear message.
- Asking a question before uploading any document returns a helpful error (in `hybrid` mode it answers from general knowledge instead).
- Missing API keys and LLM/API failures (timeouts, quota, unavailable model) return readable errors shown in the chat.
- Re-uploading a file with the same name replaces its old chunks.

## Configuration reference

All settings live in `.env` (see `.env.example`): LLM provider/model/temperature/timeout, embedding model, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `ANSWER_MODE`, storage paths and `BACKEND_URL`.
