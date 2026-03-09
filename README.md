# FactMCP Shopping Assistant

Vercel-friendly shopping assistant API that calls an MCP inventory backend and Ollama.

## Project layout

```text
.
├── api/
│   └── index.py
├── data/
│   ├── faq.json
│   └── store_info.json
├── e_series_mcp/
│   ├── __init__.py
│   └── server.py
├── shopping_assistant/
│   ├── __init__.py
│   └── app.py
├── .env.example
├── README.md
├── requirements.txt
├── server.py
└── vercel.json
```

## Environment

Create a `.env` file from `.env.example` and set:

- `MCP_URL`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `OLLAMA_API_KEY` if you use hosted Ollama
- `API_KEY` to protect `/chat`, `/catalog`, and `/clear-cache`

## Endpoints

- `POST /chat`
- `GET /health`
- `GET /catalog`
- `POST /clear-cache`

## Local run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn api.index:app --reload
```

## Notes

- `api/index.py` exports the FastAPI app Vercel serves as a Python serverless function.
- The assistant talks to your MCP backend via `MCP_URL/tools/...`.
- The assistant talks to Ollama via `/api/chat`.
- The original MCP backend code is still in `e_series_mcp/` if you need it separately.
