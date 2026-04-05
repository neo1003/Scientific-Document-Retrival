# Scientific Document RAG

Scientific Document RAG is an end-to-end retrieval-augmented generation pipeline for scientific content. It ingests page-image documents, extracts text with OCR, builds a searchable vector index, and returns grounded answers to natural-language questions.

## Project Overview

The pipeline combines OCR, hierarchical chunking, hybrid retrieval, and LLM-based reranking to improve answer quality on dense scientific documents.

Key technologies:

- `PaddleOCR` for local OCR on page images.
- `text-embedding-3-large` for semantic chunk embeddings.
- `ChromaDB` for persistent vector storage.
- OpenAI chat models for reranking and final answer generation.

## Core Capabilities

- OCR ingestion from zipped page-image collections.
- Hierarchical chunking into section and leaf chunks.
- Hybrid retrieval that fuses dense vector and keyword matching.
- LLM reranking of retrieval candidates before generation.
- Grounded answer generation from top retrieved evidence.
- Optional local chatbot UI for interactive querying.

## Quickstart

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Set your OpenAI API key

```bash
export OPENAI_API_KEY="your_key_here"
```

### 3) Build the index

Provide a zip file containing document page images:

```bash
python scripts/build_index.py --zip-path path/to/pages.zip --output-dir artifacts_paddle_openai
```

### 4) Ask a question

```bash
python src/rag_pipeline.py --mode query --output-dir artifacts_paddle_openai --query "What are the key findings on eGFR equations?"
```

### 5) Run the optional web UI

```bash
python scripts/chatbot_ui.py --output-dir artifacts_paddle_openai --port 8080
```

Open `http://127.0.0.1:8080` in your browser.

## Minimal Project Structure

- [src/rag_pipeline.py](src/rag_pipeline.py): core ingestion, retrieval, reranking, and answer pipeline.
- [scripts/build_index.py](scripts/build_index.py): index build entrypoint.
- [scripts/query_rag.py](scripts/query_rag.py): CLI query entrypoint.
- [scripts/chatbot_ui.py](scripts/chatbot_ui.py): local web server for chat + API.
- [ui/index.html](ui/index.html): browser chat client.
- [docs/ragas_queries.jsonl](docs/ragas_queries.jsonl): sample evaluation query set.

## Notes and Limitations

- `OPENAI_API_KEY` is required for embeddings, reranking, and answers.
- OCR quality depends on source document quality and layout complexity.
- First run may be slower while OCR models initialize/download.
- Default models are `text-embedding-3-large` (1536 dimensions) and `gpt-4.1-mini` for reranking and answer generation.

## Next Steps

1. Add citation formatting for answer snippets in UI and CLI outputs.
2. Expand evaluation coverage with additional domain-specific query sets.
3. Add deployment packaging for a long-running API service.
