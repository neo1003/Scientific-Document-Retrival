from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_pipeline import RAGPipeline


class ChatbotRequestHandler(BaseHTTPRequestHandler):
    pipeline: RAGPipeline | None = None
    static_dir: Path
    default_top_k: int = 4
    default_vector_candidates: int = 12
    default_keyword_candidates: int = 12
    default_rerank_top_n: int = 8

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "pipeline_ready": self.pipeline is not None,
                }
            )
            return

        if path == "/":
            path = "/index.html"

        target = (self.static_dir / path.lstrip("/")).resolve()
        if not str(target).startswith(str(self.static_dir.resolve())) or not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content_type, _ = mimetypes.guess_type(str(target))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.end_headers()
        self.wfile.write(target.read_bytes())

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
            return

        if self.pipeline is None:
            self._send_json(
                {"error": "RAG pipeline is not initialized."},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Request body must be valid JSON."}, status=HTTPStatus.BAD_REQUEST)
            return

        query = str(payload.get("query", "")).strip()
        if not query:
            self._send_json({"error": "The 'query' field is required."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            response = self.pipeline.answer(
                query=query,
                top_k=self._coerce_positive_int(payload.get("top_k"), self.default_top_k),
                vector_candidates=self._coerce_positive_int(
                    payload.get("vector_candidates"),
                    self.default_vector_candidates,
                ),
                keyword_candidates=self._coerce_positive_int(
                    payload.get("keyword_candidates"),
                    self.default_keyword_candidates,
                ),
                rerank_top_n=self._coerce_positive_int(
                    payload.get("rerank_top_n"),
                    self.default_rerank_top_n,
                ),
            )
        except Exception as error:
            self._send_json(
                {"error": f"Failed to answer query: {error}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(response.to_dict())

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _coerce_positive_int(self, value: object, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local chatbot UI for the RAG pipeline.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local UI server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the local UI server.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts_paddle_openai"),
        help="Directory containing the Chroma store and chunk manifest.",
    )
    parser.add_argument(
        "--collection-name",
        default="document_chunks_hierarchical_paddle_openai",
        help="Base collection name for the RAG pipeline.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="Embedding model used by the retriever.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=1536,
        help="Embedding dimensions used by the retriever.",
    )
    parser.add_argument(
        "--answer-model",
        default="gpt-4.1-mini",
        help="Model used for final grounded answers.",
    )
    parser.add_argument(
        "--reranker-model",
        default="gpt-4.1-mini",
        help="Model used for LLM reranking.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Default number of reranked chunks sent to generation.",
    )
    parser.add_argument(
        "--vector-candidates",
        type=int,
        default=12,
        help="Default dense retrieval candidate count.",
    )
    parser.add_argument(
        "--keyword-candidates",
        type=int,
        default=12,
        help="Default sparse retrieval candidate count.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=8,
        help="Default rerank window size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    static_dir = ROOT / "ui"
    pipeline = RAGPipeline(
        output_dir=args.output_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        answer_model=args.answer_model,
        reranker_model=args.reranker_model,
    )

    ChatbotRequestHandler.pipeline = pipeline
    ChatbotRequestHandler.static_dir = static_dir
    ChatbotRequestHandler.default_top_k = args.top_k
    ChatbotRequestHandler.default_vector_candidates = args.vector_candidates
    ChatbotRequestHandler.default_keyword_candidates = args.keyword_candidates
    ChatbotRequestHandler.default_rerank_top_n = args.rerank_top_n

    server = ThreadingHTTPServer((args.host, args.port), ChatbotRequestHandler)
    print(f"RAG chatbot UI running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
