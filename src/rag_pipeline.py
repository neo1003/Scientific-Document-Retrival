from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import chromadb
from openai import OpenAI
from PIL import Image

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR
import tiktoken


def require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running the OpenAI-backed pipeline."
        )
    return api_key


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    chunk_index: int
    page_number: int
    chunk_level: str
    section_path: str
    parent_chunk_id: str | None
    text: str
    preview: str
    character_count: int
    token_estimate: int
    page_width: int
    page_height: int
    schema_version: str = "1.0"
    ocr_engine: str = "paddleocr"
    embedding_model: str = "text-embedding-3-large"

    def metadata(self) -> dict[str, str | int]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "chunk_level": self.chunk_level,
            "section_path": self.section_path,
            "parent_chunk_id": self.parent_chunk_id or "",
            "preview": self.preview,
            "character_count": self.character_count,
            "token_estimate": self.token_estimate,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "schema_version": self.schema_version,
            "ocr_engine": self.ocr_engine,
            "embedding_model": self.embedding_model,
            "text": self.text,
        }

    def document_text(self) -> str:
        return self.text


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    section_path: str
    text: str
    parent_chunk_id: str | None
    vector_rank: int | None
    keyword_rank: int | None
    hybrid_score: float
    rerank_score: float | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section_path": self.section_path,
            "parent_chunk_id": self.parent_chunk_id,
            "vector_rank": self.vector_rank,
            "keyword_rank": self.keyword_rank,
            "hybrid_score": round(self.hybrid_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
        }


@dataclass(frozen=True)
class RAGResponse:
    query: str
    answer: str
    contexts: list[str]
    retrieved_chunks: list[SearchResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "answer": self.answer,
            "contexts": self.contexts,
            "retrieved_chunks": [
                {
                    **chunk.metadata(),
                    "text": chunk.text,
                }
                for chunk in self.retrieved_chunks
            ],
        }


class TextEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dimensions: int = 1536,
        cache_path: Path | None = None,
        batch_size: int = 64,
        max_retries: int = 3,
    ) -> None:
        self.client = OpenAI(api_key=require_openai_api_key())
        self.model = model
        self.dimensions = dimensions
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(texts):
            cache_key = self._cache_key(text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                embeddings[index] = cached
            else:
                missing_indices.append(index)
                missing_texts.append(text)

        for start in range(0, len(missing_texts), self.batch_size):
            batch_texts = missing_texts[start : start + self.batch_size]
            batch_indices = missing_indices[start : start + self.batch_size]
            response = self._create_embeddings_with_retry(batch_texts)
            ordered = sorted(response.data, key=lambda item: item.index)
            for source_index, item, text in zip(batch_indices, ordered, batch_texts):
                vector = item.embedding
                embeddings[source_index] = vector
                self.cache[self._cache_key(text)] = vector

        return [embedding for embedding in embeddings if embedding is not None]

    def flush_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def _cache_key(self, text: str) -> str:
        payload = f"{self.model}:{self.dimensions}:{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _create_embeddings_with_retry(self, texts: list[str]):
        delay_seconds = 1.0
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dimensions,
                )
            except Exception as error:
                last_error = error
                if attempt == self.max_retries:
                    break
                time.sleep(delay_seconds)
                delay_seconds *= 2
        raise RuntimeError(f"Embedding request failed after {self.max_retries} attempts") from last_error


class PaddleOCREngine:
    def __init__(self, lang: str = "en", text_rec_score_thresh: float = 0.7) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        self.text_rec_score_thresh = text_rec_score_thresh
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=0.0,
        )

    def extract_markdown(self, image: Image.Image, document_id: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = Path(handle.name)
            image.convert("RGB").save(handle, format="PNG")

        try:
            result = self.ocr.predict(str(temp_path))[0]
        finally:
            temp_path.unlink(missing_ok=True)

        records = self._extract_records(result)
        if not records:
            return ""
        return self._records_to_markdown(records)

    def _extract_records(self, result: dict) -> list[dict[str, float | str]]:
        records: list[dict[str, float | str]] = []
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        polys = result.get("dt_polys", [])

        for text, score, poly in zip(texts, scores, polys):
            cleaned = " ".join(str(text).split())
            if not cleaned or float(score) < self.text_rec_score_thresh:
                continue
            x_coords = [int(point[0]) for point in poly]
            y_coords = [int(point[1]) for point in poly]
            records.append(
                {
                    "text": cleaned,
                    "score": float(score),
                    "x0": float(min(x_coords)),
                    "x1": float(max(x_coords)),
                    "y0": float(min(y_coords)),
                    "y1": float(max(y_coords)),
                    "height": float(max(y_coords) - min(y_coords)),
                    "width": float(max(x_coords) - min(x_coords)),
                }
            )
        return records

    def _records_to_markdown(self, records: list[dict[str, float | str]]) -> str:
        if not records:
            return ""

        page_width = max(float(item["x1"]) for item in records)
        column_split = self._find_column_split(records, page_width)
        ordered = self._order_records(records, column_split)
        median_height = statistics.median(item["height"] for item in ordered) if ordered else 12.0
        line_gap_threshold = max(8.0, median_height * 0.7)
        paragraph_gap_threshold = max(14.0, median_height * 1.4)

        lines: list[list[dict[str, float | str]]] = []
        current_line: list[dict[str, float | str]] = []
        current_center: float | None = None

        for record in ordered:
            center = (float(record["y0"]) + float(record["y1"])) / 2
            if current_center is None or abs(center - current_center) <= line_gap_threshold:
                current_line.append(record)
                if current_center is None:
                    current_center = center
                else:
                    current_center = (current_center * (len(current_line) - 1) + center) / len(current_line)
            else:
                lines.append(sorted(current_line, key=lambda item: item["x0"]))  # type: ignore[index]
                current_line = [record]
                current_center = center
        if current_line:
            lines.append(sorted(current_line, key=lambda item: item["x0"]))  # type: ignore[index]

        rendered_lines: list[dict[str, float | str | bool]] = []
        for line in lines:
            text = " ".join(str(item["text"]) for item in line).strip()
            if not text:
                continue
            y0 = min(float(item["y0"]) for item in line)
            y1 = max(float(item["y1"]) for item in line)
            height = y1 - y0
            x0 = min(float(item["x0"]) for item in line)
            x1 = max(float(item["x1"]) for item in line)
            rendered_lines.append(
                {
                    "text": text,
                    "y0": y0,
                    "y1": y1,
                    "height": height,
                    "x0": x0,
                    "x1": x1,
                    "is_heading": self._is_heading_line(
                        text=text,
                        height=height,
                        median_height=median_height,
                        line_width=x1 - x0,
                        page_width=page_width,
                    ),
                }
            )

        blocks: list[str] = []
        current_block: list[str] = []
        previous_y1: float | None = None

        for line in rendered_lines:
            gap = None if previous_y1 is None else float(line["y0"]) - previous_y1
            text = str(line["text"])
            if bool(line["is_heading"]):
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = []
                blocks.append(f"# {text}")
            else:
                if gap is not None and gap > paragraph_gap_threshold and current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = [text]
                else:
                    current_block.append(text)
            previous_y1 = float(line["y1"])

        if current_block:
            blocks.append("\n".join(current_block).strip())

        markdown = "\n\n".join(block for block in blocks if block)
        markdown = markdown.replace("\r\n", "\n")
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        return markdown.strip()

    def _find_column_split(
        self,
        records: list[dict[str, float | str]],
        page_width: float,
    ) -> float | None:
        x_starts = sorted(float(item["x0"]) for item in records)
        if len(x_starts) < 10:
            return None
        largest_gap = 0.0
        split = None
        for left, right in zip(x_starts, x_starts[1:]):
            gap = right - left
            if gap > largest_gap:
                largest_gap = gap
                split = (left + right) / 2
        if largest_gap >= page_width * 0.18:
            return split
        return None

    def _order_records(
        self,
        records: list[dict[str, float | str]],
        column_split: float | None,
    ) -> list[dict[str, float | str]]:
        if column_split is None:
            return sorted(records, key=lambda item: (item["y0"], item["x0"]))  # type: ignore[index]

        def key(item: dict[str, float | str]) -> tuple[int, float, float]:
            column = 0 if float(item["x0"]) < column_split else 1
            return (column, float(item["y0"]), float(item["x0"]))

        return sorted(records, key=key)

    def _is_heading_line(
        self,
        text: str,
        height: float,
        median_height: float,
        line_width: float,
        page_width: float,
    ) -> bool:
        stripped = text.strip()
        if len(stripped) > 100:
            return False
        if stripped.endswith((".", "?", "!")):
            return False
        words = stripped.split()
        if not words:
            return False
        letters = [char for char in stripped if char.isalpha()]
        if not letters:
            return False
        digit_ratio = sum(char.isdigit() for char in stripped) / max(len(stripped), 1)
        uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
        title_case_ratio = sum(word[:1].isupper() for word in words if word) / len(words)
        if digit_ratio > 0.15:
            return False
        if line_width > page_width * 0.9:
            return False
        return height >= median_height * 1.2 or uppercase_ratio >= 0.7 or title_case_ratio >= 0.8


class HierarchicalChunker:
    def __init__(
        self,
        token_model: str = "text-embedding-3-large",
        max_chars: int = 1200,
        overlap_chars: int = 160,
        min_leaf_chars: int = 250,
    ) -> None:
        self.token_model = token_model
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_leaf_chars = min_leaf_chars
        self.encoding = self._load_encoding(token_model)

    def chunk_page(
        self,
        document_id: str,
        page_number: int,
        text: str,
        page_size: tuple[int, int],
    ) -> list[Chunk]:
        sections = self._parse_sections(text, page_number)
        chunks: list[Chunk] = []
        chunk_index = 1
        page_width, page_height = page_size

        for section_title, section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue

            section_chunk_id = f"{document_id}-section-{chunk_index:03d}"
            section_chunk = Chunk(
                document_id=document_id,
                chunk_id=section_chunk_id,
                chunk_index=chunk_index,
                page_number=page_number,
                chunk_level="section",
                section_path=section_title,
                parent_chunk_id=None,
                text=section_text,
                preview=section_text[:120].replace("\n", " "),
                character_count=len(section_text),
                token_estimate=self._estimate_tokens(section_text),
                page_width=page_width,
                page_height=page_height,
            )
            chunks.append(section_chunk)
            chunk_index += 1

            for leaf_number, leaf_text in enumerate(self._split_leaf_chunks(section_text), start=1):
                leaf_chunk = Chunk(
                    document_id=document_id,
                    chunk_id=f"{section_chunk_id}-leaf-{leaf_number:03d}",
                    chunk_index=chunk_index,
                    page_number=page_number,
                    chunk_level="leaf",
                    section_path=section_title,
                    parent_chunk_id=section_chunk_id,
                    text=leaf_text,
                    preview=leaf_text[:120].replace("\n", " "),
                    character_count=len(leaf_text),
                    token_estimate=self._estimate_tokens(leaf_text),
                    page_width=page_width,
                    page_height=page_height,
                )
                chunks.append(leaf_chunk)
                chunk_index += 1

        return chunks

    def _parse_sections(self, text: str, page_number: int) -> list[tuple[str, str]]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        sections: list[tuple[str, str]] = []
        current_title = f"Page {page_number}"
        current_blocks: list[str] = []

        for block in blocks:
            if self._is_heading(block):
                if current_blocks:
                    sections.append((current_title, "\n\n".join(current_blocks).strip()))
                    current_blocks = []
                current_title = self._clean_heading(block)
            else:
                current_blocks.append(block)

        if current_blocks:
            sections.append((current_title, "\n\n".join(current_blocks).strip()))

        if not sections and text.strip():
            sections.append((f"Page {page_number}", text.strip()))
        return sections

    def _is_heading(self, block: str) -> bool:
        stripped = block.strip()
        if stripped.startswith("#"):
            return True
        if "\n" in stripped:
            return False
        if len(stripped) > 100:
            return False
        if stripped.endswith((".", "?", "!")):
            return False
        letters = [char for char in stripped if char.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
        title_case_words = sum(word[:1].isupper() for word in stripped.split() if word)
        title_case_ratio = title_case_words / max(len(stripped.split()), 1)
        return uppercase_ratio >= 0.7 or title_case_ratio >= 0.8

    def _clean_heading(self, block: str) -> str:
        return re.sub(r"^#+\s*", "", block).strip()

    def _split_leaf_chunks(self, text: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        if not paragraphs:
            return []

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= self.max_chars:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
            if len(paragraph) <= self.max_chars:
                current = paragraph
            else:
                oversized = self._split_oversized_paragraph(paragraph)
                chunks.extend(oversized[:-1])
                current = oversized[-1]

        if current:
            chunks.append(current.strip())

        merged: list[str] = []
        for chunk in chunks:
            if merged and len(chunk) < self.min_leaf_chars:
                merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
            else:
                merged.append(chunk)
        return merged

    def _split_oversized_paragraph(self, paragraph: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        parts: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= self.max_chars:
                current = candidate
                continue

            if current:
                parts.append(current.strip())
                overlap = current[-self.overlap_chars :].strip()
                current = f"{overlap} {sentence}".strip() if overlap else sentence
            else:
                parts.append(sentence[: self.max_chars].strip())
                current = sentence[self.max_chars :].strip()

        if current:
            parts.append(current.strip())
        return [part for part in parts if part]

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(self.encoding.encode(text)))

    def _load_encoding(self, model_name: str):
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


class DocumentIndexer:
    def __init__(
        self,
        zip_path: Path,
        output_dir: Path,
        ocr_lang: str = "en",
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 1536,
        collection_name: str = "document_chunks_hierarchical_paddle_openai",
    ) -> None:
        self.zip_path = zip_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunker = HierarchicalChunker(token_model=embedding_model)
        self.ocr = PaddleOCREngine(lang=ocr_lang)
        self.embedder = TextEmbedder(
            model=embedding_model,
            dimensions=embedding_dimensions,
            cache_path=self.output_dir / "embedding_cache.json",
        )
        self.collection_name = collection_name
        self.section_collection_name = f"{collection_name}_sections"
        self.leaf_collection_name = f"{collection_name}_leafs"
        self.client = chromadb.PersistentClient(path=str(output_dir / "chroma"))
        self.section_collection = self.client.get_or_create_collection(
            name=self.section_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.leaf_collection = self.client.get_or_create_collection(
            name=self.leaf_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build(self) -> dict[str, object]:
        records: list[dict[str, object]] = []
        retrieval_times_ms: list[float] = []
        errors: list[dict[str, object]] = []

        with ZipFile(self.zip_path) as archive:
            image_names = sorted(name for name in archive.namelist() if name.lower().endswith(".jpg"))
            for image_name in image_names:
                document_id = Path(image_name).stem
                try:
                    page_number = self._infer_page_number(document_id)
                    image = Image.open(BytesIO(archive.read(image_name))).convert("RGB")
                    page_markdown = self.ocr.extract_markdown(image=image, document_id=document_id)
                    chunks = self.chunker.chunk_page(
                        document_id=document_id,
                        page_number=page_number,
                        text=page_markdown,
                        page_size=image.size,
                    )
                    chunks = [
                        replace(
                            chunk,
                            ocr_engine="paddleocr",
                            embedding_model=self.embedder.model,
                        )
                        for chunk in chunks
                    ]
                    if not chunks:
                        errors.append(
                            {
                                "document_id": document_id,
                                "stage": "chunking",
                                "error": "No chunks generated from OCR output.",
                            }
                        )
                        continue

                    texts = [chunk.document_text() for chunk in chunks]
                    chunk_embeddings = self.embedder.embed(texts)

                    section_records = self._prepare_collection_payload(
                        chunks=chunks,
                        embeddings=chunk_embeddings,
                        level="section",
                    )
                    leaf_records = self._prepare_collection_payload(
                        chunks=chunks,
                        embeddings=chunk_embeddings,
                        level="leaf",
                    )

                    if leaf_records["embeddings"]:
                        start = time.perf_counter()
                        self.leaf_collection.query(
                            query_embeddings=[leaf_records["embeddings"][0]],
                            n_results=min(3, max(self.leaf_collection.count(), 1)),
                        )
                        retrieval_times_ms.append((time.perf_counter() - start) * 1000)

                    if section_records["ids"]:
                        self.section_collection.upsert(
                            ids=section_records["ids"],
                            embeddings=section_records["embeddings"],
                            metadatas=section_records["metadatas"],
                            documents=section_records["documents"],
                        )
                    if leaf_records["ids"]:
                        self.leaf_collection.upsert(
                            ids=leaf_records["ids"],
                            embeddings=leaf_records["embeddings"],
                            metadatas=leaf_records["metadatas"],
                            documents=leaf_records["documents"],
                        )

                    for chunk in chunks:
                        records.append(
                            {
                                "document_id": chunk.document_id,
                                "chunk_id": chunk.chunk_id,
                                "chunk_index": chunk.chunk_index,
                                "page_number": chunk.page_number,
                                "chunk_level": chunk.chunk_level,
                                "section_path": chunk.section_path,
                                "parent_chunk_id": chunk.parent_chunk_id,
                                "preview": chunk.preview,
                                "text": chunk.text,
                                "character_count": chunk.character_count,
                                "token_estimate": chunk.token_estimate,
                                "page_width": chunk.page_width,
                                "page_height": chunk.page_height,
                                "schema_version": chunk.schema_version,
                                "ocr_engine": chunk.ocr_engine,
                                "embedding_model": chunk.embedding_model,
                            }
                        )
                except Exception as error:
                    errors.append(
                        {
                            "document_id": document_id,
                            "stage": "document_processing",
                            "error": str(error),
                        }
                    )
                    continue

        self.embedder.flush_cache()
        summary = self._build_summary(records, retrieval_times_ms)
        summary["processing_errors"] = len(errors)
        summary["embedding_cache_entries"] = len(self.embedder.cache)
        (self.output_dir / "chunk_manifest.json").write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )
        (self.output_dir / "evaluation.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        if errors:
            error_lines = "\n".join(json.dumps(error) for error in errors)
            (self.output_dir / "processing_errors.jsonl").write_text(error_lines, encoding="utf-8")
        return summary

    def _prepare_collection_payload(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        level: str,
    ) -> dict[str, list]:
        payload = {
            "ids": [],
            "embeddings": [],
            "metadatas": [],
            "documents": [],
        }
        for chunk, embedding in zip(chunks, embeddings):
            if chunk.chunk_level != level:
                continue
            payload["ids"].append(chunk.chunk_id)
            payload["embeddings"].append(embedding)
            payload["metadatas"].append(chunk.metadata())
            payload["documents"].append(chunk.document_text())
        return payload

    def _build_summary(
        self,
        records: list[dict[str, object]],
        retrieval_times_ms: list[float],
    ) -> dict[str, object]:
        chunk_counts: dict[str, int] = {}
        level_counts: dict[str, int] = {}
        text_lengths = [int(record["character_count"]) for record in records]
        token_estimates = [int(record["token_estimate"]) for record in records]

        for record in records:
            document_id = str(record["document_id"])
            chunk_counts[document_id] = chunk_counts.get(document_id, 0) + 1
            chunk_level = str(record["chunk_level"])
            level_counts[chunk_level] = level_counts.get(chunk_level, 0) + 1

        return {
            "zip_path": str(self.zip_path),
            "collection_name": self.collection_name,
            "section_collection_name": self.section_collection_name,
            "leaf_collection_name": self.leaf_collection_name,
            "documents_processed": len(chunk_counts),
            "chunks_created": len(records),
            "average_chunks_per_document": round(len(records) / max(len(chunk_counts), 1), 2),
            "median_chunks_per_document": statistics.median(chunk_counts.values()) if chunk_counts else 0,
            "chunk_level_distribution": level_counts,
            "average_characters_per_chunk": round(statistics.mean(text_lengths), 2) if text_lengths else 0.0,
            "median_characters_per_chunk": statistics.median(text_lengths) if text_lengths else 0,
            "average_estimated_tokens_per_chunk": round(statistics.mean(token_estimates), 2)
            if token_estimates
            else 0.0,
            "retrieval_latency_ms_avg": round(statistics.mean(retrieval_times_ms), 3)
            if retrieval_times_ms
            else 0.0,
            "retrieval_latency_ms_p95": round(self._percentile(retrieval_times_ms, 95), 3)
            if retrieval_times_ms
            else 0.0,
            "storage_path": str(self.output_dir / "chroma"),
        }

    def _infer_page_number(self, document_id: str) -> int:
        tail = document_id.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 1

    def _percentile(self, values: list[float], percentile: int) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        rank = (len(ordered) - 1) * (percentile / 100)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        weight = rank - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight


class SparseKeywordIndex:
    def __init__(
        self,
        manifest: list[dict[str, object]],
        chunk_level: str = "leaf",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.records_by_id: dict[str, dict[str, object]] = {}
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.doc_freq: Counter[str] = Counter()
        self.doc_lengths: dict[str, int] = {}

        filtered_records = [
            record for record in manifest if str(record.get("chunk_level", "")) == chunk_level
        ]
        for record in filtered_records:
            chunk_id = str(record["chunk_id"])
            tokens = self._tokenize(str(record.get("text", "")))
            if not tokens:
                continue

            term_counts = Counter(tokens)
            self.records_by_id[chunk_id] = record
            self.doc_lengths[chunk_id] = len(tokens)

            for term, frequency in term_counts.items():
                self.postings[term].append((chunk_id, frequency))
            self.doc_freq.update(term_counts.keys())

        self.corpus_size = len(self.records_by_id)
        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / self.corpus_size if self.corpus_size else 0.0
        )

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        if not self.corpus_size:
            return []

        scores: defaultdict[str, float] = defaultdict(float)
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue

            document_frequency = self.doc_freq[term]
            inverse_document_frequency = math.log(
                1 + (self.corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for chunk_id, frequency in postings:
                document_length = self.doc_lengths[chunk_id]
                normalization = self.k1 * (
                    1 - self.b + self.b * document_length / max(self.avg_doc_length, 1.0)
                )
                score = inverse_document_frequency * (
                    frequency * (self.k1 + 1) / (frequency + normalization)
                )
                scores[chunk_id] += score

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(token) >= 2]


class HybridChunkRetriever:
    def __init__(
        self,
        output_dir: Path,
        collection_name: str,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 1536,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> None:
        self.output_dir = output_dir
        self.collection_name = collection_name
        self.section_collection_name = f"{collection_name}_sections"
        self.leaf_collection_name = f"{collection_name}_leafs"
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

        manifest_path = self.output_dir / "chunk_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"Chunk manifest not found at {manifest_path}. Run the build mode first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.keyword_index = SparseKeywordIndex(manifest=manifest, chunk_level="leaf")
        self.section_records = {
            str(record["chunk_id"]): record
            for record in manifest
            if str(record.get("chunk_level", "")) == "section"
        }

        self.embedder = TextEmbedder(
            model=embedding_model,
            dimensions=embedding_dimensions,
            cache_path=self.output_dir / "query_embedding_cache.json",
        )
        self.client = chromadb.PersistentClient(path=str(self.output_dir / "chroma"))
        try:
            self.leaf_collection = self.client.get_collection(self.leaf_collection_name)
        except Exception as error:
            raise RuntimeError(
                f"Leaf collection '{self.leaf_collection_name}' was not found in {self.output_dir / 'chroma'}."
            ) from error

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        vector_candidates: int = 12,
        keyword_candidates: int = 12,
        rerank_top_n: int = 8,
    ) -> list[SearchResult]:
        vector_limit = max(top_k, vector_candidates, rerank_top_n)
        keyword_limit = max(top_k, keyword_candidates, rerank_top_n)
        vector_hits = self._vector_search(query=query, limit=vector_limit)
        keyword_hits = self.keyword_index.search(query=query, limit=keyword_limit)
        candidate_ids = self._fuse_rankings(vector_hits=vector_hits, keyword_hits=keyword_hits)

        results: list[SearchResult] = []
        vector_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(vector_hits, start=1)}
        keyword_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(keyword_hits, start=1)}
        fusion_scores = self._hybrid_scores(vector_ranks=vector_ranks, keyword_ranks=keyword_ranks)

        for chunk_id in candidate_ids[: max(top_k, rerank_top_n)]:
            record = self.keyword_index.records_by_id.get(chunk_id)
            if record is None:
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(record.get("document_id", "")),
                    section_path=str(record.get("section_path", "")),
                    text=self._format_context_text(record),
                    parent_chunk_id=str(record.get("parent_chunk_id") or "") or None,
                    vector_rank=vector_ranks.get(chunk_id),
                    keyword_rank=keyword_ranks.get(chunk_id),
                    hybrid_score=fusion_scores.get(chunk_id, 0.0),
                )
            )

        return results

    def close(self) -> None:
        self.embedder.flush_cache()

    def _vector_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        collection_size = self.leaf_collection.count()
        if collection_size == 0:
            return []

        query_embedding = self.embedder.embed([query])[0]
        result = self.leaf_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, collection_size),
            include=["distances"],
        )
        distances = result.get("distances", [[]])[0]
        chunk_ids = result.get("ids", [[]])[0]
        return [
            (str(chunk_id), 1.0 / (1.0 + float(distance)))
            for chunk_id, distance in zip(chunk_ids, distances)
        ]

    def _fuse_rankings(
        self,
        vector_hits: list[tuple[str, float]],
        keyword_hits: list[tuple[str, float]],
    ) -> list[str]:
        vector_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(vector_hits, start=1)}
        keyword_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(keyword_hits, start=1)}
        fused_scores = self._hybrid_scores(vector_ranks=vector_ranks, keyword_ranks=keyword_ranks)
        return [
            chunk_id
            for chunk_id, _ in sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

    def _hybrid_scores(
        self,
        vector_ranks: dict[str, int],
        keyword_ranks: dict[str, int],
    ) -> dict[str, float]:
        scores: defaultdict[str, float] = defaultdict(float)
        for chunk_id, rank in vector_ranks.items():
            scores[chunk_id] += self.vector_weight / (self.rrf_k + rank)
        for chunk_id, rank in keyword_ranks.items():
            scores[chunk_id] += self.keyword_weight / (self.rrf_k + rank)
        return dict(scores)

    def _format_context_text(self, record: dict[str, object]) -> str:
        section_path = str(record.get("section_path", "")).strip()
        document_id = str(record.get("document_id", "")).strip()
        text = str(record.get("text", "")).strip()
        fragments = [
            f"Document: {document_id}",
            f"Section: {section_path or 'Unknown'}",
            text,
        ]
        return "\n".join(fragment for fragment in fragments if fragment)


class LLMReranker:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        max_retries: int = 2,
    ) -> None:
        self.client = OpenAI(api_key=require_openai_api_key())
        self.model = model
        self.max_retries = max_retries

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        rerank_window = min(top_n, len(candidates))
        rerank_candidates = candidates[:rerank_window]
        prompt = {
            "query": query,
            "candidates": [
                {
                    "chunk_id": candidate.chunk_id,
                    "document_id": candidate.document_id,
                    "section_path": candidate.section_path,
                    "text": candidate.text[:2000],
                }
                for candidate in rerank_candidates
            ],
        }
        response_content = self._request_scores(prompt)
        if response_content is None:
            return candidates

        scored_results = self._parse_scores(
            content=response_content,
            candidates=rerank_candidates,
        )
        if not scored_results:
            return candidates

        tail = candidates[rerank_window:]
        return scored_results + tail

    def _request_scores(self, prompt: dict[str, object]) -> str | None:
        delay_seconds = 1.0
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You rerank retrieval candidates for RAG. "
                                "Return strict JSON with a top-level 'results' array. "
                                "Each item must include 'chunk_id', 'score' (0-100), and 'reason'."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Score each candidate by how directly it helps answer the query. "
                                "Favor exact answer-bearing passages over broad background. "
                                "Use only the provided chunk ids.\n\n"
                                f"{json.dumps(prompt, ensure_ascii=True)}"
                            ),
                        },
                    ],
                )
                return response.choices[0].message.content
            except Exception as error:
                last_error = error
                if attempt == self.max_retries:
                    break
                time.sleep(delay_seconds)
                delay_seconds *= 2
        if last_error is not None:
            return None
        return None

    def _parse_scores(
        self,
        content: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return []

        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        results: list[SearchResult] = []
        for item in payload.get("results", []):
            chunk_id = str(item.get("chunk_id", ""))
            candidate = by_id.get(chunk_id)
            if candidate is None:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            results.append(replace(candidate, rerank_score=score))

        if not results:
            return []

        untouched = [
            replace(candidate, rerank_score=0.0)
            for candidate in candidates
            if candidate.chunk_id not in {result.chunk_id for result in results}
        ]
        combined = results + untouched
        return sorted(
            combined,
            key=lambda item: ((item.rerank_score or 0.0), item.hybrid_score),
            reverse=True,
        )


class RAGPipeline:
    def __init__(
        self,
        output_dir: Path,
        collection_name: str,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 1536,
        answer_model: str = "gpt-4.1-mini",
        reranker_model: str = "gpt-4.1-mini",
    ) -> None:
        self.client = OpenAI(api_key=require_openai_api_key())
        self.answer_model = answer_model
        self.retriever = HybridChunkRetriever(
            output_dir=output_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )
        self.reranker = LLMReranker(model=reranker_model)

    def answer(
        self,
        query: str,
        top_k: int = 4,
        vector_candidates: int = 12,
        keyword_candidates: int = 12,
        rerank_top_n: int = 8,
    ) -> RAGResponse:
        hybrid_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            vector_candidates=vector_candidates,
            keyword_candidates=keyword_candidates,
            rerank_top_n=rerank_top_n,
        )
        reranked_results = self.reranker.rerank(
            query=query,
            candidates=hybrid_results,
            top_n=rerank_top_n,
        )
        final_results = reranked_results[:top_k]
        contexts = [result.text for result in final_results]
        answer = self._answer_query(query=query, contexts=contexts, source_ids=[result.chunk_id for result in final_results])
        self.retriever.close()
        return RAGResponse(
            query=query,
            answer=answer,
            contexts=contexts,
            retrieved_chunks=final_results,
        )

    def _answer_query(self, query: str, contexts: list[str], source_ids: list[str]) -> str:
        joined_context = "\n\n---\n\n".join(
            f"[{chunk_id}]\n{context}"
            for chunk_id, context in zip(source_ids, contexts)
        )
        response = self.client.chat.completions.create(
            model=self.answer_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer using only the retrieved context. "
                        "If the context is insufficient, say so explicitly. "
                        "When you make a factual claim, cite the supporting chunk id in square brackets."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\n"
                        f"Context:\n{joined_context[:14000]}"
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the OCR + hierarchical chunk index or run a full RAG query with "
            "hybrid retrieval and LLM reranking."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("build", "query"),
        default="build",
        help="Run ingestion/indexing or query the existing Chroma-backed RAG system.",
    )
    parser.add_argument(
        "--zip-path",
        default="examples.zip",
        type=Path,
        help="Path to the zip archive that contains the document page images.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts_paddle_openai",
        type=Path,
        help="Directory used for manifests, evaluation results, and the Chroma store.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="en",
        help="PaddleOCR language code.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model used for chunk vectors.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        default=1536,
        type=int,
        help="Embedding vector dimensions. OpenAI supports reducing the default size with the dimensions parameter.",
    )
    parser.add_argument(
        "--collection-name",
        default="document_chunks_hierarchical_paddle_openai",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Question to answer when --mode query is used.",
    )
    parser.add_argument(
        "--top-k",
        default=4,
        type=int,
        help="Number of final reranked chunks to pass to answer generation.",
    )
    parser.add_argument(
        "--vector-candidates",
        default=12,
        type=int,
        help="Number of dense vector candidates to fetch before hybrid fusion.",
    )
    parser.add_argument(
        "--keyword-candidates",
        default=12,
        type=int,
        help="Number of sparse keyword candidates to fetch before hybrid fusion.",
    )
    parser.add_argument(
        "--rerank-top-n",
        default=8,
        type=int,
        help="How many hybrid candidates to send to the LLM reranker.",
    )
    parser.add_argument(
        "--answer-model",
        default="gpt-4.1-mini",
        help="OpenAI model used to synthesize the final grounded answer.",
    )
    parser.add_argument(
        "--reranker-model",
        default="gpt-4.1-mini",
        help="OpenAI model used for candidate reranking.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "build":
        indexer = DocumentIndexer(
            zip_path=args.zip_path,
            output_dir=args.output_dir,
            ocr_lang=args.ocr_lang,
            embedding_model=args.embedding_model,
            embedding_dimensions=args.embedding_dimensions,
            collection_name=args.collection_name,
        )
        summary = indexer.build()
        print(json.dumps(summary, indent=2))
        return 0

    if not args.query:
        raise RuntimeError("--query is required when --mode query is used.")

    pipeline = RAGPipeline(
        output_dir=args.output_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        answer_model=args.answer_model,
        reranker_model=args.reranker_model,
    )
    response = pipeline.answer(
        query=args.query,
        top_k=args.top_k,
        vector_candidates=args.vector_candidates,
        keyword_candidates=args.keyword_candidates,
        rerank_top_n=args.rerank_top_n,
    )
    print(json.dumps(response.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
