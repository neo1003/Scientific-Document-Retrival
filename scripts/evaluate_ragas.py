from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithoutReference, Faithfulness

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_pipeline import RAGPipeline, require_openai_api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run no-reference RAGAS evaluation over the Chroma-backed RAG pipeline."
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
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=None,
        help="Optional JSONL or JSON file of evaluation queries. If omitted, queries are generated from the manifest.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12,
        help="Number of auto-generated evaluation queries when --queries-path is omitted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of reranked chunks to pass into answer generation.",
    )
    parser.add_argument(
        "--vector-candidates",
        type=int,
        default=12,
        help="Number of dense vector candidates to gather before hybrid fusion.",
    )
    parser.add_argument(
        "--keyword-candidates",
        type=int,
        default=12,
        help="Number of sparse keyword candidates to gather before hybrid fusion.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=8,
        help="Number of hybrid candidates sent to the LLM reranker.",
    )
    parser.add_argument(
        "--answer-model",
        default="gpt-4.1-mini",
        help="OpenAI model used to answer questions from retrieved chunks.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        help="OpenAI model used internally by RAGAS metrics.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model used for query vectors and Answer Relevancy.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        default=1536,
        type=int,
        help="Embedding dimensions for query retrieval.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional explicit path for the JSON report.",
    )
    return parser.parse_args()


def load_queries(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        queries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            queries.append(str(item["query"]).strip())
        return queries
    items = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["query"]).strip() if isinstance(item, dict) else str(item).strip() for item in items]


def choose_seed_chunks(manifest: list[dict[str, object]], sample_size: int) -> list[dict[str, object]]:
    leaf_chunks = [
        record
        for record in manifest
        if str(record.get("chunk_level", "")) == "leaf" and len(str(record.get("text", "")).strip()) >= 300
    ]
    if not leaf_chunks:
        leaf_chunks = [record for record in manifest if len(str(record.get("text", "")).strip()) >= 300]
    leaf_chunks = sorted(leaf_chunks, key=lambda item: (str(item.get("document_id", "")), int(item.get("chunk_index", 0))))
    return leaf_chunks[:sample_size]


def generate_queries(
    client: OpenAI,
    manifest: list[dict[str, object]],
    sample_size: int,
    model: str,
) -> list[str]:
    queries: list[str] = []
    for record in choose_seed_chunks(manifest, sample_size):
        source_text = str(record.get("text", "")).strip()
        prompt = (
            "Create one realistic user question that can be answered using only the following document chunk.\n"
            "The question should sound natural, avoid quoting the chunk verbatim, and focus on the chunk's main topic.\n"
            "Return only the question.\n\n"
            f"Chunk:\n{source_text[:4000]}"
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You generate evaluation questions for RAG systems."},
                {"role": "user", "content": prompt},
            ],
        )
        question = (response.choices[0].message.content or "").strip()
        if question:
            queries.append(question)
    return queries

def main() -> int:
    args = parse_args()
    require_openai_api_key()
    report_path = args.report_path or (args.output_dir / "ragas_evaluation.json")
    manifest_path = args.output_dir / "chunk_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if args.queries_path:
        queries = load_queries(args.queries_path)
    else:
        queries = generate_queries(
            client=client,
            manifest=manifest,
            sample_size=args.sample_size,
            model=args.answer_model,
        )

    pipeline = RAGPipeline(
        output_dir=args.output_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        answer_model=args.answer_model,
        reranker_model=args.judge_model,
    )

    samples: list[dict[str, object]] = []
    per_query: list[dict[str, object]] = []

    for query in queries:
        response = pipeline.answer(
            query=query,
            top_k=args.top_k,
            vector_candidates=args.vector_candidates,
            keyword_candidates=args.keyword_candidates,
            rerank_top_n=args.rerank_top_n,
        )
        contexts = response.contexts
        chunk_ids = [chunk.chunk_id for chunk in response.retrieved_chunks]
        answer = response.answer
        samples.append(
            {
                "user_input": query,
                "retrieved_contexts": contexts,
                "response": answer,
            }
        )
        per_query.append(
            {
                "query": query,
                "response": answer,
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_contexts": contexts,
            }
        )

    dataset = EvaluationDataset.from_list(samples)
    ragas_llm = llm_factory(args.judge_model, provider="openai", client=client)
    ragas_embeddings = RagasOpenAIEmbeddings(client=client, model=args.embedding_model)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecisionWithoutReference(llm=ragas_llm),
    ]
    result = evaluate(dataset=dataset, metrics=metrics, show_progress=True)
    frame = result.to_pandas()

    metric_columns = ["faithfulness", "answer_relevancy", "context_precision_without_reference"]
    summary = {
        "queries_evaluated": len(per_query),
        "collection_name": args.collection_name,
        "top_k": args.top_k,
        "vector_candidates": args.vector_candidates,
        "keyword_candidates": args.keyword_candidates,
        "rerank_top_n": args.rerank_top_n,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "embedding_model": args.embedding_model,
    }
    for column in metric_columns:
        if column in frame.columns:
            summary[column] = round(float(pd.to_numeric(frame[column], errors="coerce").mean()), 4)

    report = {
        "summary": summary,
        "per_query": [
            {
                **query_row,
                **{
                    column: (
                        None
                        if pd.isna(frame.iloc[index][column])
                        else round(float(frame.iloc[index][column]), 4)
                    )
                    for column in metric_columns
                    if column in frame.columns
                },
            }
            for index, query_row in enumerate(per_query)
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
