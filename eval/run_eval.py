"""
Author: Nigel J. Navarro
Retrieval Evaluation: Score the Golden Q&A Set against the live knowledge base.

For every question in eval/golden_qa.json, this runs the same retrieval the `ask` command uses and grades the result by the question's expected_route:
  - "knowledge": PASS if retrieval returns chunks and the top chunk comes from expected_source.
  - "boundary":  PASS if retrieval returns nothing (no chunk met the relevance threshold).

It prints one line per question, then two scores: the source hit rate (knowledge questions) and the boundary accuracy (boundary questions).
Each line also shows the top chunk's raw score, so a failure shows why it failed (for example, a boundary question scoring just above the threshold).

This is NOT a pytest test. It needs a running Ollama and an ingested knowledge base, and its results depend on the embed model and the threshold.
The hermetic tests live in tests/.

Run (from the repo root):
    .venv/bin/python eval/run_eval.py
    .venv/bin/python eval/run_eval.py --threshold 0.6     # try a different threshold
    .venv/bin/python eval/run_eval.py --golden other.json # use a different golden set
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from framework.config import load_config
from framework.models.ollama import OllamaModelClient
from framework.knowledge.vectorstore import ChromaVectorStore
from framework.knowledge.retrieve import Retriever

# Default - The Golden Set sits next to this script in the eval/ directory
DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_qa.json"

def filename(source: str) -> str:
    """
    Reduce a stored source path to its filename (ex. 'tests/books/mobydick.txt' to 'mobydick.txt')
    """
    return source.split("/")[-1]

def grade(entry: dict, retriever: Retriever) -> dict:
    """
    Run one golden question through retrieval and grade it against its expected route.

    retrieve() is the system under test: it is exactly what `ask` calls, threshold included.
    retrieve_with_scores() is called only for diagnostics, to report the top chunk's raw score even when retrieve() filtered everything out
    """

    question = entry["question"]
    route = entry["expected_route"]
    expected = entry["expected_source"]

    kept = retriever.retrieve(question)                 # Filtered: what `ask` would answer from.
    scored = retriever.retrieve_with_scores(question)   # Unfiltered: the raw evidence/data
    top_chunk, top_score = scored[0] if scored else (None, None)
    top_source = filename(top_chunk.source) if top_chunk else "-"

    if route == "knowledge":
        passed = bool(kept) and filename(kept[0].source) == expected
    elif route == "boundary":
        passed = not kept
    else:
        raise ValueError(f"Unknown expected_route {route!r} for question: {question!r}")

    return {
        "question": question,
        "route": route,
        "passed": passed,
        "kept": len(kept),
        "top_source": top_source,
        "top_score": top_score,
    }

def percent(hits: int, total: int) -> str:
    """
    Format a hit count as 'hits / total (%)', guarding against an empty group.
    """
    return f"{hits}/{total} ({hits / total:.0%})" if total else "0/0 (n/a)"

def main() -> int:
    parser = argparse.ArgumentParser(description="Score the golden Q&A set against the knowledge base.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Path to the golden set JSON")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override relevance_threshold for this run (default: configured value).")
    args = parser.parse_args()

    config = load_config()
    threshold = config.relevance_threshold if args.threshold is None else args.threshold

    # Build it the same way cmd_ask does in main, so the evaluation measures what the user actually gets.
    client = OllamaModelClient(config)
    store = ChromaVectorStore(config)
    retriever = Retriever(client, store, config.top_k, threshold)

    golden = json.loads(args.golden.read_text())
    print(f"Golden set: {args.golden.name} ({len(golden)} questions)")
    print(f"Embed model: {config.embed_model}   top_k: {config.top_k}   threshold: {threshold}\n")

    results = [grade(entry, retriever) for entry in golden]

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        score = f"{r['top_score']:.3f}" if r["top_score"] is not None else "  -  "
        print(f"{status}  {r['route']:<9}  top={score}  kept={r['kept']}  {r['top_source']:<22}  {r['question']}")

    knowledge = [r for r in results if r["route"] == "knowledge"]
    boundary = [r for r in results if r["route"] == "boundary"]
    print()
    print(f"Source hit rate:    {percent(sum(r['passed'] for r in knowledge), len(knowledge))}")
    print(f"Boundary accuracy:  {percent(sum(r['passed'] for r in boundary), len(boundary))}")

    # Exit code 0 only when every question passed so the script can gate a check later if wanted.
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())