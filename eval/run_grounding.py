"""
Author: Nigel J. Navarro
Grounding Evaluation: Run the agent on every golden qeustion, and grade each answer with the grounding judge.

This scores full agent answers by looking into the following:
  1. The agent answers the question (real model, real loop, real knowledge base).
  2. Provenance labels where the answer's information came from (label_provenance).
  3. The grounding judge grades it against the golden set (judge_grouding): route, citations_exist, cites_expected.

It prints one line per question, then totals for each check, the number of fully passing answers, an d the mean score.

Read-only by design:
  - Only the search_knowledge tool is registered. remember and draft_document would write to memory and data/drafts, and 
    leaving out recall also keeps every citation checkable (see grounding.py).
  - No tracer is passed so nothing is written to data/traces.

This is not a pytest test, as it needs Ollama and an ingested knowledge base, and the model's answer vary from run to run,
so scores can change between runs. It makes one full agent run per question, so expect a few minutes.

Run (from the repo root):
    .venv/bin/python eval/run_grounding.py
    .venv/bin/python eval/run_grounding.py --show-answers       # also print each answer under its result line
    .venv/bin/python eval/run_grounding.py --golden other.json
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
from framework.tools.registry import ToolRegistry
from framework.tools.builtin import make_search_knowledge_tool
from framework.agent.loop import run_agent_with_record
from framework.agent.provenance import label_provenance
from framework.agent.grounding import GroundingReport, judge_grounding, citation_map_from_run

# By default the golden set sits next to this script, wherever it is run from.
DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_qa.json"

def mark(result: bool | None) -> str:
    """
    Show one check compactly: 'ok' for True, 'FAIL' for False, '-' for not applicable (None).
    """
    return "-" if result is None else ("ok" if result else "FAIL")

def tally(reports: list[GroundingReport], check: str) -> str:
    """
    Count how many reports passed one check, out of the reports where that check applied.
    """
    applicable = [report.checks[check] for report in reports if report.checks[check] is not None]
    return f"{sum(applicable)}/{len(applicable)}" if applicable else "0/0 (n/a)"

def main() -> int:
    parser = argparse.ArgumentParser(description="Grade agent answers to the golden set with the grounding judge.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Path to the golden set JSON.")
    parser.add_argument("--show-answers", action="store_true", help="Print each answer under its result line.")
    args = parser.parse_args()

    config = load_config()
    client = OllamaModelClient(config)
    retriever = Retriever(client, ChromaVectorStore(config), config.top_k, config.relevance_threshold)

    # Read-only registry: search only (see the module docstring).
    registry = ToolRegistry()
    registry.register(make_search_knowledge_tool(retriever))

    golden = json.loads(args.golden.read_text())
    print(f"Golden set: {args.golden.name} ({len(golden)} questions)")
    print(f"Model: {config.model}   threshold: {config.relevance_threshold}   max rounds: {config.max_tool_rounds}\n")

    reports: list[GroundingReport] = []
    for entry in golden:
        run = run_agent_with_record(client, registry, entry["question"], config.max_tool_rounds)    # tracer=None
        provenance = label_provenance(run)
        report = judge_grounding(run.answer, provenance, citation_map_from_run(run), 
                                 entry["expected_route"], entry["expected_source"])
        reports.append(report)

        status = "PASS" if report.passed else "FAIL"
        checks = "  ".join(f"{name}={mark(result)}" for name, result in report.checks.items())
        declined = "  declined" if report.declined else ""
        print(f"{status}  {report.score:.2f}  {entry['expected_route']:<9}  {checks}  "
              f"[{report.provenance_label}]{declined}  {entry['question']}")
        if args.show_answers:
            print(f"      answer: {run.answer.strip()[:300]!r}\n")

    passed = sum(report.passed for report in reports)
    mean = sum(report.score for report in reports) / len(reports)
    print()
    print(f"route:            {tally(reports, 'route')}")
    print(f"citations_exist:  {tally(reports, 'citations_exist')}")
    print(f"cites_expected:   {tally(reports, 'cites_expected')}")
    print(f"Fully passed:     {passed}/{len(reports)}")
    print(f"Mean score:       {mean:.2f}")

    # Exit code 0 only when every answer passed every applicable check.
    return 0 if passed == len(reports) else 1

if __name__ == "__main__":
    sys.exit(main())