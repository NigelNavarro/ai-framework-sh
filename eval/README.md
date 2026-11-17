# Eval

Measures how well retrieval and agent answers work against the real, ingested knowledge base. Unlike `tests/`, which is hermetic (stub models, temporary stores, no live services), everything here runs against a live Ollama and the actual `data/chroma` collection, so results depend on the embed model, the corpus, and the configured `relevance_threshold`.

## Files

| File               | Purpose                                                                                  |
| ----------------   | ---------------------------------------------------------------------------------------- |
| `golden_qa.json`   | The golden set: fixed questions with known expected outcomes.                            |
| `run_eval.py`      | Runs every golden question through retrieval and prints a score.                         |
| `run_grounding.py` | Runs the agent on every golden question and grades each answer with the grounding judge. |

## The golden set

Each entry has three fields:

| Field             | Meaning                                                                       |
| ----------------- | ----------------------------------------------------------------------------- |
| `question`        | The question, phrased as a user would ask it.                                 |
| `expected_route`  | `"knowledge"`: the answer is in the corpus. `"boundary"`: it is not.          |
| `expected_source` | For `knowledge`, the filename that should rank first. For `boundary`, `null`. |

```json
{
  "question": "Who is Queequeg?",
  "expected_route": "knowledge",
  "expected_source": "mobydick.txt"
}
```

Sources are matched by filename only, so `tests/books/mobydick.txt` in the store matches `mobydick.txt` here.

## Running it

Prerequisites: Ollama is running (`curl -s http://localhost:11434/api/version`) and the corpus is ingested (`.venv/bin/python apps/cli/main.py ingest tests/books/`).

From the repo root:

```console
.venv/bin/python eval/run_eval.py                      # use the configured threshold
.venv/bin/python eval/run_eval.py --threshold 0.65     # try another threshold for one run
.venv/bin/python eval/run_eval.py --golden other.json  # use a different golden set
```

The script only Reads. It embeds questions and queries the store, but never Writes to it.

## Reading the output

One line per question, then two scores:

```console
PASS  knowledge  top=0.655  kept=5  mobydick.txt            Who is Queequeg?
PASS  boundary   top=0.450  kept=0  mobydick.txt            How do I change a flat tire on a car?

Source hit rate:    9/9 (100%)
Boundary accuracy:  5/5 (100%)
```

- `top` is the best chunk's cosine similarity, shown even when that chunk was filtered out.
- `kept` is how many chunks met the threshold (what `ask` would answer from).
- The source column is the best chunk's file. For a boundary question it shows the closest file,
  even though nothing was kept.
- **Source hit rate:** knowledge questions whose top kept chunk came from the expected file.
- **Boundary accuracy:** boundary questions where nothing met the threshold.

The exit code is 0 only when every question passes.

A failure usually points one of two ways:

- A boundary question with `top` just above the threshold: the threshold is too low.
- A knowledge question with the wrong file at a high `top`: retrieval found a similar passage in
  another book.

## Calibrating the threshold

`relevance_threshold` is specific to the embed model. Use `--threshold` to compare values: the right value sits between the highest boundary score and the lowest knowledge score.

Keep the golden set honest: questions used to choose a threshold can't also prove it works. When the threshold changes, add fresh questions before trusting the score again.

Re-calibrate whenever `embed_model` changes (which also requires a re-index).

## Grading full answers (run_grounding.py)

`run_eval.py` checks retrieval only. `run_grounding.py` checks what the agent does with what it retrieves: for every
golden question it runs the full agent, labels the answer's provenance, and grades it with the grounding judge
(`framework/agent/grounding.py`).

```console
.venv/bin/python eval/run_grounding.py                  # grade all 14 golden questions (a few minutes)
.venv/bin/python eval/run_grounding.py --show-answers   # also print each answer, to investigate a FAIL
```

It is read-only: only the `search_knowledge` tool is registered (no memory writes, no drafts) and nothing is traced.

Each line shows PASS/FAIL, the score, the expected route, the three checks, the provenance label, and a `declined`
flag when the answer said it couldn't answer:

```console
PASS  1.00  knowledge  route=ok  citations_exist=ok  cites_expected=ok  [documents]  Who is Queequeg?
FAIL  0.67  knowledge  route=ok  citations_exist=ok  cites_expected=FAIL  [documents]  declined  Where does Victor Frankenstein go to university?
```

- `route`: knowledge questions should retrieve documents; boundary questions should not.
- `citations_exist`: every `[n]` in the answer points to a chunk that was actually retrieved.
- `cites_expected`: a knowledge answer cites the expected book.
- `-` means the check doesn't apply (for example, `cites_expected` on a boundary question).
- `declined` is a heuristic and never changes the score.

The score is the fraction of applicable checks that passed. Answers vary from run to run, so compare totals over several runs before drawing conclusions. A FAIL with `route=ok` and `declined` usually means retrieval returned the right book but not the passage with the answer: a retrieval miss, not a grounding failure.
