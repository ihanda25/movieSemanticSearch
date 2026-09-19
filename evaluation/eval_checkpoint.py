"""Run any cross-encoder checkpoint through evaluate.py's exact scoring logic.

Run: .venv/bin/python evaluation/eval_checkpoint.py --model models/cross-encoder-finetuned-v1/final

Exists for the dev-set regression check in the fine-tuning plan (README.md
"Train separately, evaluate, then decide whether to deploy"): a fine-tuned
checkpoint is judged on evaluation/splits.json's held-out test movies (see
finetune_cross_encoder.py's own before/after numbers for that), but it must
ALSO not regress on the original hand-written queries.json dev set, which this
whole synthetic pipeline never touches. Reuses evaluate.py's evaluate()/
print_summary() rather than re-implementing them, so the numbers are computed
identically to the production baseline everyone already trusts.
"""
import argparse
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import disney_overview_search.disney_cross_encode as dce  # noqa: E402
import evaluate as evalmod  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="checkpoint dir or HF model name")
    parser.add_argument("--queries", type=Path, default=evalmod.QUERY_FILE)
    parser.add_argument("--candidate-k", type=int, default=dce.CANDIDATE_K)
    args = parser.parse_args()

    print(f"Loading {args.model} in place of the production cross-encoder...")
    dce._cross_encoder = CrossEncoder(args.model)

    import json
    rows = json.loads(args.queries.read_text())["queries"]
    print(f"Scoring {len(rows)} queries from {args.queries.name}...\n")
    results = evalmod.evaluate(rows, args.candidate_k)
    evalmod.print_summary(results, args.candidate_k)


if __name__ == "__main__":
    main()
