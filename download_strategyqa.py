#!/usr/bin/env python3
"""
Download StrategyQA and export to JSONL for your log-odds harness.

Recommended HF dataset (works cleanly):
  ChilleD/StrategyQA

Fields (observed): qid, term, description, question, answer (bool), facts
Splits: train, test
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset


def _facts_to_str(facts: Any) -> str:
    if facts is None:
        return ""
    if isinstance(facts, str):
        return facts.strip()
    if isinstance(facts, (list, tuple)):
        # Join list of facts if the dataset stores it that way
        return "\n".join(str(x).strip() for x in facts if str(x).strip()).strip()
    return str(facts).strip()


def to_record(ex: Dict[str, Any], *, uid: str, split: str, include_facts: bool) -> Dict[str, Any]:
    question = (ex.get("question") or "").strip()
    answer_bool = bool(ex.get("answer"))
    answer_01 = 1 if answer_bool else 0

    qid: Optional[str] = ex.get("qid")
    term: str = (ex.get("term") or "").strip()
    description: str = (ex.get("description") or "").strip()
    facts_str: str = _facts_to_str(ex.get("facts"))

    question_01 = f"{question} (0=no, 1=yes)"

    if include_facts and facts_str:
        prompt_with_context = (
            f"Facts:\n{facts_str}\n\n"
            f"Question:\n{question_01}\n\n"
            f"Answer (0 or 1):"
        )
    else:
        prompt_with_context = (
            f"Question:\n{question_01}\n\n"
            f"Answer (0 or 1):"
        )

    return {
        "id": uid,
        "source": "strategyqa",
        "split": split,
        "qid": qid,
        "term": term,
        "description": description,
        "question": question,
        "facts": facts_str,
        "answer_bool": answer_bool,
        "answer_01": answer_01,
        "question_01": question_01,
        "prompt_with_context": prompt_with_context,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data", help="Output directory")
    ap.add_argument(
        "--dataset_id",
        type=str,
        default="ChilleD/StrategyQA",
        help='HF dataset id (default: "ChilleD/StrategyQA")',
    )
    ap.add_argument("--splits", type=str, default="train,test", help="Comma-separated splits")
    ap.add_argument("--max_examples", type=int, default=0, help="0 = all")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before truncating")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    ap.add_argument(
        "--include_facts",
        action="store_true",
        help="If set, include the dataset facts inside prompt_with_context",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_id)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in ds:
            raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")

        d = ds[split]
        if args.shuffle:
            d = d.shuffle(seed=args.seed)

        if args.max_examples and args.max_examples > 0:
            d = d.select(range(min(args.max_examples, len(d))))

        out_path = out_dir / f"strategyqa_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for i, ex in enumerate(d):
                uid = f"strategyqa-{split}-{i:06d}"
                rec = to_record(ex, uid=uid, split=split, include_facts=args.include_facts)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[StrategyQA] Wrote {len(d)} examples -> {out_path}")


if __name__ == "__main__":
    main()