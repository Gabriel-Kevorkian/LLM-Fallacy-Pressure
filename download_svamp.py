# download_svamp.py
# Run: pip install datasets
# Then: python download_svamp.py

from datasets import load_dataset
import json
from pathlib import Path

# ── Download ──────────────────────────────────────────────────────────────────
print("Downloading SVAMP from HuggingFace...")
ds = load_dataset("ChilleD/SVAMP")

# SVAMP fields:
#   ID, Body, Question, Formula, Answer (int), Type
# We merge Body + Question into a single question string,
# exactly like GSM8K's natural language format.

def convert(record):
    body     = record.get("Body", "").strip()
    question = record.get("Question", "").strip()
    # Join body and question — Body is the story, Question is the final ask
    full_q   = f"{body} {question}".strip()
    answer   = str(record["Answer"])            # keep as string, gsm8k.py parses it
    q_type   = record.get("Type", "Unknown")    # e.g. "Addition", "Multiplication"
    return {
        "question":       full_q,
        "answer":         f"#### {answer}",     # GSM8K answer format
        "correct_answer": answer,               # pre-parsed for gsm8k.py loader
        "difficulty":     q_type,               # gsm8k.py uses this as difficulty tag
        "id":             str(record.get("ID", "")),
    }

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path("data/svamp")
out_dir.mkdir(parents=True, exist_ok=True)

for split_name, split_key in [("train", "train"), ("test", "test")]:
    if split_key not in ds:
        continue
    out_path = out_dir / f"{split_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for record in ds[split_key]:
            f.write(json.dumps(convert(record), ensure_ascii=False) + "\n")
    print(f"Saved {len(ds[split_key])} records → {out_path}")

print("\nDone. Run your experiment with:")
print("  python gsm8k.py --file data/svamp/test.jsonl --limit 50 --max_turns 5 --workers 20")