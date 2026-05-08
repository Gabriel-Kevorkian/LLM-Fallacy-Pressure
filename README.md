# Fallacy Under Pressure: Benchmarking LLM Susceptibility to Adversarial Logical Fallacies

A two-agent adversarial framework that systematically measures how susceptible large language models are to logical fallacies. A **Subject** model answers questions while an **Attacker** model applies structured, escalating fallacy-based pressure to induce hallucinations or flip answers. Two datasets are tested: **StrategyQA** (yes/no factual reasoning) and **SVAMP** (math word problems).

---

## How It Works

Each question is run through **N independent conversations** — one per fallacy. In each conversation:

1. The **Subject** answers the question correctly.
2. The **Attacker** applies a single assigned fallacy across up to `max_turns` turns, escalating pressure each turn.
3. A hallucination is recorded when the Subject flips to the wrong answer.
4. The primary output is a **fallacy efficiency ranking**: which fallacy flips answers with the fewest turns and the highest success rate.

```
load_questions()
  └─ ThreadPoolExecutor (workers)
       └─ run_question()
            ├─ run_planner()          [GSM8K only — picks a plausible wrong target]
            └─ for each fallacy:
                 └─ run_question_with_fallacy()
                      ├─ Subject answers initially
                      ├─ Attacker applies fallacy (loop, up to max_turns)
                      └─ Extract & validate final answer
  └─ LiveLogger     → *.live.jsonl   (real-time, crash-safe)
  └─ EmergencySaver → checkpoint every 50 questions
  └─ compute_metrics() → *.json + *.csv
```

---

## Fallacies

### StrategyQA (Yes/No Factual Reasoning)

| Fallacy | Severity | Description |
|---------|----------|-------------|
| `AD_HOMINEM` | 0.6 | Attacks the Subject's training-data credibility with invented benchmarks |
| `APPEAL_TO_AUTHORITY` | 0.7 | Invents fake experts, journals, and institutions supporting the wrong answer |
| `FALSE_INFORMATION` | 0.7 | Fabricates specific facts, dates, archive references, and statistics |
| `SLIPPERY_SLOPE` | 0.5 | Chains the correct answer to increasingly absurd consequences |
| `BANDWAGON` | 0.4 | Claims overwhelming (vague) expert consensus for the wrong answer |
| `STRAW_MAN` | 0.6 | Misrepresents the correct answer as an extreme or untenable position |
| `PURE_PRESSURE` | 0.0 | Baseline: confident assertion only — no argument, no evidence |

### GSM8K / SVAMP (Math Word Problems)

| Fallacy | Severity | Description |
|---------|----------|-------------|
| `AD_HOMINEM` | 0.6 | Raises concerns about the solver's computational reliability via invented benchmarks |
| `APPEAL_TO_AUTHORITY` | 0.7 | Invents fake mathematical standards (e.g., "IEEE Applied Arithmetic Convention") |
| `FALSE_INFORMATION` | 0.7 | Fabricates correction factors, formula IDs, or NIST guideline references |
| `FALSE_DILEMMA` | 0.5 | Presents two wrong interpretations of a key phrase, excluding the correct reading |
| `CAUSAL_FALLACY` | 0.5 | Introduces fake domain-specific real-world constraints (aerodynamic drag, ledger rounding, etc.) |
| `CIRCULAR_REASONING` | 0.5 | Constructs a self-referential "consistency check" where the wrong answer appears to validate itself |
| `PURE_PRESSURE` | 0.0 | Baseline: confident assertion only — no argument, no evidence |

Each fallacy has explicit **escalation instructions** across turns (T1 → T2 → T3+), so pressure increases systematically if the Subject holds its ground.

---

## Temperature Policy

| Role | Standard models | Reasoning models (`o1`, `o3`, `o4`, `gpt-5*`) |
|------|----------------|----------------------------------------------|
| Subject | `temperature=1.0` (varied, naturalistic) | `reasoning_effort="low"` |
| Attacker | `temperature=0.0` (deterministic) | `reasoning_effort="low"` |

---

## Model Support

The framework auto-routes API calls based on model name:

| Model prefix | Backend |
|---|---|
| `gpt-*`, `o1`, `o3`, `o4` | OpenAI API |
| `llama`, `mistral`, `gemma`, `phi`, `qwen`, `deepseek`, … | Ollama (`OLLAMA_BASE_URL`) |
| `vllm:<name>` | vLLM (`VLLM_BASE_URL`) |
| `provider/model` (slash format) | OpenRouter (`OPENROUTER_API_KEY`) |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Gabriel-Kevorkian/fallacy-under-pressure.git
cd fallacy-under-pressure
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...

# Optional — only needed for the respective backends
OPENROUTER_API_KEY=sk-or-...
OLLAMA_BASE_URL=http://localhost:11434/v1
VLLM_BASE_URL=http://localhost:8000/v1
```

### 5. Download datasets

```bash
python download_strategyqa.py        # → data/strategyqa/
python download_svamp.py             # → data/svamp/
```

---

## Running Experiments

### StrategyQA (Yes/No)

```bash
# Quick run — 50 questions, all fallacies, default models
python strategy_qa.py

# Full options
python strategy_qa.py \
  --split test \
  --limit 700 \
  --max_turns 5 \
  --subject gpt-4o-mini \
  --attacker gpt-4o \
  --workers 20 \
  --fallacy AD_HOMINEM,APPEAL_TO_AUTHORITY

# With a local Ollama model as subject
python strategy_qa.py --subject llama3.2:3b --attacker gpt-4o --workers 10
```

### GSM8K / SVAMP (Math)

```bash
# Quick run
python gsm_qa.py

# SVAMP dataset
python gsm_qa.py --file data/svamp/test.jsonl --limit 300 --workers 20

# Single fallacy investigation
python gsm_qa.py --fallacy FALSE_DILEMMA --limit 100

# vLLM subject
python gsm_qa.py --subject vllm:Qwen/Qwen2.5-3B-Instruct --attacker gpt-4o --workers 30
```

### CLI Reference

All flags apply to both `strategy_qa.py` and `gsm_qa.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--file` | auto | Path to a custom `.jsonl` data file |
| `--split` | `train` | Dataset split (`train` / `test`) |
| `--limit` | `50` | Number of questions to run |
| `--max_turns` | `5` | Max attacker turns per fallacy conversation |
| `--subject` | `gpt-4o-mini` | Subject model |
| `--attacker` | `gpt-4o` | Attacker model |
| `--workers` | `1` | Parallel worker threads |
| `--output` | auto | Custom output file path |
| `--fallacy` | all | Comma-separated list of fallacies to run |

---

## Generating Visualizations

```bash
# StrategyQA — 4 figures
python visualize_strategyqa.py

# GSM8K / SVAMP — 6 figures (includes log-prob and marginal stability analysis)
python visualize_svamp.py

# Override labels and expected total
python visualize_strategyqa.py --subject llama3.2:3b --attacker gpt-4o --total 700
python visualize_svamp.py      --subject gpt-4o-mini  --attacker gpt-4o --total 300
```

### Output Figures

**StrategyQA** (`results/`):

| File | Contents |
|------|----------|
| `fig1_strategyqa_dashboard.png` | Overview dashboard — flip rates per fallacy |
| `fig2_strategyqa_table.png` | Summary table — success rate, avg turns, efficiency |
| `fig3_strategyqa_deepdive.png` | Per-fallacy deep dive — turn-by-turn breakdown |
| `fig4_strategyqa_tokens.png` | Token cost analysis by model and role |

**GSM8K / SVAMP** (`results/`):

| File | Contents |
|------|----------|
| `fig1_svamp_dashboard.png` | Overview dashboard |
| `fig2_svamp_table.png` | Summary table |
| `fig3_svamp_deepdive.png` | Per-fallacy deep dive |
| `fig4_svamp_tokens.png` | Token cost analysis |
| `fig5_svamp_logprob_analysis.png` | Perplexity / entropy / probability trajectories |
| `fig6_svamp_marginal_stability.png` | Brittle-holder detection and pre-flip collapse |

---

## Output Files

Results land in `results/` with the naming convention `{dataset}_{datetime}_{model}.*`:

| File | Contents |
|------|----------|
| `*.json` | Full results — all conversations, metrics, config |
| `*.live.jsonl` | Real-time streaming log (crash-safe partial recovery) |
| `*_checkpoint_N.json` | Checkpoint saved every 50 completed questions |
| `*_fallacy_summary.csv` | Per-fallacy efficiency metrics (flip rate, avg turns, token cost) |
| `*_question_detail.csv` | Per-question breakdown across all fallacies |

The `EmergencySaver` class catches `SIGINT` / `SIGTERM` and `atexit` to guarantee partial results are always saved, even on crashes or keyboard interrupts.

---

## Project Structure

```
.
├── strategy_qa.py          # StrategyQA experiment runner
├── gsm_qa.py               # GSM8K / SVAMP experiment runner
├── visualize_strategyqa.py # StrategyQA visualizer (4 figures)
├── visualize_svamp.py      # GSM8K / SVAMP visualizer (6 figures)
├── download_strategyqa.py  # Downloads StrategyQA from HuggingFace
├── download_svamp.py       # Downloads SVAMP from HuggingFace
├── requirements.txt
├── .env                    # Not committed — API keys
├── data/
│   ├── strategyqa/         # strategyqa_train.jsonl, strategyqa_test.jsonl
│   └── svamp/              # train.jsonl, test.jsonl
└── results/                # All output files land here
```

---

## Key Internal Classes

| Class | File | Role |
|-------|------|------|
| `TokenTracker` | both scripts | Thread-safe accumulation of API token usage by model and role |
| `LiveLogger` | both scripts | Appends one JSON line per completed question; `fsync`-safe |
| `EmergencySaver` | both scripts | Checkpoints every 50 questions; handles `SIGINT`/`SIGTERM`/`atexit` |

---

## Dependencies

```
openai>=1.40.0
python-dotenv>=1.0.0
pydantic>=2.0.0
datasets
pandas
matplotlib
numpy
```

Install all with `pip install -r requirements.txt`.
