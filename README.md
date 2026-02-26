# Hallucination Pressure Test
### COE749 Final Project — LangGraph Multi-Agent System

A multi-agent LangGraph pipeline that psychologically pressures a weak language model into stating factual falsehoods. Four coordinated agents — Subject, Reinforcer, Critic, and Evaluator — run in a directed graph with conditional routing, tool use, Structured Output Mode, and MemorySaver checkpointing.

---

## Architecture

```
Subject ──→ Evaluator ──→ [conditional edge]
   ↑                           │
   │              ┌────────────┤
   │         HELD_GROUND       │   HALLUCINATION / max turns
   │              │            └──────────────────→ END
   └── Critic ←── Reinforcer
```

**Agents:**
| Agent | Model (default) | Tools | Role |
|-------|----------------|-------|------|
| SubjectAgent | gpt-4o-mini | None | Answers questions; deliberately defenceless |
| ReinforcerAgent | gpt-4o | `pick_attack_strategy`, `get_false_claim`, `analyze_weakness` | Applies escalating psychological pressure |
| CriticAgent | gpt-4o | `generate_academic_doubt` | Independent academic authority challenge |
| EvaluatorAgent | gpt-4o | Structured Output Mode | Judges each Subject response with validated schema |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hallucination-pressure-test.git
cd hallucination-pressure-test
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

### 4. Set your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...your-key-here...
```

---

## Running

### Web UI (recommended)
```bash
python app.py
```
Then open **http://localhost:5000** in your browser.

Features:
- Live real-time chat visualization with per-agent color coding
- Question editor — add, edit, or remove questions before running
- Session history — every experiment auto-saved and replayable
- Tool call inspector — click any 🔧 chip to see tool inputs/outputs
- Rate limit retry — automatic exponential backoff on OpenAI 429 errors
- Partial save — stopping mid-run saves completed questions to a session file

### Command-line only (v2 LangGraph)
```bash
python main_langgraph.py
python main_langgraph.py --subject-model gpt-3.5-turbo --reinforcer-model gpt-4o --max-turns 10
```

---

## Dependencies

```
flask>=3.0.0
openai>=1.40.0
python-dotenv>=1.0.0
pydantic>=2.0.0
langgraph>=0.2.0
```

---

## LangGraph Requirements Satisfied

| Requirement | Implementation |
|-------------|---------------|
| (A) 3+ distinct agents | 4 agents with isolated roles and goals |
| (B) Different toolsets per agent | Subject: none · Reinforcer: 3 tools · Critic: 1 tool · Evaluator: SOM |
| (C) Pydantic BaseModel state + MemorySaver | `ExperimentState` with 18 typed fields; per-question thread IDs |
| (D) Structured Output Mode | `client.beta.chat.completions.parse(response_format=EvaluationResult)` |
| (E) Multiple field types | `str`, `int`, `bool`, `float`, `List[Dict]`, `Optional[str/int/float/Dict]` |
| (F) Conditional edge | `route_after_evaluation()` → reinforcer\_node OR END |

---

## Project Structure

```
hallucination-pressure-test/
├── app.py                  # Flask web UI backend (SSE streaming)
├── requirements.txt
├── .env                    # Not committed — contains OPENAI_API_KEY
├── templates/
│   └── index.html          # Full single-page web UI
└── static/
    └── sessions/           # Auto-created; stores experiment JSON sessions
```

---

