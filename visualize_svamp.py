"""
GSM8K Fallacy Pressure Test — Visualizer
==========================================
Reads results automatically from the results/ folder.
Supports single run OR multiple runs (auto-merges them).

USAGE:
    python visualize_svamp.py
    python visualize_svamp.py --results results/
    python visualize_svamp.py --summary results/gsm8k_..._summary.csv

Outputs 6 PNG figures into results/:
    fig1_svamp_dashboard.png
    fig2_svamp_table.png
    fig3_svamp_deepdive.png
    fig4_svamp_tokens.png
    fig5_svamp_logprob_analysis.png   ← NEW: perplexity / entropy / prob trajectories
    fig6_svamp_marginal_stability.png ← NEW: brittle-holder detection, pre-flip collapse
"""

import argparse
import re
from pathlib import Path

import numpy as np
import math as _math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas is required: pip install pandas")


# ==============================================================================
# ARGS
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SVAMP Fallacy Visualizer — supports partial runs")
    p.add_argument("--results",  type=str, default="results",
                   help="Folder containing result files (default: results/)")
    p.add_argument("--summary",  type=str, default=None,
                   help="Path to a specific _fallacy_summary.csv (overrides --results)")
    p.add_argument("--total",    type=int, default=300,
                   help="Total expected questions across all partial runs (default: 300)")
    p.add_argument("--subject",  type=str, default=None,
                   help="Subject model label for figure titles (e.g. llama3.2:3b)")
    p.add_argument("--attacker", type=str, default=None,
                   help="Attacker model label for figure titles (e.g. gpt-4o-mini)")
    return p.parse_args()


# ==============================================================================
# DATA LOADERS
# ==============================================================================

def load_summary_csvs(results_dir: Path) -> pd.DataFrame:
    """
    Load and merge fallacy summary CSVs from the results folder.
    Uses final files only; falls back to latest checkpoint for in-progress runs.
    """
    all_csvs = sorted(results_dir.glob("gsm8k_*_fallacy_summary.csv"))
    if not all_csvs:
        raise FileNotFoundError(
            f"No gsm8k_*_fallacy_summary.csv files found in '{results_dir}'.\n"
            "Run gsm_qa.py first to generate results."
        )

    run_files: dict = {}
    for csv in all_csvs:
        name = csv.name
        ts_m = re.match(
            r"gsm8k_(\d{8}_\d{6}_.+?)(?:_checkpoint_\d+)?_fallacy_summary\.csv$",
            name
        )
        if not ts_m:
            continue
        run_id = ts_m.group(1)
        if run_id not in run_files:
            run_files[run_id] = {"final": None, "checkpoints": []}
        if "_checkpoint_" in name:
            run_files[run_id]["checkpoints"].append(csv)
        else:
            run_files[run_id]["final"] = csv

    csvs = []
    for run_id, files in sorted(run_files.items()):
        if files["final"] is not None:
            csvs.append(files["final"])
        elif files["checkpoints"]:
            best = sorted(
                files["checkpoints"],
                key=lambda p: int(re.search(r"checkpoint_(\d+)", p.name).group(1))
            )[-1]
            csvs.append(best)
            print(f"  [In-progress] using latest checkpoint: {best.name}")

    if not csvs:
        raise FileNotFoundError("No usable summary CSVs found.")

    print(f"Found {len(csvs)} run(s) to merge:")
    for c in csvs:
        print(f"  {c.name}")

    frames = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        frames.append(df)

    if len(frames) == 1:
        return frames[0]

    combined = pd.concat(frames)

    if "total_no_answer" not in combined.columns:
        combined["total_no_answer"] = 0
    if "total_unresolvable" not in combined.columns:
        combined["total_unresolvable"] = 0

    sum_cols = [
        "total_hallucinated", "total_held_ground",
        "total_no_answer", "total_unresolvable",
        "total_pre_pressure_fail", "total_eligible",
    ]
    turn_cols = [c for c in combined.columns if re.match(r"hallucinated_at_T\d+", c)]
    sum_cols += turn_cols

    if "total_error" in combined.columns:
        sum_cols.append("total_error")

    agg = combined.groupby("fallacy")[sum_cols].sum().reset_index()

    agg["success_rate_pct"] = (
        agg["total_hallucinated"] / agg["total_eligible"].replace(0, np.nan) * 100
    ).fillna(0).round(2)

    def _avg_win(row):
        wins = weighted = 0
        for t_col in turn_cols:
            t = int(re.search(r"\d+", t_col).group())
            n = row[t_col]
            wins += n
            weighted += t * n
        return round(weighted / wins, 2) if wins > 0 else None

    agg["avg_turns_when_successful"] = agg.apply(_avg_win, axis=1)

    max_t = max(int(re.search(r"\d+", c).group()) for c in turn_cols) if turn_cols else 3

    def _avg_all(row):
        wins = weighted = 0
        for t_col in turn_cols:
            t = int(re.search(r"\d+", t_col).group())
            n = row[t_col]
            wins += n
            weighted += t * n
        elig  = row.get("total_eligible", 0)
        dnh   = max(0, elig - wins)
        total_w = weighted + dnh * (max_t + 1)
        return round(total_w / elig, 2) if elig > 0 else None

    agg["avg_turns_overall"] = agg.apply(_avg_all, axis=1)
    agg = agg.sort_values("success_rate_pct", ascending=False).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    return agg


def load_summary_txt(results_dir: Path) -> dict:
    txts = sorted(set(
        list(results_dir.glob("gsm8k_*_summary.txt")) +
        list(results_dir.glob("gsm8k_*.summary.txt"))
    ))
    txts = [t for t in txts if "_checkpoint_" not in t.name]
    if not txts:
        return {}

    total_tokens = gpt4o_tokens = mini_tokens = total_calls = 0
    total_hall = total_conv = total_pre = 0
    runs = []

    for txt_path in txts:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Session total\s*:\s*([\d,]+)", text)
        if m:
            t = int(m.group(1).replace(",", ""))
            total_tokens += t
            mq = re.search(r"(\d+)\s+questions\s+x\s+\d+\s+fallacies", text)
            n_q = int(mq.group(1)) if mq else 0
            runs.append({"file": txt_path.name, "tokens": t, "questions": n_q})

        model_matches = re.findall(r"^\s+(\S+)\s+total=([\d,]+)", text, re.MULTILINE)
        for i, (_, tok_str) in enumerate(model_matches):
            tok = int(tok_str.replace(",", ""))
            if i == 0: gpt4o_tokens += tok
            else:      mini_tokens  += tok

        mc = re.search(r"calls=(\d+)\)", text)
        if mc: total_calls += int(mc.group(1))
        mh = re.search(r"Pressure-induced hallucinations[:\s]+([\d]+)", text)
        if mh: total_hall += int(mh.group(1))
        mt = re.search(r"Total fallacy conversations[:\s]+([\d]+)", text)
        if mt: total_conv += int(mt.group(1))
        mp = re.search(r"Pre-pressure fails[^:]*:\s*([\d]+)", text)
        if mp: total_pre += int(mp.group(1))

    return {
        "total_tokens": total_tokens, "gpt4o_tokens": gpt4o_tokens,
        "mini_tokens": mini_tokens,   "total_calls":  total_calls,
        "total_hall":  total_hall,    "total_conv":   total_conv,
        "total_pre":   total_pre,     "runs":         runs,
        "questions_done": sum(r.get("questions", 0) for r in runs),
    }


def load_detail_csv(results_dir: Path) -> pd.DataFrame:
    csvs = sorted(results_dir.glob("gsm8k_*_question_detail.csv"))
    if not csvs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def load_turn_probabilities(results_dir: Path) -> pd.DataFrame:
    """
    Load and merge all turn_probabilities CSVs.
    These contain per-turn logprob, perplexity, entropy, and prob_stated/prob_target.
    Returns empty DataFrame if none found.
    """
    csvs = sorted(results_dir.glob("gsm8k_*_turn_probabilities.csv"))
    if not csvs:
        print("[Info] No turn_probabilities CSVs found — skipping logprob figures.")
        return pd.DataFrame()
    frames = []
    for c in csvs:
        df = pd.read_csv(c)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    # Cast numeric columns
    for col in ["prob_stated", "prob_target", "log_perplexity",
                "mean_token_entropy", "response_length_tokens", "perplexity"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["turn"] = pd.to_numeric(combined["turn"], errors="coerce")
    combined["clarification_turn"] = pd.to_numeric(
        combined.get("clarification_turn", 0), errors="coerce"
    ).fillna(0).astype(int)
    return combined


def load_answer_token_entropy(results_dir: Path) -> pd.DataFrame:
    """
    Compute answer_token_entropy from token_logprobs CSVs.

    This is the CORRECT measure of marginal answer stability.

    Global metrics (log_perplexity, mean_token_entropy) average over all ~300
    tokens in the response — most of which are reasoning prose with their own
    uncertainty profile unrelated to the answer decision. This dilutes the
    signal by a factor of ~300.

    answer_token_entropy is computed ONLY at the token where the model writes
    its final answer (the number after 'FINAL ANSWER:'). It measures:
        H = -∑ p_k × log(p_k)   for the top-5 alternatives at that position

    High entropy → model is spread across multiple candidate answers (unstable)
    Low entropy  → model is a spike at one answer (stable, committed)

    Tracking this across turns shows whether the answer commitment erodes
    before the surface answer changes — the true marginal stability trajectory.
    """
    csvs = sorted(results_dir.glob("gsm8k_*_token_logprobs.csv"))
    if not csvs:
        print("[Info] No token_logprobs CSVs found — skipping answer_token_entropy.")
        return pd.DataFrame()

    ANCHOR_NORM = "FINALANSWER"

    def _entropy_from_top5(row: pd.Series) -> float | None:
        """
        Compute Shannon entropy from the top-5 logprob columns of one token row.
        Returns None if fewer than 2 top-k entries exist.
        """
        lps = []
        for k in range(1, 6):
            lp_val = row.get(f"top{k}_logprob")
            if pd.notna(lp_val):
                try:
                    lp = float(lp_val)
                    if lp > -9998:          # filter sentinel values
                        lps.append(lp)
                except (ValueError, TypeError):
                    pass
        if len(lps) < 2:
            return None
        probs    = [_math.exp(lp) for lp in lps]
        residual = max(0.0, 1.0 - sum(probs))
        if residual > 1e-9:
            probs.append(residual)
        return -sum(p * _math.log(p + 1e-12) for p in probs)

    print(f"Computing answer_token_entropy from {len(csvs)} token_logprobs file(s)...")
    rows_out = []

    for csv_path in csvs:
        print(f"  Reading {csv_path.name}  ({csv_path.stat().st_size // 1_000_000}MB)...")
        try:
            tok_df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  [Error] Could not read {csv_path.name}: {e}")
            continue

        # Normalise column names — some files use 'token_pos', some 'pos'
        if "token_pos" not in tok_df.columns and "pos" in tok_df.columns:
            tok_df = tok_df.rename(columns={"pos": "token_pos"})

        tok_df["token_pos"] = pd.to_numeric(tok_df.get("token_pos", 0), errors="coerce")
        tok_df["logprob"]   = pd.to_numeric(tok_df.get("logprob", np.nan), errors="coerce")

        for (qid, fallacy, turn), group in tok_df.groupby(
                ["question_id", "fallacy", "turn"]):
            group = group.sort_values("token_pos")

            # Find the LAST occurrence of the FINALANSWER anchor
            text_so_far   = ""
            collapsed_prev = ""
            last_anchor_idx = -1

            for i, (_, row) in enumerate(group.iterrows()):
                tok = str(row.get("token", ""))
                text_so_far  += tok
                collapsed_now = re.sub(r"\s+", "", text_so_far).upper()
                if collapsed_now.count(ANCHOR_NORM) > collapsed_prev.count(ANCHOR_NORM):
                    last_anchor_idx = i
                collapsed_prev = collapsed_now

            if last_anchor_idx == -1:
                continue     # no FINAL ANSWER tag found

            # Scan up to 15 tokens after anchor for the first numeric token
            rows_list = list(group.iterrows())
            scanned   = 0
            MAX_AFTER = 15
            entropy_val = None

            for _, row in rows_list[last_anchor_idx + 1:]:
                tok = str(row.get("token", "")).strip().replace(",", "")
                if re.match(r"^[\s:.()\[\]/*\\|]+$", tok) or tok == "":
                    continue   # skip punctuation without counting budget
                scanned += 1
                if scanned > MAX_AFTER:
                    break
                if re.match(r"^-?\d[\d.]*$", tok):
                    entropy_val = _entropy_from_top5(row)
                    break

            rows_out.append({
                "question_id":          qid,
                "fallacy":              fallacy,
                "turn":                 int(turn),
                "answer_token_entropy": round(entropy_val, 6) if entropy_val is not None else None,
            })

    if not rows_out:
        print("  [Info] No answer tokens found in token_logprobs.")
        return pd.DataFrame()

    result = pd.DataFrame(rows_out)
    n_found  = result["answer_token_entropy"].notna().sum()
    n_total  = len(result)
    print(f"  answer_token_entropy computed for {n_found}/{n_total} turns "
          f"({100*n_found/n_total:.1f}%)")
    return result


def _merge_outcome(prob_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge hallucination outcome from question_detail into turn_probabilities.
    Adds columns: hallucinated, turns_to_hallucinate, hall_turn_num.
    """
    if detail_df.empty or prob_df.empty:
        return prob_df
    keep = detail_df[["question_id", "fallacy", "hallucinated",
                       "turns_to_hallucinate"]].copy()
    keep["hall_turn_num"] = pd.to_numeric(
        keep["turns_to_hallucinate"], errors="coerce"
    )
    merged = prob_df.merge(keep, on=["question_id", "fallacy"], how="left")
    merged["hallucinated"] = pd.to_numeric(
        merged.get("hallucinated", 0), errors="coerce"
    ).fillna(0).astype(int)
    return merged


# ==============================================================================
# STYLE
# ==============================================================================

BG     = "#0f1117"
PANEL  = "#1a1d27"
PANEL2 = "#20243a"
BORDER = "#2e3250"
TEXT   = "#e8eaf6"
SUBTEXT= "#8892b0"
ACCENT = "#7c83ff"

PALETTE = {
    "FALSE_INFORMATION":   "#ff4c6a",
    "APPEAL_TO_AUTHORITY": "#ff8c42",
    "AD_HOMINEM":          "#ffd166",
    "FALSE_DILEMMA":       "#06d6a0",
    "CAUSAL_FALLACY":      "#4cc9f0",
    "CIRCULAR_REASONING":  "#a855f7",
    "PURE_PRESSURE":       "#64748b",
    "SLIPPERY_SLOPE":      "#38bdf8",
    "BANDWAGON":           "#fb923c",
    "STRAW_MAN":           "#34d399",
}
DEFAULT_COLOR = "#94a3b8"
COLOR_HALL  = "#ff4c6a"   # red  — hallucinated
COLOR_HELD  = "#4cc9f0"   # blue — held ground

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "legend.facecolor":  PANEL2,
    "legend.edgecolor":  BORDER,
})


def get_color(fallacy: str) -> str:
    return PALETTE.get(fallacy, DEFAULT_COLOR)

def short_name(fallacy: str) -> str:
    return {
        "FALSE_INFORMATION":   "False Info",
        "APPEAL_TO_AUTHORITY": "Appeal to Auth.",
        "AD_HOMINEM":          "Ad Hominem",
        "FALSE_DILEMMA":       "False Dilemma",
        "CAUSAL_FALLACY":      "Causal Fallacy",
        "CIRCULAR_REASONING":  "Circular Reason.",
        "PURE_PRESSURE":       "Pure Pressure",
        "SLIPPERY_SLOPE":      "Slippery Slope",
        "BANDWAGON":           "Bandwagon",
        "STRAW_MAN":           "Straw Man",
    }.get(fallacy, fallacy)


# ==============================================================================
# FIGURE 1 — Main Dashboard (unchanged)
# ==============================================================================

def plot_dashboard(df, token_stats, out_dir, total_target=300,
                   subject=None, attacker=None):
    fallacies = df["fallacy"].tolist()
    labels    = [short_name(f) for f in fallacies]
    colors    = [get_color(f) for f in fallacies]
    rates     = df["success_rate_pct"].tolist()
    turn_cols = sorted(
        [c for c in df.columns if re.match(r"hallucinated_at_T\d+", c)],
        key=lambda x: int(re.search(r"\d+", x).group())
    )
    n_fallacies    = len(fallacies)
    runs_done      = len(token_stats.get("runs", []))
    questions_done = token_stats.get("questions_done", 0)
    if questions_done == 0 and n_fallacies:
        total_all = int((df["total_eligible"] + df["total_pre_pressure_fail"]).sum())
        questions_done = total_all // n_fallacies if n_fallacies else 0
    progress_pct = min(100, 100 * questions_done / total_target) if total_target else 100
    subj_label = subject or "Subject"
    atk_label  = attacker or "Attacker"

    fig = plt.figure(figsize=(20, 24), facecolor=BG)
    fig.suptitle(
        f"SVAMP Fallacy Pressure Test  |  {subj_label}  vs  {atk_label}",
        fontsize=17, fontweight="bold", color=TEXT, y=0.99
    )
    ax_prog = fig.add_axes([0.08, 0.955, 0.88, 0.018])
    ax_prog.set_xlim(0, 100); ax_prog.set_ylim(0, 1)
    ax_prog.set_facecolor(PANEL2)
    ax_prog.barh([0.5], [progress_pct], height=1.0,
                 color="#4cc9f0" if progress_pct < 100 else "#06d6a0",
                 edgecolor="none")
    ax_prog.set_xticks([]); ax_prog.set_yticks([])
    for spine in ax_prog.spines.values(): spine.set_edgecolor(BORDER)
    ax_prog.text(progress_pct / 2, 0.5,
                 f"{questions_done} / {total_target} questions"
                 f"  ({progress_pct:.0f}%)  —  {runs_done} partial run(s)",
                 ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.44, wspace=0.35,
                           left=0.08, right=0.96, top=0.93, bottom=0.04)
    ax_a = fig.add_subplot(gs[0, :])
    bars = ax_a.barh(labels, rates, color=colors, edgecolor="none", height=0.62)
    ax_a.set_xlabel("Hallucination Rate (%)", fontsize=11, color=TEXT)
    ax_a.set_title("A  |  Hallucination Success Rate by Fallacy  (all questions)",
                   fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax_a.invert_yaxis()
    max_rate = max(rates) if rates else 10
    ax_a.set_xlim(0, max(max_rate * 1.35, 1))
    ax_a.axvline(0, color=BORDER, lw=1)
    ax_a.grid(axis="x", alpha=0.4)
    ax_a.set_facecolor(PANEL)
    for bar, rate, f in zip(bars, rates, fallacies):
        n    = int(df.loc[df["fallacy"] == f, "total_hallucinated"].values[0])
        elig = int(df.loc[df["fallacy"] == f, "total_eligible"].values[0])
        ax_a.text(rate + max_rate * 0.01,
                  bar.get_y() + bar.get_height() / 2,
                  f"  {rate:.1f}%  ({n}/{elig})",
                  va="center", ha="left", fontsize=10, color=TEXT, fontweight="bold")

    ax_b = fig.add_subplot(gs[1, 0])
    turn_colors = ["#7c83ff", "#4cc9f0", "#ff8c42", "#ff4c6a", "#a855f7"]
    x = np.arange(len(fallacies)); w = 0.55
    bottoms = np.zeros(len(fallacies))
    for i, t_col in enumerate(turn_cols):
        vals  = df[t_col].fillna(0).tolist()
        t_num = re.search(r"\d+", t_col).group()
        ax_b.bar(x, vals, w, bottom=bottoms, label=f"Turn {t_num}",
                 color=turn_colors[i % len(turn_colors)], edgecolor="none")
        bottoms += np.array(vals)
    ax_b.set_xticks(x); ax_b.set_xticklabels(labels, rotation=33, ha="right", fontsize=9)
    ax_b.set_ylabel("# Hallucinations", fontsize=10)
    ax_b.set_title("B  |  Turn at Which Hallucination Occurred",
                   fontsize=11, fontweight="bold", pad=8)
    ax_b.legend(fontsize=9, loc="upper right")
    ax_b.grid(axis="y", alpha=0.4); ax_b.set_facecolor(PANEL)
    ax_b.yaxis.set_major_locator(MaxNLocator(integer=True))
    for i, total in enumerate(bottoms):
        if total > 0:
            ax_b.text(i, total + 0.3, str(int(total)),
                      ha="center", va="bottom", fontsize=9,
                      color=TEXT, fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 1])
    hall_vals = df["total_hallucinated"].fillna(0).tolist()
    held_vals = df["total_held_ground"].fillna(0).tolist()
    na_vals   = (df["total_no_answer"].fillna(0).tolist()
                 if "total_no_answer" in df.columns else [0]*len(fallacies))
    pre_vals  = df["total_pre_pressure_fail"].fillna(0).tolist()
    totals    = [h+he+na+p for h,he,na,p in zip(hall_vals,held_vals,na_vals,pre_vals)]
    def pct(v, t): return [100*a/b if b else 0 for a,b in zip(v,t)]
    pre_p = pct(pre_vals,totals); held_p = pct(held_vals,totals)
    na_p  = pct(na_vals,totals);  hall_p = pct(hall_vals,totals)
    left1 = pre_p
    left2 = [a+b for a,b in zip(left1,held_p)]
    left3 = [a+b for a,b in zip(left2,na_p)]
    ax_c.barh(labels, pre_p,  color="#374151", edgecolor="none", label="Pre-pressure fail")
    ax_c.barh(labels, held_p, left=left1, color="#1e3a5f", edgecolor="none", label="Held Ground")
    ax_c.barh(labels, na_p,   left=left2, color="#854d0e", edgecolor="none", label="No Answer")
    ax_c.barh(labels, hall_p, left=left3, color=colors, edgecolor="none", label="Hallucinated")
    ax_c.set_xlim(0, 100); ax_c.set_xlabel("% of all conversations", fontsize=10)
    ax_c.set_title("C  |  Conversation Outcome Breakdown",
                   fontsize=11, fontweight="bold", pad=8)
    ax_c.invert_yaxis(); ax_c.legend(fontsize=8.5, loc="lower right")
    ax_c.grid(axis="x", alpha=0.4); ax_c.set_facecolor(PANEL)

    ax_d = fig.add_subplot(gs[2, 0]); ax_d.set_facecolor(PANEL)
    avg_win_vals = pd.to_numeric(df["avg_turns_when_successful"], errors="coerce").tolist()
    avg_all_vals = pd.to_numeric(df["avg_turns_overall"], errors="coerce").tolist()
    x2 = np.arange(len(fallacies)); w2 = 0.38
    ax_d.bar(x2-w2/2, [v if not np.isnan(v) else 0 for v in avg_win_vals],
             w2, label="Avg turns (wins only)", color="#7c83ff", edgecolor="none", alpha=0.9)
    ax_d.bar(x2+w2/2, [v if not np.isnan(v) else 0 for v in avg_all_vals],
             w2, label="Avg turns (all incl. DNH)", color="#ff4c6a", edgecolor="none", alpha=0.9)
    ax_d.set_xticks(x2); ax_d.set_xticklabels(labels, rotation=33, ha="right", fontsize=9)
    ax_d.set_ylabel("Avg Turns", fontsize=10)
    ax_d.set_title("D  |  Average Turns to Hallucinate",
                   fontsize=11, fontweight="bold", pad=8)
    ax_d.legend(fontsize=9); ax_d.grid(axis="y", alpha=0.4)

    ax_e = fig.add_subplot(gs[2, 1], polar=True)
    N = len(fallacies)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    max_r  = max(rates) if max(rates) > 0 else 1
    vals_n = [r/max_r for r in rates] + [rates[0]/max_r]
    ax_e.set_facecolor(PANEL)
    ax_e.plot(angles, vals_n, color="#ff4c6a", linewidth=2.2)
    ax_e.fill(angles, vals_n, color="#ff4c6a", alpha=0.22)
    ax_e.set_xticks(angles[:-1]); ax_e.set_xticklabels(labels, size=9, color=TEXT)
    ax_e.set_yticklabels([]); ax_e.spines["polar"].set_color(BORDER)
    ax_e.grid(color=BORDER, linewidth=0.8)
    ax_e.set_title("E  |  Effectiveness Radar\n(normalized to max)",
                   fontsize=11, fontweight="bold", pad=18, color=TEXT)
    for angle, val, rate in zip(angles[:-1], vals_n[:-1], rates):
        ax_e.annotate(f"{rate:.1f}%", xy=(angle, val),
                      xytext=(angle, val+0.09), ha="center", va="center",
                      fontsize=7.5, color=TEXT, fontweight="bold")

    out = out_dir / "fig1_svamp_dashboard.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"Saved -> {out}")


# ==============================================================================
# FIGURE 2 — Summary Table (unchanged)
# ==============================================================================

def plot_table(df, token_stats, out_dir, total_target=300,
               subject=None, attacker=None):
    fig2, ax = plt.subplots(figsize=(22, 5.5), facecolor=BG)
    ax.set_facecolor(BG); ax.axis("off")
    questions_done = token_stats.get("questions_done", 0)
    subj_label = subject or "Subject"; atk_label = attacker or "Attacker"
    progress   = f"{questions_done}/{total_target}q" if questions_done else ""
    fig2.suptitle(
        f"SVAMP Fallacy Efficiency Summary Table  |  {subj_label} vs {atk_label}"
        + (f"  |  {progress}" if progress else ""),
        fontsize=14, fontweight="bold", color=TEXT, y=1.01
    )
    turn_cols = sorted(
        [c for c in df.columns if re.match(r"hallucinated_at_T\d+", c)],
        key=lambda x: int(re.search(r"\d+", x).group())
    )
    seen = set()
    turn_cols = [c for c in turn_cols if not (c in seen or seen.add(c))]
    t_headers = [f"T{re.search(r'[0-9]+',c).group()} Wins" for c in turn_cols]
    has_na = "total_no_answer" in df.columns
    cols = (["Rank","Fallacy","Success\nRate %","Avg Turns\n(Wins)","Avg Turns\n(All)"]
            + t_headers
            + ["Total\nHall.","Total\nHeld"]
            + (["No\nAnswer"] if has_na else [])
            + ["Pre-\nFail","Eligible"])
    rows = []
    for _, row in df.sort_values("rank").iterrows():
        r_vals = [
            str(int(row.get("rank",""))), short_name(row["fallacy"]),
            f"{row['success_rate_pct']:.2f}%",
            f"{row['avg_turns_when_successful']:.2f}" if pd.notna(row.get("avg_turns_when_successful")) else "—",
            f"{row['avg_turns_overall']:.2f}" if pd.notna(row.get("avg_turns_overall")) else "—",
        ]
        for t_col in turn_cols: r_vals.append(str(int(row.get(t_col, 0))))
        r_vals += [str(int(row.get("total_hallucinated",0))),
                   str(int(row.get("total_held_ground",0)))]
        if has_na: r_vals.append(str(int(row.get("total_no_answer",0))))
        r_vals += [str(int(row.get("total_pre_pressure_fail",0))),
                   str(int(row.get("total_eligible",0)))]
        rows.append(r_vals)
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1, 2.5)
    for j in range(len(cols)):
        cell = table[0, j]; cell.set_facecolor(ACCENT)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor(BG)
    for i, (_, row) in enumerate(df.sort_values("rank").iterrows(), 1):
        f = row["fallacy"]; row_bg = "#1e2235" if i%2==0 else PANEL2
        for j in range(len(cols)):
            cell = table[i, j]; cell.set_facecolor(row_bg)
            cell.set_text_props(color=TEXT); cell.set_edgecolor(BG)
        table[i, 2].set_text_props(color=get_color(f), fontweight="bold")
        table[i, 0].set_text_props(color=get_color(f), fontweight="bold")
    plt.tight_layout()
    out = out_dir / "fig2_svamp_table.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"Saved -> {out}")


# ==============================================================================
# FIGURE 3 — Deep-Dive (unchanged)
# ==============================================================================

def plot_deepdive(df, detail_df, out_dir):
    fallacies = df["fallacy"].tolist()
    labels    = [short_name(f) for f in fallacies]
    colors    = [get_color(f) for f in fallacies]
    rates     = df["success_rate_pct"].tolist()
    hall_vals = df["total_hallucinated"].fillna(0).tolist()

    fig3, axes = plt.subplots(1, 3, figsize=(19, 6.5), facecolor=BG)
    fig3.suptitle("SVAMP Deep-Dive: Top Fallacy Analysis",
                  fontsize=15, fontweight="bold", color=TEXT, y=1.02)

    ax3a = axes[0]; ax3a.set_facecolor(BG)
    pie_data = [(f, lbl, col, n) for f, lbl, col, n
                in zip(fallacies, labels, colors, hall_vals) if n > 0]
    total_hall = sum(hall_vals)
    if not pie_data:
        ax3a.text(0.5, 0.5, "No hallucinations\nrecorded yet",
                  ha="center", va="center", fontsize=13, color=SUBTEXT,
                  transform=ax3a.transAxes)
        ax3a.set_title(f"A  |  Share of All Hallucinations\n(total = 0)",
                       fontsize=11, fontweight="bold", color=TEXT, pad=12)
    else:
        wedges, texts, autotexts = ax3a.pie(
            [d[3] for d in pie_data], colors=[d[2] for d in pie_data],
            autopct=lambda p: f"{p:.1f}%" if p > 1.5 else "",
            startangle=140, wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
            pctdistance=0.75)
        for at in autotexts:
            at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
        ax3a.set_title(f"A  |  Share of All Hallucinations\n(total = {int(total_hall)})",
                       fontsize=11, fontweight="bold", color=TEXT, pad=12)
        legend_handles = [
            mpatches.Patch(facecolor=get_color(f), label=f"{short_name(f)} ({int(n)})")
            for f, n in zip(fallacies, hall_vals)
        ]
        ax3a.legend(handles=legend_handles, loc="lower left", fontsize=8,
                    framealpha=0.7, bbox_to_anchor=(-0.3, -0.22))

    ax3b = axes[1]; ax3b.set_facecolor(PANEL)
    if rates and any(r > 0 for r in rates):
        top_f = fallacies[0]; top_rate = rates[0]
        other_avg = np.mean([r for r in rates[1:] if r > 0]) if any(r > 0 for r in rates[1:]) else 0
        b = ax3b.bar([short_name(top_f), "Others\n(avg)"],
                     [top_rate, other_avg],
                     color=[get_color(top_f), "#4a5568"], edgecolor="none", width=0.45)
        for bar, val in zip(b, [top_rate, other_avg]):
            ax3b.text(bar.get_x()+bar.get_width()/2, val+max(top_rate,other_avg)*0.02,
                      f"{val:.2f}%", ha="center", va="bottom",
                      fontsize=13, fontweight="bold", color=TEXT)
        if other_avg > 0:
            ax3b.text(0.5, max(top_rate,other_avg)*0.5,
                      f"{top_rate/other_avg:.1f}x\nmore effective",
                      ha="center", fontsize=12, color="#ffd166", fontweight="bold")
    else:
        ax3b.text(0.5, 0.5, "No hallucinations\nrecorded yet",
                  transform=ax3b.transAxes, ha="center", va="center",
                  fontsize=13, color=SUBTEXT)
    ax3b.set_ylabel("Hallucination Rate (%)", fontsize=10)
    ax3b.set_title("B  |  Top Fallacy vs\nAverage of All Others",
                   fontsize=11, fontweight="bold", pad=8)
    ax3b.grid(axis="y", alpha=0.4)

    ax3c = axes[2]; ax3c.set_facecolor(PANEL)
    turn_cols = sorted([c for c in df.columns if re.match(r"hallucinated_at_T\d+", c)],
                       key=lambda x: int(re.search(r"\d+", x).group()))
    eligible_vals = df["total_eligible"].fillna(1).tolist()
    if turn_cols:
        t1_col = turn_cols[0]
        t1_vals = df[t1_col].fillna(0).tolist()
        t1_rates = [100*t1/elig if elig else 0 for t1,elig in zip(t1_vals,eligible_vals)]
        bars_c = ax3c.barh(labels, t1_rates, color=colors, edgecolor="none", height=0.6)
        ax3c.invert_yaxis()
        ax3c.set_xlabel("% of eligible questions fooled at Turn 1", fontsize=10)
        ax3c.set_title("C  |  Turn-1 Capture Rate",
                       fontsize=11, fontweight="bold", pad=8)
        ax3c.grid(axis="x", alpha=0.4)
        max_t1 = max(t1_rates) if t1_rates else 1
        for bar, val, t1_n in zip(bars_c, t1_rates, t1_vals):
            ax3c.text(val+max(max_t1*0.01,0.1), bar.get_y()+bar.get_height()/2,
                      f"  {val:.1f}% ({int(t1_n)})", va="center", fontsize=9, color=TEXT)
    else:
        ax3c.text(0.5, 0.5, "No turn data available",
                  transform=ax3c.transAxes, ha="center", va="center",
                  fontsize=13, color=SUBTEXT)

    plt.tight_layout(pad=1.5)
    out = out_dir / "fig3_svamp_deepdive.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"Saved -> {out}")


# ==============================================================================
# FIGURE 4 — Token Cost (unchanged)
# ==============================================================================

def plot_tokens(token_stats, df, out_dir, subject=None, attacker=None):
    if not token_stats or token_stats.get("total_tokens", 0) == 0:
        print("[Skip] No token data found — skipping fig4."); return
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig4.suptitle("Token Cost Analysis  (all runs combined)",
                  fontsize=14, fontweight="bold", color=TEXT, y=1.02)
    ax4a = axes4[0]; ax4a.set_facecolor(PANEL)
    gpt4o = token_stats.get("gpt4o_tokens",0); mini = token_stats.get("mini_tokens",0)
    total = token_stats.get("total_tokens",0)
    if gpt4o==0 and mini==0:
        ax4a.bar(["Total"], [total/1e6], color=ACCENT, edgecolor="none", width=0.4)
        ax4a.text(0, total/1e6+0.1, f"{total/1e6:.1f}M",
                  ha="center", va="bottom", fontsize=11, color=TEXT, fontweight="bold")
    else:
        atk_lbl = attacker or "Attacker"; subj_lbl = subject or "Subject"
        b = ax4a.bar([f"{atk_lbl}\n(Attacker)",f"{subj_lbl}\n(Subject)","Combined"],
                     [gpt4o/1e6, mini/1e6, total/1e6],
                     color=["#7c83ff","#ff4c6a","#ffd166"], edgecolor="none", width=0.45)
        for bar, val in zip(b, [gpt4o/1e6, mini/1e6, total/1e6]):
            ax4a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                      f"{val:.1f}M", ha="center", va="bottom",
                      fontsize=10, color=TEXT, fontweight="bold")
    ax4a.set_ylabel("Tokens (millions)", fontsize=10)
    ax4a.set_title("A  |  Token Usage by Model", fontsize=11, fontweight="bold", pad=8)
    ax4a.grid(axis="y", alpha=0.4)
    ax4b = axes4[1]; ax4b.set_facecolor(PANEL)
    total_hall = token_stats.get("total_hall",0); total_conv = token_stats.get("total_conv",0)
    if total_hall > 0 and total > 0:
        tph = total/total_hall; tpc = total/total_conv if total_conv>0 else 0
        b2 = ax4b.bar(["Tokens per\nHallucination","Tokens per\nConversation"],
                      [tph/1000, tpc/1000], color=["#ff4c6a","#7c83ff"],
                      edgecolor="none", width=0.4)
        for bar, val in zip(b2, [tph/1000, tpc/1000]):
            ax4b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                      f"{val:.0f}K\ntokens", ha="center", va="bottom",
                      fontsize=11, color=TEXT, fontweight="bold")
        ax4b.set_ylabel("Tokens (thousands)", fontsize=10)
        ax4b.set_title("B  |  Token Efficiency Metrics",
                       fontsize=11, fontweight="bold", pad=8)
        ax4b.grid(axis="y", alpha=0.4)
        ax4b.text(0.5, 0.05,
                  f"Total: {total/1e6:.1f}M tokens\n"
                  f"{int(total_hall)} hallucinations  |  {int(total_conv)} conversations",
                  transform=ax4b.transAxes, ha="center", fontsize=9, color=SUBTEXT)
    else:
        ax4b.text(0.5, 0.5, "Token-per-hallucination data\nnot available yet",
                  transform=ax4b.transAxes, ha="center", va="center",
                  fontsize=12, color=SUBTEXT)
    plt.tight_layout(pad=1.5)
    out = out_dir / "fig4_svamp_tokens.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"Saved -> {out}")


# ==============================================================================
# FIGURE 5 — Logprob / Perplexity / Entropy Analysis  ← NEW
# ==============================================================================

def plot_logprob_analysis(prob_df: pd.DataFrame, detail_df: pd.DataFrame,
                          summary_df: pd.DataFrame, out_dir: Path,
                          subject: str = None, attacker: str = None):
    """
    Figure 5: Four panels examining how log_perplexity, mean_token_entropy,
    and prob_stated evolve across turns split by outcome.

    Panel A — log_perplexity trajectory by turn: hallucinated vs held-ground
               (mean ± std across all conversations in each group)
    Panel B — mean_token_entropy trajectory by turn: same grouping
    Panel C — T0 log_perplexity distribution: hallucinated vs held-ground
               (boxplot — tests whether baseline predicts outcome)
    Panel D — log_perplexity at T0 vs at flip/last turn per fallacy
               (scatter: each dot = one fallacy, size = success rate)
    """
    if prob_df.empty or detail_df.empty:
        print("[Skip] No turn_probabilities data — skipping fig5.")
        return

    merged = _merge_outcome(prob_df, detail_df)
    # exclude clarification turns from trajectory analysis
    merged = merged[merged["clarification_turn"] == 0].copy()
    merged["turn"] = pd.to_numeric(merged["turn"], errors="coerce")

    subj_label = subject or "Subject"
    atk_label  = attacker or "Attacker"

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), facecolor=BG)
    fig.suptitle(
        f"Fig 5  |  Internal Confidence Dynamics Under Adversarial Pressure\n"
        f"{subj_label}  vs  {atk_label}",
        fontsize=14, fontweight="bold", color=TEXT, y=1.01
    )
    axes = axes.flatten()

    # ── Panel A: log_perplexity trajectory ───────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)
    turns_present = sorted(merged["turn"].dropna().unique().astype(int))

    for outcome, color, label in [
        (1, COLOR_HALL, "Hallucinated"),
        (0, COLOR_HELD, "Held Ground"),
    ]:
        sub = merged[merged["hallucinated"] == outcome]
        means, stds, xs = [], [], []
        for t in turns_present:
            vals = sub[sub["turn"] == t]["log_perplexity"].dropna()
            if len(vals) >= 2:
                means.append(vals.mean()); stds.append(vals.std()); xs.append(t)
        if means:
            xs_ = np.array(xs); means_ = np.array(means); stds_ = np.array(stds)
            ax.plot(xs_, means_, "o-", color=color, linewidth=2.5,
                    markersize=7, label=label, zorder=3)
            ax.fill_between(xs_, means_-stds_, means_+stds_,
                            alpha=0.18, color=color)

    ax.set_xlabel("Turn", fontsize=11); ax.set_xticks(turns_present)
    ax.set_ylabel("log-Perplexity (mean ± 1 std)", fontsize=11)
    ax.set_title(
        "A  |  log-Perplexity Trajectory by Outcome\n"
        "(higher = model more uncertain / generating more diverse tokens)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.legend(fontsize=10); ax.grid(alpha=0.35)
    ax.text(0.02, 0.97,
            "Null result: if lines overlap at T0,\nbaseline uncertainty does not predict outcome",
            transform=ax.transAxes, fontsize=8, color=SUBTEXT,
            va="top", style="italic")

    # ── Panel B: answer_token_entropy trajectory (true marginal stability) ────
    # answer_token_entropy = entropy computed at the FINAL ANSWER decision token
    # only — NOT averaged over all response tokens like mean_token_entropy.
    # This directly measures: how spread is the model across candidate answers?
    # Falls back to mean_token_entropy if token_logprobs CSV was not provided.
    ax = axes[1]
    ax.set_facecolor(PANEL)

    ate_col  = "answer_token_entropy"
    has_ate  = ate_col in merged.columns and merged[ate_col].notna().any()
    plot_col = ate_col if has_ate else "mean_token_entropy"

    for outcome, color, label in [
        (1, COLOR_HALL, "Hallucinated"),
        (0, COLOR_HELD, "Held Ground"),
    ]:
        sub = merged[merged["hallucinated"] == outcome]
        means, stds, xs = [], [], []
        for t in turns_present:
            vals = sub[sub["turn"] == t][plot_col].dropna()
            if len(vals) >= 2:
                means.append(vals.mean()); stds.append(vals.std()); xs.append(t)
        if means:
            xs_ = np.array(xs); means_ = np.array(means); stds_ = np.array(stds)
            ax.plot(xs_, means_, "s-", color=color, linewidth=2.5,
                    markersize=7, label=label, zorder=3)
            ax.fill_between(xs_, means_-stds_, means_+stds_,
                            alpha=0.18, color=color)

    ax.set_xlabel("Turn", fontsize=11); ax.set_xticks(turns_present)
    if has_ate:
        ax.set_ylabel("Answer Token Entropy (nats)", fontsize=11)
        ax.set_title(
            "B  |  Answer Token Entropy Trajectory by Outcome\n"
            "(entropy at FINAL ANSWER decision token — true marginal stability)",
            fontsize=11, fontweight="bold", pad=8
        )
        ax.text(0.02, 0.97,
                "HIGH = model spread across candidate answers (unstable)\n"
                "LOW  = model committed to one answer (stable)\n"
                "Rising gap = answer commitment eroding in hallucinated conversations",
                transform=ax.transAxes, fontsize=8, color=SUBTEXT,
                va="top", style="italic")
    else:
        ax.set_ylabel("Mean Token Entropy — global proxy (nats)", fontsize=11)
        ax.set_title(
            "B  |  Mean Token Entropy Trajectory — global proxy\n"
            "⚠ Provide token_logprobs CSV for true answer_token_entropy",
            fontsize=11, fontweight="bold", pad=8
        )
        ax.text(0.02, 0.97,
                "⚠ Global measure: averaged over all ~300 tokens\n"
                "NOT specific to the answer decision point\n"
                "Add token_logprobs CSV for the correct metric",
                transform=ax.transAxes, fontsize=8, color="#ffd166",
                va="top", style="italic")
    ax.legend(fontsize=10); ax.grid(alpha=0.35)

    # ── Panel C: T0 log_perplexity boxplot — does baseline predict outcome? ──
    ax = axes[2]
    ax.set_facecolor(PANEL)
    t0_data = merged[merged["turn"] == 0]
    hall_t0 = t0_data[t0_data["hallucinated"] == 1]["log_perplexity"].dropna().tolist()
    held_t0 = t0_data[t0_data["hallucinated"] == 0]["log_perplexity"].dropna().tolist()

    if hall_t0 and held_t0:
        bp = ax.boxplot(
            [hall_t0, held_t0],
            labels=["Hallucinated\n(eventually)", "Held Ground\n(all turns)"],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=SUBTEXT), capprops=dict(color=SUBTEXT),
            flierprops=dict(marker="o", color=SUBTEXT, markersize=4, alpha=0.5),
        )
        bp["boxes"][0].set_facecolor(COLOR_HALL + "55")
        bp["boxes"][0].set_edgecolor(COLOR_HALL)
        bp["boxes"][1].set_facecolor(COLOR_HELD + "55")
        bp["boxes"][1].set_edgecolor(COLOR_HELD)

        diff      = abs(np.mean(hall_t0) - np.mean(held_t0))
        hall_std  = np.std(hall_t0)
        # Honest annotation: if diff < 10% of std, boxes overlap — null result
        is_null   = diff < 0.1 * hall_std
        annotation = (
            f"Mean diff: {diff:.4f}\n"
            + ("← NULL RESULT\nBoxes overlap completely.\n"
               "Baseline confidence does NOT\npredict susceptibility."
               if is_null else
               f"← detectable (diff > 10% of σ)")
        )
        annot_color = SUBTEXT if is_null else "#ffd166"
        ax.text(0.5, 0.95, annotation,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=annot_color, fontweight="bold")

    ax.set_ylabel("log-Perplexity at Turn 0", fontsize=11)
    ax.set_title(
        "C  |  T0 Baseline Uncertainty Distribution\n"
        "(tests whether initial confidence predicts susceptibility)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.grid(axis="y", alpha=0.35)

    # ── Panel D: per-fallacy T0 vs final log_ppl (scatter) ───────────────────
    ax = axes[3]
    ax.set_facecolor(PANEL)
    fallacies_in_data = merged["fallacy"].unique()

    for fallacy in fallacies_in_data:
        sub  = merged[merged["fallacy"] == fallacy]
        t0_m = sub[sub["turn"] == 0]["log_perplexity"].dropna()
        tf_m = sub[sub["turn"] == sub["turn"].max()]["log_perplexity"].dropna()
        if t0_m.empty or tf_m.empty:
            continue
        t0_val = t0_m.mean(); tf_val = tf_m.mean()
        rate_row = summary_df[summary_df["fallacy"] == fallacy]
        rate = float(rate_row["success_rate_pct"].values[0]) if not rate_row.empty else 20
        size = max(60, rate * 3.5)
        color = get_color(fallacy)
        ax.scatter(t0_val, tf_val, s=size, color=color,
                   alpha=0.85, edgecolors="white", linewidth=0.8, zorder=3)
        ax.annotate(short_name(fallacy), (t0_val, tf_val),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=color)

    lim_min = merged["log_perplexity"].dropna().min()
    lim_max = merged["log_perplexity"].dropna().max()
    margin  = (lim_max - lim_min) * 0.1
    diag_range = [lim_min - margin, lim_max + margin]
    ax.plot(diag_range, diag_range, "--", color=SUBTEXT,
            linewidth=1.2, alpha=0.6, label="y = x  (no change)")
    ax.fill_between(diag_range, diag_range, [lim_max+margin]*2,
                    alpha=0.05, color=COLOR_HALL)
    ax.text(lim_min + margin*0.5, lim_max,
            "↑ pressure increased entropy", fontsize=8, color=COLOR_HALL, alpha=0.8)

    ax.set_xlabel("Mean log-Perplexity at Turn 0 (baseline)", fontsize=11)
    ax.set_ylabel("Mean log-Perplexity at Last Turn", fontsize=11)
    ax.set_title(
        "D  |  Perplexity Shift per Fallacy  (T0 → Last Turn)\n"
        "(dot size = success rate %)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.35)

    plt.tight_layout(pad=2.0)
    out = out_dir / "fig5_svamp_logprob_analysis.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"Saved -> {out}")


# ==============================================================================
# FIGURE 6 — Marginal Stability / Brittle-Holder Analysis  ← NEW
# ==============================================================================

def plot_marginal_stability(prob_df: pd.DataFrame, detail_df: pd.DataFrame,
                            summary_df: pd.DataFrame, out_dir: Path,
                            subject: str = None, attacker: str = None):
    """
    Figure 6: Four panels examining pre-flip confidence collapse and the
    prob_stated signal as a leading indicator of hallucination.

    Panel A — prob_stated distribution at the FLIP TURN vs held-ground turns
               (shows whether the model is confident when it flips)
    Panel B — prob_stated at T-1 (one turn BEFORE the flip) for hallucinated
               conversations vs the same turn for held-ground conversations
               (detects brittle holders: correct answer but low confidence)
    Panel C — prob_target at each turn: hallucinated vs held-ground
               (shows whether the wrong answer creeps into top-k before flip)
    Panel D — Response length (tokens) trajectory: hallucinated vs held-ground
               (shorter = model capitulating vs longer = model defending)
    """
    if prob_df.empty or detail_df.empty:
        print("[Skip] No turn_probabilities data — skipping fig6.")
        return

    merged = _merge_outcome(prob_df, detail_df)
    merged = merged[merged["clarification_turn"] == 0].copy()
    merged["turn"] = pd.to_numeric(merged["turn"], errors="coerce")
    merged["prob_stated"] = pd.to_numeric(merged["prob_stated"], errors="coerce")
    merged["prob_target"] = pd.to_numeric(merged["prob_target"], errors="coerce")
    merged["response_length_tokens"] = pd.to_numeric(
        merged["response_length_tokens"], errors="coerce"
    )

    subj_label = subject or "Subject"
    atk_label  = attacker or "Attacker"
    turns_present = sorted(merged["turn"].dropna().unique().astype(int))

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), facecolor=BG)
    fig.suptitle(
        f"Fig 6  |  Marginal Stability & Pre-Flip Confidence Collapse\n"
        f"{subj_label}  vs  {atk_label}",
        fontsize=14, fontweight="bold", color=TEXT, y=1.01
    )
    axes = axes.flatten()

    # ── Panel A: prob_stated distribution at flip turn vs held-ground ─────────
    ax = axes[0]
    ax.set_facecolor(PANEL)

    # Flip-turn rows: turn == hall_turn_num
    hall_rows  = merged[merged["hallucinated"] == 1].copy()
    flip_rows  = hall_rows[hall_rows["turn"] == hall_rows["hall_turn_num"]]
    held_rows  = merged[merged["hallucinated"] == 0].copy()
    held_attack = held_rows[held_rows["turn"] > 0]   # exclude T0

    flip_ps  = flip_rows["prob_stated"].dropna().tolist()
    held_ps  = held_attack["prob_stated"].dropna().tolist()

    if flip_ps and held_ps:
        bins = np.linspace(0, 1, 25)
        ax.hist(held_ps, bins=bins, color=COLOR_HELD, alpha=0.55,
                label=f"Held-ground turns (n={len(held_ps)})", density=True)
        ax.hist(flip_ps, bins=bins, color=COLOR_HALL, alpha=0.55,
                label=f"Hallucination flip turn (n={len(flip_ps)})", density=True)
        ax.axvline(np.mean(flip_ps), color=COLOR_HALL, linewidth=2, linestyle="--",
                   label=f"Mean flip: {np.mean(flip_ps):.3f}")
        ax.axvline(np.mean(held_ps), color=COLOR_HELD, linewidth=2, linestyle="--",
                   label=f"Mean held: {np.mean(held_ps):.3f}")

    ax.set_xlabel("prob_stated at decision token", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "A  |  prob_stated Distribution at Flip Turn vs Held-Ground\n"
        "(both groups ≈1.0 confirms flips are confident, not hedged)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.35)
    ax.text(0.02, 0.97,
            "If both peaks at 1.0: model always\ncommits fully regardless of direction",
            transform=ax.transAxes, fontsize=8, color=SUBTEXT,
            va="top", style="italic")

    # ── Panel B: pre-flip prob_target — ALL three flip cohorts T1/T2/T3 ──────
    # T1 flips: pre-flip = T0 (attacker not yet spoken) → prob_target = 0% always
    # T2 flips: pre-flip = T1 (model correct after 1 attack) → some signal
    # T3 flips: pre-flip = T2 (model correct after 2 attacks) → more signal
    # Showing all three together reveals the GROWING SIGNAL pattern:
    # the longer the model holds, the more the wrong answer accumulates internally.
    ax = axes[1]
    ax.set_facecolor(PANEL)

    cohorts = {
        "T1 flips\n(pre-flip=T0)\nno attack yet": [],
        "T2 flips\n(pre-flip=T1)\n1 attack resisted": [],
        "T3 flips\n(pre-flip=T2)\n2 attacks resisted": [],
    }
    cohort_colors = ["#4a5568", "#ffd166", "#ff8c42"]

    for (qid, fallacy), group in merged.groupby(["question_id", "fallacy"]):
        group = group.sort_values("turn")
        if group["hallucinated"].iloc[0] != 1:
            continue
        ht = group["hall_turn_num"].iloc[0]
        if pd.isna(ht) or ht < 1:
            continue
        pre_turn = int(ht) - 1
        pre_row  = group[group["turn"] == pre_turn]
        if pre_row.empty:
            continue
        pt = pre_row["prob_target"].values[0]
        pt_val = float(pt) if not pd.isna(pt) else 0.0

        if int(ht) == 1:
            key = "T1 flips\n(pre-flip=T0)\nno attack yet"
        elif int(ht) == 2:
            key = "T2 flips\n(pre-flip=T1)\n1 attack resisted"
        else:
            key = "T3 flips\n(pre-flip=T2)\n2 attacks resisted"
        cohorts[key].append(pt_val)

    # Compute signal rates for annotation
    signal_rates = {}
    for label, vals in cohorts.items():
        if not vals:
            continue
        n_any    = sum(1 for v in vals if v > 0)
        n_strong = sum(1 for v in vals if v > 0.5)
        signal_rates[label] = {
            "n": len(vals), "any": n_any, "strong": n_strong,
            "rate_any":    100 * n_any    / len(vals),
            "rate_strong": 100 * n_strong / len(vals),
        }

    cohort_labels = list(cohorts.keys())
    cohort_data   = [cohorts[k] for k in cohort_labels]

    if all(len(d) > 0 for d in cohort_data):
        parts = ax.violinplot(cohort_data, positions=[1, 2, 3],
                              showmedians=True, showextrema=True)
        for pc, col in zip(parts["bodies"], cohort_colors):
            pc.set_facecolor(col); pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmins"].set_color(SUBTEXT)
        parts["cmaxes"].set_color(SUBTEXT)
        parts["cbars"].set_color(SUBTEXT)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(cohort_labels, fontsize=8.5)

        # Annotate signal rates above each violin
        y_top = 1.08
        for i, (label, col) in enumerate(zip(cohort_labels, cohort_colors), 1):
            sr = signal_rates.get(label, {})
            if sr:
                ax.text(i, y_top,
                        f"Signal: {sr['rate_any']:.1f}%\n"
                        f"Strong: {sr['rate_strong']:.1f}%\n"
                        f"n={sr['n']}",
                        ha="center", va="bottom", fontsize=8.5,
                        color=col if col != "#4a5568" else SUBTEXT,
                        fontweight="bold")

    ax.axhline(0.5, color="white", linewidth=1.2, linestyle=":",
               label="p=0.5 (strong signal)")
    ax.set_ylim(-0.05, 1.45)
    ax.set_ylabel("prob_target at pre-flip turn", fontsize=11)
    ax.set_title(
        "B  |  Pre-Flip prob_target Across All Flip Cohorts\n"
        "T1=0% (expected, no attack yet) → T2=8.6% → T3=20.2%\n"
        "Signal GROWS with each resisted attack turn",
        fontsize=10, fontweight="bold", pad=8
    )
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.35)
    ax.text(0.5, 0.55,
            "↑ Growing signal\n   = gradual erosion\n   in late flips",
            transform=ax.transAxes, fontsize=9, color="#ffd166",
            va="center", ha="center", fontweight="bold")

    # ── Panel C: per-fallacy pre-flip signal rate (T2/T3 flips only) ─────────
    ax = axes[2]
    ax.set_facecolor(PANEL)

    fallacy_signal = {}
    for (qid, fallacy), group in merged.groupby(["question_id", "fallacy"]):
        group = group.sort_values("turn")
        if group["hallucinated"].iloc[0] != 1:
            continue
        ht = group["hall_turn_num"].iloc[0]
        if pd.isna(ht) or ht < 2:
            continue    # only T2/T3 flips
        pre_turn = int(ht) - 1
        pre_row  = group[group["turn"] == pre_turn]
        if pre_row.empty:
            continue
        pt = pre_row["prob_target"].values[0]
        pt_val = float(pt) if not pd.isna(pt) else 0.0
        if fallacy not in fallacy_signal:
            fallacy_signal[fallacy] = {"n": 0, "any": 0, "strong": 0}
        fallacy_signal[fallacy]["n"] += 1
        if pt_val > 0:     fallacy_signal[fallacy]["any"]    += 1
        if pt_val > 0.5:   fallacy_signal[fallacy]["strong"] += 1

    if fallacy_signal:
        sorted_fallacies = sorted(
            fallacy_signal.keys(),
            key=lambda f: fallacy_signal[f]["any"] / max(fallacy_signal[f]["n"], 1),
            reverse=True
        )
        labels_c  = [short_name(f) for f in sorted_fallacies]
        rates_any = [100 * fallacy_signal[f]["any"]    / max(fallacy_signal[f]["n"], 1)
                     for f in sorted_fallacies]
        rates_str = [100 * fallacy_signal[f]["strong"] / max(fallacy_signal[f]["n"], 1)
                     for f in sorted_fallacies]
        ns        = [fallacy_signal[f]["n"] for f in sorted_fallacies]
        colors_f  = [get_color(f) for f in sorted_fallacies]

        x_pos = np.arange(len(sorted_fallacies)); w = 0.38
        ax.bar(x_pos - w/2, rates_any, w, color=colors_f, alpha=0.65,
               edgecolor="none", label="Any signal (pt > 0)")
        ax.bar(x_pos + w/2, rates_str, w, color=colors_f, alpha=1.0,
               edgecolor="white", linewidth=0.5, label="Strong signal (pt > 0.5)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_c, rotation=33, ha="right", fontsize=9)
        for i, (r, n) in enumerate(zip(rates_any, ns)):
            ax.text(i - w/2, r + 0.8, f"{r:.0f}%", ha="center",
                    fontsize=8, color="white", fontweight="bold")
        for i, (r, n) in enumerate(zip(rates_str, ns)):
            if r > 0:
                ax.text(i + w/2, r + 0.8, f"{r:.0f}%", ha="center",
                        fontsize=8, color="white", fontweight="bold")
        # n labels below x-axis
        for i, n in enumerate(ns):
            ax.text(i, -4, f"n={n}", ha="center", fontsize=7.5, color=SUBTEXT)

        ax.set_ylim(-6, max(rates_any) * 1.5 if rates_any else 30)
        ax.set_ylabel("% of T2/T3 flips with pre-flip signal", fontsize=10)

    ax.set_title(
        "C  |  Pre-Flip Signal Rate per Fallacy  (T2/T3 flips only)\n"
        "Light = any signal (pt>0) | Dark = strong signal (pt>0.5)\n"
        "Higher = gradual internal erosion | Lower = sudden flip",
        fontsize=9.5, fontweight="bold", pad=8
    )
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.35)
    ax.text(0.98, 0.97,
            "FALSE_INFO ≈ 0%: always sudden\neven after multiple turns",
            transform=ax.transAxes, fontsize=8, color=SUBTEXT,
            va="top", ha="right", style="italic")

    # ── Panel D: response length trajectory ───────────────────────────────────
    ax = axes[3]
    ax.set_facecolor(PANEL)

    for outcome, color, label in [
        (1, COLOR_HALL, "Hallucinated"),
        (0, COLOR_HELD, "Held Ground"),
    ]:
        sub = merged[merged["hallucinated"] == outcome]
        means, stds, xs = [], [], []
        for t in turns_present:
            vals = sub[sub["turn"] == t]["response_length_tokens"].dropna()
            if len(vals) >= 2:
                means.append(vals.mean()); stds.append(vals.std()); xs.append(t)
        if means:
            xs_ = np.array(xs); means_ = np.array(means); stds_ = np.array(stds)
            ax.plot(xs_, means_, "^-", color=color, linewidth=2.5,
                    markersize=7, label=label, zorder=3)
            ax.fill_between(xs_, means_-stds_, means_+stds_,
                            alpha=0.18, color=color)

    ax.set_xlabel("Turn", fontsize=11); ax.set_xticks(turns_present)
    ax.set_ylabel("Response Length (tokens, mean ± 1 std)", fontsize=11)
    ax.set_title(
        "D  |  Response Length Trajectory by Outcome\n"
        "(shorter = model conceding / longer = model elaborating a defence)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.legend(fontsize=10); ax.grid(alpha=0.35)
    ax.text(0.02, 0.97,
            "If hallucinated conversations get shorter:\nmodel stops defending → flip approaches",
            transform=ax.transAxes, fontsize=8, color=SUBTEXT,
            va="top", style="italic")

    plt.tight_layout(pad=2.0)
    out = out_dir / "fig6_svamp_marginal_stability.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"Saved -> {out}")


# ==============================================================================
# EXPORT — Pre-Flip Signal Cases CSV  ← NEW
# ==============================================================================

def export_pre_flip_signal_csv(prob_df: pd.DataFrame, detail_df: pd.DataFrame,
                               out_dir: Path):
    """
    Export a CSV of every hallucination showing the prob_target value
    at the turn BEFORE the flip — the pre-flip signal analysis.

    Includes ALL flip timings:
      T1 flips: pre-flip turn = T0  (baseline, BEFORE any attack)
      T2 flips: pre-flip turn = T1  (after first attack, model still correct)
      T3 flips: pre-flip turn = T2  (after second attack, model still correct)

    Including T1 flips is critical: their T0 prob_target = 0.0 universally,
    showing that fast flips give NO pre-flip warning signal. Only some T2/T3
    flips show early signal. This contrast is the publishable finding.

    Columns:
      question_id, fallacy, correct_answer, target_wrong_answer
      flip_turn           — turn where hallucination occurred (1, 2, or 3)
      pre_flip_turn       — flip_turn - 1
      answer_at_pre_flip  — what the model said at pre_flip_turn
      model_still_correct — True if answer_at_pre_flip == correct_answer
      prob_stated_pre_flip — model's confidence in its own answer at T-1
      prob_target_pre_flip — probability of the WRONG answer at T-1
                             0.0  = no signal (wrong answer not in top-k)
                             >0.0 = pre-flip signal (wrong answer creeping in)
                             1.0  = strong signal (wrong answer dominant)
      prob_target_at_flip — prob_target at the actual flip turn (usually ~1.0)
      log_ppl_pre_flip    — response-level perplexity at T-1
      signal_strength     — none / weak / strong
    """
    if prob_df.empty or detail_df.empty:
        print("[Skip] No turn_probabilities data — skipping pre_flip_signal_cases.csv")
        return

    for col in ["prob_stated", "prob_target", "log_perplexity"]:
        if col in prob_df.columns:
            prob_df[col] = pd.to_numeric(prob_df[col], errors="coerce")
    prob_df["turn"] = pd.to_numeric(prob_df["turn"], errors="coerce")

    detail_df["hall_turn_num"] = pd.to_numeric(
        detail_df["turns_to_hallucinate"], errors="coerce"
    )
    merged = prob_df.merge(
        detail_df[["question_id", "fallacy", "hallucinated", "hall_turn_num",
                   "correct_answer", "target_wrong_answer"]],
        on=["question_id", "fallacy"], how="left"
    )
    merged["hallucinated"] = pd.to_numeric(
        merged.get("hallucinated", 0), errors="coerce"
    ).fillna(0).astype(int)
    merged["clarification_turn"] = pd.to_numeric(
        merged.get("clarification_turn", 0), errors="coerce"
    ).fillna(0).astype(int)
    merged = merged[merged["clarification_turn"] == 0]

    rows = []
    for (qid, fallacy), group in merged.groupby(["question_id", "fallacy"]):
        group = group.sort_values("turn")
        if group["hallucinated"].iloc[0] != 1:
            continue
        ht = group["hall_turn_num"].iloc[0]
        # Include ALL flip timings: T1, T2, T3
        # For T1 flips: pre-flip turn is T0 (baseline, before any attack)
        # For T2 flips: pre-flip turn is T1 (first attack, model still correct)
        # For T3 flips: pre-flip turn is T2 (second attack, model still correct)
        if pd.isna(ht) or ht < 1:
            continue

        pre_turn = int(ht) - 1
        pre_row  = group[group["turn"] == pre_turn]
        flip_row = group[group["turn"] == int(ht)]
        if pre_row.empty or flip_row.empty:
            continue

        pt_pre    = pre_row["prob_target"].values[0]
        ps_pre    = pre_row["prob_stated"].values[0]
        ans_pre   = pre_row["answer"].values[0]
        pt_flip   = flip_row["prob_target"].values[0]
        ps_flip   = flip_row["prob_stated"].values[0]   # confidence in whatever was written at flip
        ans_flip  = flip_row["answer"].values[0]
        lpl_pre   = pre_row["log_perplexity"].values[0]
        correct   = group["correct_answer"].iloc[0]
        wrong     = group["target_wrong_answer"].iloc[0]

        # If prob_target is NaN treat as 0 (not in top-k)
        pt_pre_val = float(pt_pre) if not pd.isna(pt_pre) else 0.0

        if pt_pre_val >= 0.5:
            strength = "strong"
        elif pt_pre_val > 0.001:
            strength = "weak"
        else:
            strength = "none"

        # Non-targeted hallucination: model flipped but NOT to the attacker's
        # intended wrong answer. It accepted the false framing but computed a
        # different wrong number (e.g. correct=98, target=88, actual=44).
        # prob_target is always null for these cases because the target number
        # never appeared in the top-k — it is a mechanistically different type
        # of hallucination.
        def _normalise(val):
            """
            Normalise a numeric value to a canonical string for comparison.
            Handles: int, float, numpy types, and strings like '23.00' or '23.0'.
            All integer-valued numbers collapse to their int string: '23.0' → '23'.
            True decimals are preserved: '7.5' → '7.5'.
            """
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return ""
            try:
                f = float(str(val).strip())
                if pd.isna(f):
                    return ""
                # If the float is whole-number valued, drop the decimal
                return str(int(f)) if f == int(f) else str(f)
            except (ValueError, OverflowError):
                return str(val).strip()

        ans_flip_n  = _normalise(ans_flip)
        wrong_n     = _normalise(wrong)
        correct_n   = _normalise(correct)
        non_targeted = (
            ans_flip_n not in ("", "not found")
            and ans_flip_n != correct_n
            and ans_flip_n != wrong_n
        )

        rows.append({
            "question_id":                qid,
            "fallacy":                    fallacy,
            "correct_answer":             correct,
            "target_wrong_answer":        wrong,
            "flip_turn":                  int(ht),
            "pre_flip_turn":              pre_turn,
            "answer_at_pre_flip":         ans_pre,
            "answer_at_flip":             ans_flip,
            "model_still_correct":        str(ans_pre) == str(correct),
            "non_targeted_hallucination": non_targeted,
            "prob_stated_pre_flip":       round(float(ps_pre), 6) if not pd.isna(ps_pre) else None,
            "prob_target_pre_flip":       round(pt_pre_val, 6),
            # prob_stated_at_flip: confidence in whatever the model wrote at the flip turn.
            # Works for ALL hallucinations — targeted and non-targeted alike.
            # For targeted: this equals prob_target_at_flip (same token measured twice).
            # For non-targeted: prob_target_at_flip is null (intended target not in top-5)
            #   but prob_stated_at_flip still shows the model was ~100% committed to
            #   its unintended wrong answer.
            "prob_stated_at_flip":        round(float(ps_flip), 6) if not pd.isna(ps_flip) else None,
            # prob_target_at_flip: confidence in the ATTACKER's specific intended target
            # at the flip turn. Null for non-targeted hallucinations.
            "prob_target_at_flip":        round(float(pt_flip), 6) if not pd.isna(pt_flip) else None,
            "log_ppl_pre_flip":           round(float(lpl_pre), 6) if not pd.isna(lpl_pre) else None,
            "signal_strength":            strength,
        })

    if not rows:
        print("[Skip] No T2/T3 hallucinations found — skipping pre_flip_signal_cases.csv")
        return

    out_df = pd.DataFrame(rows).sort_values(
        ["prob_target_pre_flip", "flip_turn"],
        ascending=[False, True]
    ).reset_index(drop=True)

    n_total      = len(out_df)
    n_t1         = (out_df["flip_turn"] == 1).sum()
    n_t23        = (out_df["flip_turn"] >= 2).sum()
    n_strong     = (out_df["signal_strength"] == "strong").sum()
    n_weak       = (out_df["signal_strength"] == "weak").sum()
    n_none       = (out_df["signal_strength"] == "none").sum()
    n_correct    = out_df["model_still_correct"].sum()
    n_nontarget  = out_df["non_targeted_hallucination"].sum()

    out = out_dir / "pre_flip_signal_cases.csv"
    out_df.to_csv(out, index=False)
    print(f"Saved -> {out}")
    print(f"  Total hallucinations with pre-flip data: {n_total}")
    print(f"    T1 flips (pre-flip = T0, baseline): {n_t1}  — all signal=none expected")
    print(f"    T2/T3 flips (pre-flip = T1/T2):    {n_t23}")
    print(f"  Still correct at pre-flip turn: {n_correct}/{n_total}")
    print(f"  Non-targeted hallucinations (flipped to unintended wrong answer): {n_nontarget}/{n_total}")
    print(f"    → prob_target always null for these — mechanistically different")
    print(f"  Signal strength — strong: {n_strong}  weak: {n_weak}  none: {n_none}")
    print(f"  → T1-flips confirm: fast flips give no pre-flip warning at T0")
    print(f"  → Use 'strong' + model_still_correct=True rows as paper examples")


# ==============================================================================
# FIGURE 7 — T2-Flip Composite Predictor  ← NEW
# ==============================================================================

def plot_t2_flip_predictor(prob_df: pd.DataFrame, detail_df: pd.DataFrame,
                           out_dir: Path, subject: str = None, attacker: str = None):
    """
    Figure 7: Four panels examining the composite predictor for T2 hallucinations.

    Context: T1 flips (67% of hallucinations) give zero warning — the model
    looks normal at T0 and flips immediately on first contact.
    T2 flips (22% of hallucinations) give one turn of observable signals:
    at T1 the model answered correctly but generated a response whose features
    predict the impending T2 flip.

    Panel A — ROC curves for each individual feature + composite
    Panel B — Feature distributions: T2-flip vs held-ground at T1
    Panel C — Per-fallacy composite score distribution
    Panel D — Score vs flip probability (calibration)
    """
    if prob_df.empty or detail_df.empty:
        print("[Skip] No turn_probabilities data — skipping fig7.")
        return

    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("[Skip] scikit-learn not installed — skipping fig7. "
              "Install with: pip install scikit-learn")
        return

    subj_label = subject or "Subject"
    atk_label  = attacker or "Attacker"

    # Prepare merged dataframe
    for col in ["log_perplexity", "mean_token_entropy",
                "response_length_tokens", "prob_target", "prob_stated"]:
        if col in prob_df.columns:
            prob_df[col] = pd.to_numeric(prob_df[col], errors="coerce")
    prob_df["turn"] = pd.to_numeric(prob_df["turn"], errors="coerce")

    detail_df["hall_turn_num"] = pd.to_numeric(
        detail_df["turns_to_hallucinate"], errors="coerce"
    )
    merged = prob_df.merge(
        detail_df[["question_id", "fallacy", "hallucinated", "hall_turn_num"]],
        on=["question_id", "fallacy"], how="left"
    )
    merged["hallucinated"] = pd.to_numeric(
        merged.get("hallucinated", 0), errors="coerce"
    ).fillna(0).astype(int)
    merged["clarification_turn"] = pd.to_numeric(
        merged.get("clarification_turn", 0), errors="coerce"
    ).fillna(0).astype(int)
    merged = merged[merged["clarification_turn"] == 0]

    # Build per-conversation T1 feature set
    # Only include: T2-flip conversations AND held-ground conversations
    # (T1 flips excluded — no T1 signal available at T0)
    rows = []
    for (qid, fallacy), group in merged.groupby(["question_id", "fallacy"]):
        group   = group.sort_values("turn")
        hall    = group["hallucinated"].iloc[0]
        ht      = group["hall_turn_num"].iloc[0]
        t0_row  = group[group["turn"] == 0]
        t1_row  = group[group["turn"] == 1]
        if t0_row.empty or t1_row.empty:
            continue

        t0_lpl = t0_row["log_perplexity"].values[0]
        t1_lpl = t1_row["log_perplexity"].values[0]
        t0_len = t0_row["response_length_tokens"].values[0]
        t1_len = t1_row["response_length_tokens"].values[0]
        t0_ent = t0_row["mean_token_entropy"].values[0]
        t1_ent = t1_row["mean_token_entropy"].values[0]
        t1_pt  = t1_row["prob_target"].values[0]

        if any(pd.isna(x) for x in [t0_lpl, t1_lpl, t0_len, t1_len, t0_ent, t1_ent]):
            continue

        flip_at_t2 = int(hall == 1 and not pd.isna(ht) and int(ht) == 2)
        is_held    = int(hall == 0)
        if flip_at_t2 == 0 and is_held == 0:
            continue    # skip T1/T3 flips

        rows.append({
            "question_id": qid, "fallacy": fallacy,
            "flip_at_t2":  flip_at_t2,
            "lpl_delta":   t1_lpl - t0_lpl,
            "ent_delta":   t1_ent - t0_ent,
            "len_ratio":   t1_len / max(t0_len, 1),
            "pt_t1":       float(t1_pt) if not pd.isna(t1_pt) else 0.0,
            "lpl_t1":      t1_lpl,
        })

    if len(rows) < 20:
        print("[Skip] Not enough T2-flip / held-ground pairs for fig7.")
        return

    df7 = pd.DataFrame(rows)
    y   = df7["flip_at_t2"].values
    n_flip = y.sum(); n_held = len(y) - n_flip

    # Feature definitions: (column, display_name, color)
    features = [
        ("lpl_delta",  "log-Ppl rise\n(T1−T0)",    "#ff4c6a"),
        ("ent_delta",  "Entropy rise\n(T1−T0)",     "#ff8c42"),
        ("len_ratio",  "Length ratio\n(T1/T0)",     "#ffd166"),
        ("pt_t1",      "prob_target\nat T1",         "#a855f7"),
    ]

    # Build composite score (MinMax normalise each feature then sum)
    feat_cols = [f[0] for f in features]
    scaler    = MinMaxScaler()
    X_scaled  = scaler.fit_transform(df7[feat_cols].fillna(0))
    df7["composite"] = X_scaled.sum(axis=1)

    # ── Compute AUCs ─────────────────────────────────────────────────────────
    aucs = {}
    for col, name, _ in features:
        try:
            aucs[col] = roc_auc_score(y, df7[col].fillna(0))
        except Exception:
            aucs[col] = 0.5
    aucs["composite"] = roc_auc_score(y, df7["composite"])

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), facecolor=BG)
    fig.suptitle(
        f"Fig 7  |  T2-Flip Composite Predictor — Can We Detect Imminent Hallucination?\n"
        f"Conditioned on: model answered correctly at T1, first attack already fired\n"
        f"{subj_label}  vs  {atk_label}   "
        f"(n_flip={n_flip}, n_held={n_held})",
        fontsize=13, fontweight="bold", color=TEXT, y=1.02
    )
    axes = axes.flatten()

    # ── Panel A: ROC curves ───────────────────────────────────────────────────
    ax = axes[0]; ax.set_facecolor(PANEL)
    feat_colors = {f[0]: f[2] for f in features}
    feat_names  = {f[0]: f[1].replace("\n", " ") for f in features}

    for col, name, color in features:
        fpr, tpr, _ = roc_curve(y, df7[col].fillna(0))
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name.replace(chr(10),' ')}  AUC={aucs[col]:.3f}")

    # Composite
    fpr_c, tpr_c, thresholds_c = roc_curve(y, df7["composite"])
    ax.plot(fpr_c, tpr_c, color="white", linewidth=2.5, linestyle="-",
            label=f"Composite (all 4)  AUC={aucs['composite']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=SUBTEXT, linewidth=1, label="Random (AUC=0.5)")

    ax.fill_between(fpr_c, tpr_c, alpha=0.08, color="white")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        "A  |  ROC Curves — Individual vs Composite Predictor\n"
        "(predicting T2 hallucination from T1 response features)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.35)
    ax.text(0.05, 0.92,
            f"Best composite AUC: {aucs['composite']:.3f}\n"
            f"Random baseline:    0.500\n"
            f"Improvement:       +{aucs['composite']-0.5:.3f}",
            transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL2, alpha=0.8))

    # ── Panel B: Feature distributions ────────────────────────────────────────
    ax = axes[1]; ax.set_facecolor(PANEL)
    flip_rows = df7[df7["flip_at_t2"] == 1]
    held_rows = df7[df7["flip_at_t2"] == 0]

    x_pos = np.arange(len(features)); w = 0.35
    flip_means = [flip_rows[f[0]].mean() for f in features]
    held_means = [held_rows[f[0]].mean() for f in features]
    flip_stds  = [flip_rows[f[0]].std()  for f in features]
    held_stds  = [held_rows[f[0]].std()  for f in features]

    ax.bar(x_pos - w/2, flip_means, w, color=COLOR_HALL, alpha=0.85,
           edgecolor="none", label=f"T2-flip (n={n_flip})")
    ax.bar(x_pos + w/2, held_means, w, color=COLOR_HELD, alpha=0.85,
           edgecolor="none", label=f"Held-ground (n={n_held})")
    ax.errorbar(x_pos - w/2, flip_means, yerr=flip_stds,
                fmt="none", color="white", capsize=4, linewidth=1.5)
    ax.errorbar(x_pos + w/2, held_means, yerr=held_stds,
                fmt="none", color="white", capsize=4, linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f[1] for f in features], fontsize=10)

    # Annotate significance stars
    from scipy import stats as scipy_stats
    for i, (col, _, _) in enumerate(features):
        _, pval = scipy_stats.mannwhitneyu(
            flip_rows[col].dropna(), held_rows[col].dropna(), alternative="two-sided"
        )
        stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        y_top = max(flip_means[i], held_means[i]) + max(flip_stds[i], held_stds[i]) + 0.02
        ax.text(i, y_top, stars, ha="center", fontsize=11, color="#ffd166", fontweight="bold")

    ax.set_ylabel("Feature value (mean ± 1 std)", fontsize=11)
    ax.set_title(
        "B  |  T1 Feature Values: T2-Flip vs Held-Ground\n"
        "(*=p<0.05, **=p<0.01, ***=p<0.001)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.35)

    # ── Panel C: Composite score by fallacy ───────────────────────────────────
    ax = axes[2]; ax.set_facecolor(PANEL)
    fallacies_present = sorted(df7["fallacy"].unique())
    pos = 1
    positions, tick_labels = [], []

    for fallacy in fallacies_present:
        sub_flip = df7[(df7["fallacy"] == fallacy) & (df7["flip_at_t2"] == 1)]["composite"]
        sub_held = df7[(df7["fallacy"] == fallacy) & (df7["flip_at_t2"] == 0)]["composite"]
        color    = get_color(fallacy)

        if len(sub_flip) > 0:
            ax.scatter([pos - 0.2] * len(sub_flip), sub_flip,
                       alpha=0.5, s=20, color=COLOR_HALL, zorder=3)
            ax.plot([pos - 0.35, pos - 0.05],
                    [sub_flip.mean(), sub_flip.mean()],
                    color=COLOR_HALL, linewidth=2.5, zorder=4)
        if len(sub_held) > 0:
            ax.scatter([pos + 0.2] * len(sub_held), sub_held,
                       alpha=0.3, s=15, color=COLOR_HELD, zorder=3)
            ax.plot([pos + 0.05, pos + 0.35],
                    [sub_held.mean(), sub_held.mean()],
                    color=COLOR_HELD, linewidth=2.5, zorder=4)

        tick_labels.append(short_name(fallacy))
        positions.append(pos)
        pos += 1

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=33, ha="right", fontsize=9)
    ax.set_ylabel("Composite score (higher = more likely to flip)", fontsize=10)
    ax.set_title(
        "C  |  Composite Score per Fallacy\n"
        "Red dots = T2-flip | Blue dots = held-ground | Lines = means",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.grid(axis="y", alpha=0.35)
    # Custom legend
    ax.plot([], [], color=COLOR_HALL, linewidth=3, label="T2-flip mean")
    ax.plot([], [], color=COLOR_HELD, linewidth=3, label="Held-ground mean")
    ax.legend(fontsize=9)

    # ── Panel D: Score calibration — P(flip) vs composite score ──────────────
    ax = axes[3]; ax.set_facecolor(PANEL)

    # Bin composite score into deciles and compute flip rate per bin
    df7["score_bin"] = pd.qcut(df7["composite"], q=8, labels=False, duplicates="drop")
    calib = df7.groupby("score_bin", observed=True).agg(
        flip_rate=("flip_at_t2", "mean"),
        mean_score=("composite", "mean"),
        n=("flip_at_t2", "count")
    ).reset_index()

    ax.scatter(calib["mean_score"], calib["flip_rate"] * 100,
               s=calib["n"] * 3, color=ACCENT, alpha=0.9,
               edgecolors="white", linewidth=0.8, zorder=3)
    # Trend line
    if len(calib) >= 3:
        z = np.polyfit(calib["mean_score"], calib["flip_rate"] * 100, 1)
        p = np.poly1d(z)
        xs = np.linspace(calib["mean_score"].min(), calib["mean_score"].max(), 100)
        ax.plot(xs, p(xs), color="#ffd166", linewidth=2, linestyle="--",
                label=f"Linear fit (slope={z[0]:.1f})")

    # Base rate line
    base_rate = y.mean() * 100
    ax.axhline(base_rate, color=SUBTEXT, linewidth=1.5, linestyle=":",
               label=f"Base rate: {base_rate:.1f}%")

    # Annotate dots with n
    for _, row in calib.iterrows():
        ax.annotate(f"n={int(row['n'])}",
                    (row["mean_score"], row["flip_rate"] * 100),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=SUBTEXT)

    ax.set_xlabel("Composite predictor score", fontsize=11)
    ax.set_ylabel("Empirical P(flip at T2) %", fontsize=11)
    ax.set_title(
        "D  |  Score Calibration: Does Higher Score → Higher Flip Probability?\n"
        "(dot size = n conversations in bin)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.35)
    ax.text(0.02, 0.97,
            f"Base rate: {base_rate:.1f}%\n"
            f"Highest bin rate: {calib['flip_rate'].max()*100:.1f}%\n"
            f"Composite AUC: {aucs['composite']:.3f}",
            transform=ax.transAxes, fontsize=9, color=TEXT,
            va="top", fontweight="bold")

    plt.tight_layout(pad=2.0)
    out = out_dir / "fig7_t2_flip_predictor.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved -> {out}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args        = parse_args()
    results_dir = Path(args.results)
    total_q     = args.total
    subject     = args.subject
    attacker    = args.attacker

    if args.summary:
        results_dir = Path(args.summary).parent
        df = pd.read_csv(args.summary)
        df = df.sort_values("success_rate_pct", ascending=False).reset_index(drop=True)
        if "rank" not in df.columns:
            df["rank"] = range(1, len(df) + 1)
    else:
        if not results_dir.exists():
            raise FileNotFoundError(
                f"Results folder '{results_dir}' not found.\n"
                "Run gsm_qa.py first or pass --results <path>."
            )
        df = load_summary_csvs(results_dir)

    for col in ("total_no_answer", "total_pre_pressure_fail",
                "total_hallucinated", "total_held_ground", "total_eligible"):
        if col not in df.columns:
            df[col] = 0

    token_stats = load_summary_txt(results_dir)
    detail_df   = load_detail_csv(results_dir)
    prob_df     = load_turn_probabilities(results_dir)
    ate_df      = load_answer_token_entropy(results_dir)

    # Merge answer_token_entropy into prob_df if available
    if not ate_df.empty and not prob_df.empty:
        prob_df = prob_df.merge(ate_df, on=["question_id", "fallacy", "turn"], how="left")
        n_ate = prob_df["answer_token_entropy"].notna().sum()
        print(f"answer_token_entropy merged: {n_ate} turns with valid values")
    elif "answer_token_entropy" not in prob_df.columns:
        prob_df["answer_token_entropy"] = np.nan

    n_runs  = len(token_stats.get("runs", []))
    q_done  = token_stats.get("questions_done", 0)
    print(f"\nPartial runs found: {n_runs}")
    print(f"Questions completed: {q_done} / {total_q} ({100*q_done/total_q:.0f}%)")
    print(f"Fallacies: {df['fallacy'].tolist()}")
    print(f"Token total: {token_stats.get('total_tokens', 0):,}")
    print(f"Turn-prob rows loaded: {len(prob_df)}\n")

    out_dir = results_dir
    out_dir.mkdir(exist_ok=True)

    plot_dashboard(df, token_stats, out_dir,
                   total_target=total_q, subject=subject, attacker=attacker)
    plot_table(df, token_stats, out_dir,
               total_target=total_q, subject=subject, attacker=attacker)
    plot_deepdive(df, detail_df, out_dir)
    plot_tokens(token_stats, df, out_dir, subject=subject, attacker=attacker)
    plot_logprob_analysis(prob_df, detail_df, df, out_dir,
                          subject=subject, attacker=attacker)
    plot_marginal_stability(prob_df, detail_df, df, out_dir,
                            subject=subject, attacker=attacker)
    plot_t2_flip_predictor(prob_df, detail_df, out_dir,
                           subject=subject, attacker=attacker)
    export_pre_flip_signal_csv(prob_df, detail_df, out_dir)

    print(f"\nAll figures saved to: {out_dir.resolve()}")
    print("  fig1_svamp_dashboard.png")
    print("  fig2_svamp_table.png")
    print("  fig3_svamp_deepdive.png")
    print("  fig4_svamp_tokens.png")
    print("  fig5_svamp_logprob_analysis.png")
    print("  fig6_svamp_marginal_stability.png")
    print("  fig7_t2_flip_predictor.png         ← NEW")
    print("  pre_flip_signal_cases.csv")


if __name__ == "__main__":
    main()