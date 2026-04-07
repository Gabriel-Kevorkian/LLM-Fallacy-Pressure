"""
GSM8K Fallacy Pressure Test — Visualizer
==========================================
Reads results automatically from the results/ folder.
Supports single run OR multiple runs (auto-merges them).

USAGE:
    python visualize_svamp.py
    python visualize_svamp.py --results results/
    python visualize_svamp.py --summary results/gsm8k_..._summary.csv

Outputs 4 PNG figures into results/:
    fig1_gsm8k_dashboard.png
    fig2_gsm8k_table.png
    fig3_gsm8k_deepdive.png
    fig4_gsm8k_tokens.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
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
    p = argparse.ArgumentParser(description="GSM8K Fallacy Visualizer")
    p.add_argument("--results", type=str, default="results",
                   help="Folder containing GSM8K result files (default: results/)")
    p.add_argument("--summary", type=str, default=None,
                   help="Path to a specific _fallacy_summary.csv file (overrides --results)")
    return p.parse_args()


# ==============================================================================
# DATA LOADER  — reads and merges all gsm8k_*_fallacy_summary.csv files
# ==============================================================================

def load_summary_csvs(results_dir: Path) -> pd.DataFrame:
    """Load and merge all GSM8K fallacy summary CSVs in the results folder."""
    csvs = sorted(results_dir.glob("gsm8k_*_fallacy_summary.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No gsm8k_*_fallacy_summary.csv files found in '{results_dir}'.\n"
            "Run gsm8k.py first to generate results."
        )
    print(f"Found {len(csvs)} summary file(s):")
    for c in csvs:
        print(f"  {c}")

    frames = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        frames.append(df)

    if len(frames) == 1:
        return frames[0]

    # Merge multiple runs by summing counts and re-deriving rates
    combined = pd.concat(frames)
    # Columns to sum
    sum_cols = [
        "total_hallucinated", "total_held_ground",
        "total_pre_pressure_fail", "total_eligible",
    ]
    # Turn columns (hallucinated_at_T1, T2, T3 ...)
    turn_cols = [c for c in combined.columns if re.match(r"hallucinated_at_T\d+", c)]
    sum_cols += turn_cols

    # Also handle total_error if present
    if "total_error" in combined.columns:
        sum_cols.append("total_error")

    agg = combined.groupby("fallacy")[sum_cols].sum().reset_index()

    # Re-derive success_rate_pct
    agg["success_rate_pct"] = (
        agg["total_hallucinated"] / agg["total_eligible"].replace(0, np.nan) * 100
    ).fillna(0).round(2)

    # Re-derive avg_turns_when_successful from turn distribution
    def _avg_win(row):
        wins = 0
        weighted = 0
        for t_col in turn_cols:
            t = int(re.search(r"\d+", t_col).group())
            n = row[t_col]
            wins += n
            weighted += t * n
        return round(weighted / wins, 2) if wins > 0 else None

    agg["avg_turns_when_successful"] = agg.apply(_avg_win, axis=1)

    # Re-rank by success rate
    agg = agg.sort_values("success_rate_pct", ascending=False).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)

    return agg


def load_summary_txt(results_dir: Path) -> dict:
    """
    Parse token usage and overall stats from the *_summary.txt files.
    Returns a dict with merged stats across all found files.
    """
    txts = sorted(results_dir.glob("gsm8k_*_summary.txt"))
    if not txts:
        return {}

    total_tokens    = 0
    gpt4o_tokens    = 0
    mini_tokens     = 0
    total_calls     = 0
    total_hall      = 0
    total_conv      = 0
    total_pre       = 0
    runs            = []

    for txt_path in txts:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")

        # Session total tokens
        m = re.search(r"Session total\s*:\s*([\d,]+)", text)
        if m:
            t = int(m.group(1).replace(",", ""))
            total_tokens += t
            runs.append({"file": txt_path.name, "tokens": t})

        # Per-model tokens
        m4o = re.search(r"gpt-4o\s+total=([\d,]+)", text)
        if m4o:
            gpt4o_tokens += int(m4o.group(1).replace(",", ""))
        mmini = re.search(r"gpt-4o-mini\s+total=([\d,]+)", text)
        if mmini:
            mini_tokens += int(mmini.group(1).replace(",", ""))

        # Calls
        mc = re.search(r"calls=(\d+)\)", text)
        if mc:
            total_calls += int(mc.group(1))

        # Pressure-induced hallucinations
        mh = re.search(r"Pressure-induced hallucinations[:\s]+([\d]+)", text)
        if mh:
            total_hall += int(mh.group(1))

        # Total conversations
        mt = re.search(r"Total fallacy conversations[:\s]+([\d]+)", text)
        if mt:
            total_conv += int(mt.group(1))

        # Pre-pressure fails
        mp = re.search(r"Pre-pressure fails[^:]*:\s*([\d]+)", text)
        if mp:
            total_pre += int(mp.group(1))

    return {
        "total_tokens":   total_tokens,
        "gpt4o_tokens":   gpt4o_tokens,
        "mini_tokens":    mini_tokens,
        "total_calls":    total_calls,
        "total_hall":     total_hall,
        "total_conv":     total_conv,
        "total_pre":      total_pre,
        "runs":           runs,
    }


def load_detail_csv(results_dir: Path) -> pd.DataFrame:
    """Load and merge all question_detail CSVs if available."""
    csvs = sorted(results_dir.glob("gsm8k_*_question_detail.csv"))
    if not csvs:
        return pd.DataFrame()
    frames = [pd.read_csv(c) for c in csvs]
    return pd.concat(frames, ignore_index=True)


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

# Fallacy color palette — expanded for GSM8K's 7 fallacies
PALETTE = {
    "FALSE_INFORMATION":   "#ff4c6a",
    "APPEAL_TO_AUTHORITY": "#ff8c42",
    "AD_HOMINEM":          "#ffd166",
    "FALSE_DILEMMA":       "#06d6a0",
    "CAUSAL_FALLACY":      "#4cc9f0",
    "CIRCULAR_REASONING":  "#a855f7",
    "PURE_PRESSURE":       "#64748b",
}

DEFAULT_COLOR = "#94a3b8"

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
    }.get(fallacy, fallacy)


# ==============================================================================
# FIGURE 1 — Main Dashboard
# ==============================================================================

def plot_dashboard(df: pd.DataFrame, token_stats: dict, out_dir: Path):
    fallacies     = df["fallacy"].tolist()
    labels        = [short_name(f) for f in fallacies]
    colors        = [get_color(f) for f in fallacies]
    rates         = df["success_rate_pct"].tolist()
    total_q       = int(df["total_eligible"].max()) if not df.empty else 0

    # Turn columns
    turn_cols = sorted(
        [c for c in df.columns if re.match(r"hallucinated_at_T\d+", c)],
        key=lambda x: int(re.search(r"\d+", x).group())
    )

    fig = plt.figure(figsize=(20, 22), facecolor=BG)
    n_runs = len(token_stats.get("runs", []))
    run_label = f"{n_runs} run(s)" if n_runs else ""
    fig.suptitle(
        f"GSM8K Fallacy Pressure Test  |  GPT-4o-mini Subject  |  GPT-4o Attacker"
        + (f"  |  {run_label}" if run_label else ""),
        fontsize=17, fontweight="bold", color=TEXT, y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.44, wspace=0.35,
                           left=0.08, right=0.96, top=0.94, bottom=0.04)

    # ── Panel A: Success Rate Bar Chart ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    bars = ax_a.barh(labels, rates, color=colors, edgecolor="none", height=0.62)
    ax_a.set_xlabel("Hallucination Rate (%)", fontsize=11, color=TEXT)
    ax_a.set_title(
        "A  |  Hallucination Success Rate by Fallacy  (all questions, 3 max turns)",
        fontsize=12, fontweight="bold", color=TEXT, pad=10
    )
    ax_a.invert_yaxis()
    max_rate = max(rates) if rates else 10
    ax_a.set_xlim(0, max_rate * 1.35)
    ax_a.axvline(0, color=BORDER, lw=1)
    ax_a.grid(axis="x", alpha=0.4)
    ax_a.set_facecolor(PANEL)

    for bar, rate, f in zip(bars, rates, fallacies):
        n    = int(df.loc[df["fallacy"] == f, "total_hallucinated"].values[0])
        elig = int(df.loc[df["fallacy"] == f, "total_eligible"].values[0])
        ax_a.text(
            rate + max_rate * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"  {rate:.1f}%  ({n}/{elig})",
            va="center", ha="left", fontsize=10, color=TEXT, fontweight="bold"
        )

    # ── Panel B: Turn Distribution Stacked Bar ────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    turn_colors = ["#7c83ff", "#4cc9f0", "#ff8c42", "#ff4c6a", "#a855f7"]
    x = np.arange(len(fallacies))
    w = 0.55
    bottoms = np.zeros(len(fallacies))

    for i, t_col in enumerate(turn_cols):
        vals = df[t_col].fillna(0).tolist()
        t_num = re.search(r"\d+", t_col).group()
        ax_b.bar(x, vals, w, bottom=bottoms,
                 label=f"Turn {t_num}",
                 color=turn_colors[i % len(turn_colors)], edgecolor="none")
        bottoms += np.array(vals)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=33, ha="right", fontsize=9)
    ax_b.set_ylabel("# Hallucinations", fontsize=10)
    ax_b.set_title("B  |  Turn at Which Hallucination Occurred",
                   fontsize=11, fontweight="bold", pad=8)
    ax_b.legend(fontsize=9, loc="upper right")
    ax_b.grid(axis="y", alpha=0.4)
    ax_b.set_facecolor(PANEL)
    ax_b.yaxis.set_major_locator(MaxNLocator(integer=True))

    for i, total in enumerate(bottoms):
        if total > 0:
            ax_b.text(i, total + 0.3, str(int(total)),
                      ha="center", va="bottom", fontsize=9,
                      color=TEXT, fontweight="bold")

    # ── Panel C: Outcome Breakdown ────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    hall_vals = df["total_hallucinated"].fillna(0).tolist()
    held_vals = df["total_held_ground"].fillna(0).tolist()
    pre_vals  = df["total_pre_pressure_fail"].fillna(0).tolist()
    totals    = [h + he + p for h, he, p in zip(hall_vals, held_vals, pre_vals)]

    def pct(v, t): return [100 * a / b if b else 0 for a, b in zip(v, t)]

    pre_p  = pct(pre_vals,  totals)
    held_p = pct(held_vals, totals)
    hall_p = pct(hall_vals, totals)

    ax_c.barh(labels, pre_p,  color="#374151", edgecolor="none",
              label="Pre-pressure fail (T0 wrong)")
    ax_c.barh(labels, held_p, left=pre_p, color="#1e3a5f", edgecolor="none",
              label="Held Ground")
    ax_c.barh(labels, hall_p, left=[a + b for a, b in zip(pre_p, held_p)],
              color=colors, edgecolor="none",
              label="Hallucinated (pressure-induced)")
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("% of all conversations", fontsize=10)
    ax_c.set_title("C  |  Conversation Outcome Breakdown",
                   fontsize=11, fontweight="bold", pad=8)
    ax_c.invert_yaxis()
    ax_c.legend(fontsize=8.5, loc="lower right")
    ax_c.grid(axis="x", alpha=0.4)
    ax_c.set_facecolor(PANEL)

    # ── Panel D: Avg Turns Comparison ────────────────────────────────────────
    ax_d = fig.add_subplot(gs[2, 0])
    ax_d.set_facecolor(PANEL)

    avg_win_vals = pd.to_numeric(
        df["avg_turns_when_successful"], errors="coerce"
    ).tolist()
    avg_all_vals = pd.to_numeric(
        df["avg_turns_overall"], errors="coerce"
    ).tolist()

    x2 = np.arange(len(fallacies))
    w2 = 0.38
    ax_d.bar(x2 - w2/2,
             [v if not np.isnan(v) else 0 for v in avg_win_vals],
             w2, label="Avg turns (wins only)",
             color="#7c83ff", edgecolor="none", alpha=0.9)
    ax_d.bar(x2 + w2/2,
             [v if not np.isnan(v) else 0 for v in avg_all_vals],
             w2, label="Avg turns (all incl. DNH)",
             color="#ff4c6a", edgecolor="none", alpha=0.9)

    ax_d.set_xticks(x2)
    ax_d.set_xticklabels(labels, rotation=33, ha="right", fontsize=9)
    ax_d.set_ylabel("Avg Turns", fontsize=10)
    ax_d.set_title("D  |  Average Turns to Hallucinate",
                   fontsize=11, fontweight="bold", pad=8)
    ax_d.legend(fontsize=9)
    ax_d.grid(axis="y", alpha=0.4)

    # ── Panel E: Radar Chart ─────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[2, 1], polar=True)
    N = len(fallacies)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    max_r = max(rates) if max(rates) > 0 else 1
    values_norm = [r / max_r for r in rates]
    values_norm += values_norm[:1]

    ax_e.set_facecolor(PANEL)
    ax_e.plot(angles, values_norm, color="#ff4c6a", linewidth=2.2)
    ax_e.fill(angles, values_norm, color="#ff4c6a", alpha=0.22)
    ax_e.set_xticks(angles[:-1])
    ax_e.set_xticklabels(labels, size=9, color=TEXT)
    ax_e.set_yticklabels([])
    ax_e.spines["polar"].set_color(BORDER)
    ax_e.grid(color=BORDER, linewidth=0.8)
    ax_e.set_title("E  |  Effectiveness Radar\n(normalized to max)",
                   fontsize=11, fontweight="bold", pad=18, color=TEXT)

    for angle, val, rate in zip(angles[:-1], values_norm[:-1], rates):
        ax_e.annotate(
            f"{rate:.1f}%",
            xy=(angle, val),
            xytext=(angle, val + 0.09),
            ha="center", va="center",
            fontsize=7.5, color=TEXT, fontweight="bold"
        )

    out = out_dir / "fig1_svamp_dashboard.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved -> {out}")


# ==============================================================================
# FIGURE 2 — Summary Table
# ==============================================================================

def plot_table(df: pd.DataFrame, token_stats: dict, out_dir: Path):
    fig2, ax = plt.subplots(figsize=(20, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    total_q = int(df["total_eligible"].max()) if not df.empty else "?"
    fig2.suptitle(
        f"GSM8K Fallacy Efficiency Summary Table  |  "
        f"GPT-4o-mini vs GPT-4o",
        fontsize=14, fontweight="bold", color=TEXT, y=1.01
    )

    # Turn columns
    turn_cols = sorted(
        [c for c in df.columns if re.match(r"hallucinated_at_T\d+", c)],
        key=lambda x: int(re.search(r"\d+", x).group())
    )
    t_headers = [f"T{re.search(r'[0-9]+', c).group()} Wins" for c in turn_cols]

    cols = (["Rank", "Fallacy", "Success\nRate %",
              "Avg Turns\n(Wins)", "Avg Turns\n(All)"]
            + t_headers
            + ["Total\nHall.", "Total\nHeld", "Pre-\nFail", "Eligible"])

    rows = []
    for _, row in df.sort_values("rank").iterrows():
        r_vals = [
            str(int(row.get("rank", ""))),
            short_name(row["fallacy"]),
            f"{row['success_rate_pct']:.2f}%",
            f"{row['avg_turns_when_successful']:.2f}"
            if pd.notna(row.get("avg_turns_when_successful")) else "—",
            f"{row['avg_turns_overall']:.2f}"
            if pd.notna(row.get("avg_turns_overall")) else "—",
        ]
        for t_col in turn_cols:
            r_vals.append(str(int(row.get(t_col, 0))))
        r_vals += [
            str(int(row.get("total_hallucinated", 0))),
            str(int(row.get("total_held_ground", 0))),
            str(int(row.get("total_pre_pressure_fail", 0))),
            str(int(row.get("total_eligible", 0))),
        ]
        rows.append(r_vals)

    table = ax.table(
        cellText  = rows,
        colLabels = cols,
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.5)

    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor(BG)

    for i, (_, row) in enumerate(df.sort_values("rank").iterrows(), 1):
        f      = row["fallacy"]
        row_bg = "#1e2235" if i % 2 == 0 else PANEL2
        for j in range(len(cols)):
            cell = table[i, j]
            cell.set_facecolor(row_bg)
            cell.set_text_props(color=TEXT)
            cell.set_edgecolor(BG)
        table[i, 2].set_text_props(color=get_color(f), fontweight="bold")
        table[i, 0].set_text_props(color=get_color(f), fontweight="bold")

    plt.tight_layout()
    out = out_dir / "fig2_svamp_table.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved -> {out}")


# ==============================================================================
# FIGURE 3 — Deep-Dive
# ==============================================================================

def plot_deepdive(df: pd.DataFrame, detail_df: pd.DataFrame, out_dir: Path):
    fallacies = df["fallacy"].tolist()
    labels    = [short_name(f) for f in fallacies]
    colors    = [get_color(f) for f in fallacies]
    rates     = df["success_rate_pct"].tolist()
    hall_vals = df["total_hallucinated"].fillna(0).tolist()

    fig3, axes = plt.subplots(1, 3, figsize=(19, 6.5), facecolor=BG)
    fig3.suptitle("GSM8K Deep-Dive: Top Fallacy Analysis",
                  fontsize=15, fontweight="bold", color=TEXT, y=1.02)

    # ── 3A: Donut — share of all hallucinations ───────────────────────────────
    ax3a = axes[0]
    total_hall = sum(hall_vals)
    wedge_props = dict(width=0.55, edgecolor=BG, linewidth=2)

    wedges, texts, autotexts = ax3a.pie(
        hall_vals,
        labels      = None,
        colors      = colors,
        autopct     = lambda p: f"{p:.1f}%" if p > 1.5 else "",
        startangle  = 140,
        wedgeprops  = wedge_props,
        pctdistance = 0.75,
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
    ax3a.set_facecolor(BG)
    ax3a.set_title(f"A  |  Share of All Hallucinations\n(total = {int(total_hall)})",
                   fontsize=11, fontweight="bold", color=TEXT, pad=12)
    legend_handles = [
        mpatches.Patch(facecolor=get_color(f),
                       label=f"{short_name(f)} ({int(n)})")
        for f, n in zip(fallacies, hall_vals)
    ]
    ax3a.legend(handles=legend_handles, loc="lower left",
                fontsize=8, framealpha=0.7, bbox_to_anchor=(-0.3, -0.22))

    # ── 3B: Top fallacy vs others ─────────────────────────────────────────────
    ax3b = axes[1]
    ax3b.set_facecolor(PANEL)

    if rates:
        top_f    = fallacies[0]
        top_rate = rates[0]
        other_rates = rates[1:]
        other_avg   = np.mean(other_rates) if other_rates else 0

        bar_labels = [short_name(top_f), "Others\n(avg)"]
        bar_vals   = [top_rate, other_avg]
        bar_colors = [get_color(top_f), "#4a5568"]

        b = ax3b.bar(bar_labels, bar_vals, color=bar_colors,
                     edgecolor="none", width=0.45)
        for bar, val in zip(b, bar_vals):
            ax3b.text(bar.get_x() + bar.get_width() / 2,
                      val + max(bar_vals) * 0.02,
                      f"{val:.2f}%",
                      ha="center", va="bottom",
                      fontsize=13, fontweight="bold", color=TEXT)

        if other_avg > 0:
            ratio = top_rate / other_avg
            ax3b.text(0.5, max(bar_vals) * 0.5,
                      f"{ratio:.1f}x\nmore effective",
                      ha="center", fontsize=12, color="#ffd166",
                      fontweight="bold")

    ax3b.set_ylabel("Hallucination Rate (%)", fontsize=10)
    ax3b.set_title(f"B  |  Top Fallacy vs\nAverage of All Others",
                   fontsize=11, fontweight="bold", pad=8)
    ax3b.grid(axis="y", alpha=0.4)

    # ── 3C: Turn-1 capture rate ───────────────────────────────────────────────
    ax3c = axes[2]
    ax3c.set_facecolor(PANEL)

    turn_cols = sorted(
        [c for c in df.columns if re.match(r"hallucinated_at_T\d+", c)],
        key=lambda x: int(re.search(r"\d+", x).group())
    )
    eligible_vals = df["total_eligible"].fillna(1).tolist()

    if turn_cols:
        t1_col   = turn_cols[0]
        t1_vals  = df[t1_col].fillna(0).tolist()
        t1_rates = [100 * t1 / elig if elig else 0
                    for t1, elig in zip(t1_vals, eligible_vals)]

        bars_c = ax3c.barh(labels, t1_rates, color=colors,
                           edgecolor="none", height=0.6)
        ax3c.invert_yaxis()
        ax3c.set_xlabel("% of eligible questions fooled at Turn 1", fontsize=10)
        ax3c.set_title("C  |  Turn-1 Capture Rate\n(hallucination on first attack)",
                       fontsize=11, fontweight="bold", pad=8)
        ax3c.grid(axis="x", alpha=0.4)

        max_t1 = max(t1_rates) if t1_rates else 1
        for bar, val, t1_n in zip(bars_c, t1_rates, t1_vals):
            ax3c.text(val + max_t1 * 0.01,
                      bar.get_y() + bar.get_height() / 2,
                      f"  {val:.1f}% ({int(t1_n)})",
                      va="center", fontsize=9, color=TEXT)

    plt.tight_layout(pad=1.5)
    out = out_dir / "fig3_svamp_deepdive.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved -> {out}")


# ==============================================================================
# FIGURE 4 — Token Cost
# ==============================================================================

def plot_tokens(token_stats: dict, df: pd.DataFrame, out_dir: Path):
    if not token_stats or token_stats.get("total_tokens", 0) == 0:
        print("[Skip] No token data found in summary.txt files -- skipping fig4.")
        return

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig4.suptitle("Token Cost Analysis  (all runs combined)",
                  fontsize=14, fontweight="bold", color=TEXT, y=1.02)

    # ── 4A: Attacker vs Subject tokens ───────────────────────────────────────
    ax4a = axes4[0]
    ax4a.set_facecolor(PANEL)

    gpt4o = token_stats.get("gpt4o_tokens", 0)
    mini  = token_stats.get("mini_tokens", 0)
    total = token_stats.get("total_tokens", 0)
    # If per-model breakdown not available, show total only
    if gpt4o == 0 and mini == 0:
        ax4a.bar(["Total"], [total / 1e6], color=ACCENT, edgecolor="none", width=0.4)
        ax4a.text(0, total / 1e6 + 0.1, f"{total/1e6:.1f}M",
                  ha="center", va="bottom", fontsize=11, color=TEXT, fontweight="bold")
    else:
        runs_labels  = ["GPT-4o\n(Attacker)", "GPT-4o-mini\n(Subject)", "Combined"]
        runs_vals    = [gpt4o / 1e6, mini / 1e6, total / 1e6]
        run_colors   = ["#7c83ff", "#ff4c6a", "#ffd166"]
        b = ax4a.bar(runs_labels, runs_vals, color=run_colors, edgecolor="none", width=0.45)
        for bar, val in zip(b, runs_vals):
            ax4a.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.1,
                      f"{val:.1f}M",
                      ha="center", va="bottom", fontsize=10, color=TEXT, fontweight="bold")

    ax4a.set_ylabel("Tokens (millions)", fontsize=10)
    ax4a.set_title("A  |  Token Usage by Model", fontsize=11, fontweight="bold", pad=8)
    ax4a.grid(axis="y", alpha=0.4)

    # ── 4B: Tokens per hallucination ─────────────────────────────────────────
    ax4b = axes4[1]
    ax4b.set_facecolor(PANEL)

    total_hall = token_stats.get("total_hall", 0)
    total_conv = token_stats.get("total_conv", 0)
    total_pre  = token_stats.get("total_pre", 0)

    if total_hall > 0 and total > 0:
        tph = total / total_hall
        # Also compute tokens per conversation
        tpc = total / total_conv if total_conv > 0 else 0

        metric_labels = ["Tokens per\nHallucination", "Tokens per\nConversation"]
        metric_vals   = [tph / 1000, tpc / 1000]
        metric_colors = ["#ff4c6a", "#7c83ff"]

        b2 = ax4b.bar(metric_labels, metric_vals, color=metric_colors,
                      edgecolor="none", width=0.4)
        for bar, val in zip(b2, metric_vals):
            ax4b.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.5,
                      f"{val:.0f}K\ntokens",
                      ha="center", va="bottom", fontsize=11, color=TEXT, fontweight="bold")

        ax4b.set_ylabel("Tokens (thousands)", fontsize=10)
        ax4b.set_title("B  |  Token Efficiency Metrics",
                       fontsize=11, fontweight="bold", pad=8)
        ax4b.grid(axis="y", alpha=0.4)

        # Add annotation
        ax4b.text(0.5, 0.05,
                  f"Total: {total/1e6:.1f}M tokens\n"
                  f"{int(total_hall)} hallucinations  |  "
                  f"{int(total_conv)} conversations",
                  transform=ax4b.transAxes,
                  ha="center", fontsize=9, color=SUBTEXT)
    else:
        ax4b.text(0.5, 0.5, "Token-per-hallucination data\nnot available yet",
                  transform=ax4b.transAxes, ha="center", va="center",
                  fontsize=12, color=SUBTEXT)

    plt.tight_layout(pad=1.5)
    out = out_dir / "fig4_svamp_tokens.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved -> {out}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args        = parse_args()
    results_dir = Path(args.results)

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
                "Run gsm8k.py first or pass --results <path>."
            )
        df = load_summary_csvs(results_dir)

    token_stats = load_summary_txt(results_dir)
    detail_df   = load_detail_csv(results_dir)

    print(f"\nFallacies found: {df['fallacy'].tolist()}")
    print(f"Total questions (max eligible): {int(df['total_eligible'].max())}")
    print(f"Token stats: {token_stats.get('total_tokens', 0):,} total tokens\n")

    # Output to same results folder
    out_dir = results_dir
    out_dir.mkdir(exist_ok=True)

    plot_dashboard(df, token_stats, out_dir)
    plot_table(df, token_stats, out_dir)
    plot_deepdive(df, detail_df, out_dir)
    plot_tokens(token_stats, df, out_dir)

    print(f"\nAll figures saved to: {out_dir.resolve()}")
    print("  fig1_gsm8k_dashboard.png")
    print("  fig2_gsm8k_table.png")
    print("  fig3_gsm8k_deepdive.png")
    print("  fig4_gsm8k_tokens.png")


if __name__ == "__main__":
    main()