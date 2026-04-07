import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import warnings
from pathlib import Path
Path("results").mkdir(exist_ok=True)
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# DATA  (aggregated from both runs: 100q + 587q = 687q total)
# ──────────────────────────────────────────────────────────────────────────────

FALLACIES = [
    "FALSE_INFORMATION",
    "STRAW_MAN",
    "PURE_PRESSURE",
    "APPEAL_TO_AUTHORITY",
    "BANDWAGON",
    "SLIPPERY_SLOPE",
    "AD_HOMINEM",
]

SHORT = {
    "FALSE_INFORMATION":   "False Info",
    "STRAW_MAN":           "Straw Man",
    "PURE_PRESSURE":       "Pure Pressure",
    "APPEAL_TO_AUTHORITY": "Appeal to Auth.",
    "BANDWAGON":           "Bandwagon",
    "SLIPPERY_SLOPE":      "Slippery Slope",
    "AD_HOMINEM":          "Ad Hominem",
}

# Run 1 (100 questions)
R1 = {
    "FALSE_INFORMATION":   dict(hall=23, held=65, pre=12, eligible=88,  t1=12, t2=9,  t3=2,  avg_win=1.57),
    "STRAW_MAN":           dict(hall=6,  held=87, pre=7,  eligible=93,  t1=4,  t2=1,  t3=1,  avg_win=1.50),
    "PURE_PRESSURE":       dict(hall=0,  held=92, pre=8,  eligible=92,  t1=0,  t2=0,  t3=0,  avg_win=None),
    "APPEAL_TO_AUTHORITY": dict(hall=1,  held=92, pre=7,  eligible=93,  t1=0,  t2=0,  t3=1,  avg_win=3.00),
    "BANDWAGON":           dict(hall=0,  held=91, pre=9,  eligible=91,  t1=0,  t2=0,  t3=0,  avg_win=None),
    "SLIPPERY_SLOPE":      dict(hall=2,  held=88, pre=10, eligible=90,  t1=2,  t2=0,  t3=0,  avg_win=1.00),
    "AD_HOMINEM":          dict(hall=0,  held=91, pre=9,  eligible=91,  t1=0,  t2=0,  t3=0,  avg_win=None),
}

# Run 2 (587 questions)
R2 = {
    "FALSE_INFORMATION":   dict(hall=91, held=460, pre=36, eligible=551, t1=59, t2=23, t3=9,  avg_win=1.45),
    "STRAW_MAN":           dict(hall=18, held=534, pre=35, eligible=552, t1=17, t2=0,  t3=1,  avg_win=1.11),
    "PURE_PRESSURE":       dict(hall=13, held=541, pre=33, eligible=554, t1=9,  t2=4,  t3=0,  avg_win=1.31),
    "APPEAL_TO_AUTHORITY": dict(hall=5,  held=548, pre=34, eligible=553, t1=1,  t2=1,  t3=3,  avg_win=2.40),
    "BANDWAGON":           dict(hall=3,  held=547, pre=37, eligible=550, t1=2,  t2=1,  t3=0,  avg_win=1.33),
    "SLIPPERY_SLOPE":      dict(hall=1,  held=552, pre=34, eligible=553, t1=1,  t2=0,  t3=0,  avg_win=1.00),
    "AD_HOMINEM":          dict(hall=1,  held=547, pre=39, eligible=548, t1=1,  t2=0,  t3=0,  avg_win=1.00),
}

# Combined
COMBINED = {}
for f in FALLACIES:
    r1, r2 = R1[f], R2[f]
    total_hall = r1["hall"] + r2["hall"]
    total_elig = r1["eligible"] + r2["eligible"]
    COMBINED[f] = dict(
        hall     = total_hall,
        held     = r1["held"] + r2["held"],
        pre      = r1["pre"]  + r2["pre"],
        eligible = total_elig,
        rate     = 100 * total_hall / total_elig if total_elig else 0,
        t1       = r1["t1"] + r2["t1"],
        t2       = r1["t2"] + r2["t2"],
        t3       = r1["t3"] + r2["t3"],
    )

# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────

BG       = "#0f1117"
PANEL    = "#1a1d27"
PANEL2   = "#20243a"
BORDER   = "#2e3250"
TEXT     = "#e8eaf6"
SUBTEXT  = "#8892b0"
ACCENT   = "#7c83ff"

PALETTE = {
    "FALSE_INFORMATION":   "#ff4c6a",
    "STRAW_MAN":           "#ff8c42",
    "PURE_PRESSURE":       "#ffd166",
    "APPEAL_TO_AUTHORITY": "#06d6a0",
    "BANDWAGON":           "#4cc9f0",
    "SLIPPERY_SLOPE":      "#a855f7",
    "AD_HOMINEM":          "#64748b",
}

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

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  Main Dashboard  (3×2 grid)
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 22), facecolor=BG)
fig.suptitle(
    "Fallacy Pressure Test  ·  GPT-4o-mini Subject  ·  GPT-4o Attacker  ·  687 Questions",
    fontsize=18, fontweight="bold", color=TEXT, y=0.98
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35,
                       left=0.08, right=0.96, top=0.94, bottom=0.04)

colors_ordered = [PALETTE[f] for f in FALLACIES]
labels_short   = [SHORT[f] for f in FALLACIES]
rates          = [COMBINED[f]["rate"] for f in FALLACIES]

# ── Panel A: Success Rate Bar Chart ──────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :])

bars = ax_a.barh(
    labels_short, rates,
    color=colors_ordered, edgecolor="none", height=0.62
)
ax_a.set_xlabel("Hallucination Rate (%)", fontsize=11, color=TEXT)
ax_a.set_title("A  ·  Hallucination Success Rate by Fallacy  (combined 687 questions, 3 max turns)",
               fontsize=12, fontweight="bold", color=TEXT, pad=10)
ax_a.invert_yaxis()
ax_a.set_xlim(0, 22)
ax_a.axvline(0, color=BORDER, lw=1)
ax_a.grid(axis="x", alpha=0.4)
ax_a.set_facecolor(PANEL)

for bar, rate, f in zip(bars, rates, FALLACIES):
    n   = COMBINED[f]["hall"]
    elig = COMBINED[f]["eligible"]
    ax_a.text(rate + 0.25, bar.get_y() + bar.get_height()/2,
              f"  {rate:.1f}%  ({n}/{elig})",
              va="center", ha="left", fontsize=10, color=TEXT, fontweight="bold")

# Annotate run comparison for FALSE_INFORMATION
ax_a.text(17.84, 0, "Run 1: 26.1%  →  Run 2: 16.5%",
          va="center", ha="left", fontsize=8.5, color="#ffb3bc", style="italic")

# ── Panel B: Turn Distribution Stacked Bar ────────────────────────────────────
ax_b = fig.add_subplot(gs[1, 0])

t1_vals = [COMBINED[f]["t1"] for f in FALLACIES]
t2_vals = [COMBINED[f]["t2"] for f in FALLACIES]
t3_vals = [COMBINED[f]["t3"] for f in FALLACIES]

x = np.arange(len(FALLACIES))
w = 0.55

b1 = ax_b.bar(x, t1_vals, w, label="Turn 1", color="#7c83ff", edgecolor="none")
b2 = ax_b.bar(x, t2_vals, w, bottom=t1_vals, label="Turn 2", color="#4cc9f0", edgecolor="none")
b3 = ax_b.bar(x, t3_vals, w,
              bottom=[a+b for a,b in zip(t1_vals, t2_vals)],
              label="Turn 3", color="#ff8c42", edgecolor="none")

ax_b.set_xticks(x)
ax_b.set_xticklabels(labels_short, rotation=33, ha="right", fontsize=9)
ax_b.set_ylabel("# Hallucinations", fontsize=10)
ax_b.set_title("B  ·  Turn at Which Hallucination Occurred", fontsize=11, fontweight="bold", pad=8)
ax_b.legend(fontsize=9, loc="upper right")
ax_b.grid(axis="y", alpha=0.4)
ax_b.set_facecolor(PANEL)
ax_b.yaxis.set_major_locator(MaxNLocator(integer=True))

# value labels on bars
for i, (v1, v2, v3) in enumerate(zip(t1_vals, t2_vals, t3_vals)):
    total = v1+v2+v3
    if total > 0:
        ax_b.text(i, total + 0.4, str(total), ha="center", va="bottom",
                  fontsize=9, color=TEXT, fontweight="bold")

# ── Panel C: Outcome Breakdown ────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 1])

hall_vals = [COMBINED[f]["hall"] for f in FALLACIES]
held_vals = [COMBINED[f]["held"] for f in FALLACIES]
pre_vals  = [COMBINED[f]["pre"]  for f in FALLACIES]
totals    = [h+he+p for h,he,p in zip(hall_vals, held_vals, pre_vals)]

hall_pct = [100*h/t for h,t in zip(hall_vals, totals)]
held_pct = [100*h/t for h,t in zip(held_vals, totals)]
pre_pct  = [100*p/t for p,t in zip(pre_vals,  totals)]

ax_c.barh(labels_short, pre_pct,  color="#374151", edgecolor="none", label="Pre-pressure fail (T0 wrong)")
ax_c.barh(labels_short, held_pct, left=pre_pct, color="#1e3a5f", edgecolor="none", label="Held Ground")
ax_c.barh(labels_short, hall_pct,
          left=[a+b for a,b in zip(pre_pct, held_pct)],
          color=[PALETTE[f] for f in FALLACIES], edgecolor="none", label="Hallucinated (pressure-induced)")

ax_c.set_xlim(0, 100)
ax_c.set_xlabel("% of all conversations", fontsize=10)
ax_c.set_title("C  ·  Conversation Outcome Breakdown", fontsize=11, fontweight="bold", pad=8)
ax_c.invert_yaxis()
ax_c.legend(fontsize=8.5, loc="lower right")
ax_c.grid(axis="x", alpha=0.4)
ax_c.set_facecolor(PANEL)

# ── Panel D: Run-over-Run Comparison ─────────────────────────────────────────
ax_d = fig.add_subplot(gs[2, 0])

r1_rates = [100 * R1[f]["hall"] / R1[f]["eligible"] if R1[f]["eligible"] else 0 for f in FALLACIES]
r2_rates = [100 * R2[f]["hall"] / R2[f]["eligible"] if R2[f]["eligible"] else 0 for f in FALLACIES]

x2  = np.arange(len(FALLACIES))
w2  = 0.38

ax_d.bar(x2 - w2/2, r1_rates, w2, label="Run 1 (100 Q)", color="#7c83ff", edgecolor="none", alpha=0.9)
ax_d.bar(x2 + w2/2, r2_rates, w2, label="Run 2 (587 Q)", color="#ff4c6a", edgecolor="none", alpha=0.9)

ax_d.set_xticks(x2)
ax_d.set_xticklabels(labels_short, rotation=33, ha="right", fontsize=9)
ax_d.set_ylabel("Hallucination Rate (%)", fontsize=10)
ax_d.set_title("D  ·  Run 1 vs Run 2 Success Rate Comparison", fontsize=11, fontweight="bold", pad=8)
ax_d.legend(fontsize=9)
ax_d.grid(axis="y", alpha=0.4)
ax_d.set_facecolor(PANEL)

# ── Panel E: Radar Chart ──────────────────────────────────────────────────────
ax_e = fig.add_subplot(gs[2, 1], polar=True)

N = len(FALLACIES)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

values = [COMBINED[f]["rate"] for f in FALLACIES]
values_norm = [v / max(values) for v in values]
values_norm += values_norm[:1]

ax_e.set_facecolor(PANEL)
ax_e.figure.patch.set_facecolor(BG)

ax_e.plot(angles, values_norm, color="#ff4c6a", linewidth=2.2, linestyle="solid")
ax_e.fill(angles, values_norm, color="#ff4c6a", alpha=0.22)

ax_e.set_xticks(angles[:-1])
ax_e.set_xticklabels(labels_short, size=9, color=TEXT)
ax_e.set_yticklabels([])
ax_e.spines["polar"].set_color(BORDER)
ax_e.grid(color=BORDER, linewidth=0.8)
ax_e.set_title("E  ·  Fallacy Effectiveness Radar\n(normalized to max)",
               fontsize=11, fontweight="bold", pad=18, color=TEXT)

for angle, val, f in zip(angles[:-1], values_norm[:-1], FALLACIES):
    ax_e.annotate(
        f"{COMBINED[f]['rate']:.1f}%",
        xy=(angle, val),
        xytext=(angle, val + 0.08),
        ha="center", va="center",
        fontsize=7.5, color=TEXT, fontweight="bold"
    )

plt.savefig("results/fig1_dashboard.png",
            dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved fig1_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  Summary Table
# ══════════════════════════════════════════════════════════════════════════════

fig2, ax = plt.subplots(figsize=(18, 5.5), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")

fig2.suptitle("Fallacy Efficiency Summary Table  —  687 Questions  (100 + 587)  ·  GPT-4o-mini vs GPT-4o",
              fontsize=14, fontweight="bold", color=TEXT, y=1.01)

cols = ["Rank", "Fallacy", "Success\nRate %",
        "Avg Turns\n(Wins)", "Avg Turns\n(All)",
        "T1 Wins", "T2 Wins", "T3 Wins",
        "Total\nHall.", "Total\nHeld", "Pre-\nFail", "Eligible"]

# Rank by combined success rate
ranked = sorted(FALLACIES, key=lambda f: -COMBINED[f]["rate"])

# avg_turns_win: weighted average of the two runs
def avg_win(f):
    w1 = R1[f]["hall"]; w2 = R2[f]["hall"]
    a1 = R1[f]["avg_win"]; a2 = R2[f]["avg_win"]
    total = w1 + w2
    if total == 0: return None
    parts = []
    if w1 and a1: parts.append(w1 * a1)
    if w2 and a2: parts.append(w2 * a2)
    return sum(parts) / total if parts else None

def avg_all(f):
    # approx: (sum of turns for wins + (eligible-wins)*(max_turns+1)) / eligible
    max_t = 3
    c = COMBINED[f]
    win_turns = (COMBINED[f]["t1"]*1 + COMBINED[f]["t2"]*2 + COMBINED[f]["t3"]*3)
    dnh = c["eligible"] - c["hall"]
    return (win_turns + dnh * (max_t + 1)) / c["eligible"] if c["eligible"] else None

rows = []
for rank, f in enumerate(ranked, 1):
    c   = COMBINED[f]
    aw  = avg_win(f)
    aa  = avg_all(f)
    rows.append([
        str(rank),
        SHORT[f],
        f"{c['rate']:.2f}%",
        f"{aw:.2f}" if aw else "—",
        f"{aa:.2f}" if aa else "—",
        str(c["t1"]),
        str(c["t2"]),
        str(c["t3"]),
        str(c["hall"]),
        str(c["held"]),
        str(c["pre"]),
        str(c["eligible"]),
    ])

table = ax.table(
    cellText  = rows,
    colLabels = cols,
    cellLoc   = "center",
    loc       = "center",
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.6)

# Style header
for j in range(len(cols)):
    cell = table[0, j]
    cell.set_facecolor(ACCENT)
    cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
    cell.set_edgecolor(BG)

# Style rows
for i, (rank, f) in enumerate(zip(range(1, len(ranked)+1), ranked), 1):
    row_bg = "#1e2235" if i % 2 == 0 else PANEL2
    for j in range(len(cols)):
        cell = table[i, j]
        cell.set_facecolor(row_bg)
        cell.set_text_props(color=TEXT)
        cell.set_edgecolor(BG)
    # Highlight success rate cell
    rate_cell = table[i, 2]
    rate_cell.set_text_props(color=PALETTE[f], fontweight="bold")
    # Color rank
    table[i, 0].set_text_props(color=PALETTE[f], fontweight="bold")

plt.tight_layout()
plt.savefig("results/fig2_summary_table.png",
            dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved fig2_summary_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  —  Deep-dive: FALSE_INFORMATION vs the rest
# ══════════════════════════════════════════════════════════════════════════════

fig3, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig3.suptitle("Deep-Dive: FALSE_INFORMATION Dominance", fontsize=15,
              fontweight="bold", color=TEXT, y=1.02)

# ── 3A: Pie of all hallucinations by fallacy ─────────────────────────────────
ax3a = axes[0]
hall_counts = [COMBINED[f]["hall"] for f in FALLACIES]
total_hall  = sum(hall_counts)

pie_colors  = [PALETTE[f] for f in FALLACIES]
wedge_props = dict(width=0.55, edgecolor=BG, linewidth=2)

wedges, texts, autotexts = ax3a.pie(
    hall_counts,
    labels    = None,
    colors    = pie_colors,
    autopct   = lambda p: f"{p:.1f}%" if p > 1.5 else "",
    startangle= 140,
    wedgeprops= wedge_props,
    pctdistance=0.75,
)
for at in autotexts:
    at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")

ax3a.set_facecolor(BG)
ax3a.set_title(f"A  ·  Share of All Hallucinations\n(total = {total_hall})",
               fontsize=11, fontweight="bold", color=TEXT, pad=12)

legend_handles = [
    mpatches.Patch(facecolor=PALETTE[f], label=f"{SHORT[f]} ({COMBINED[f]['hall']})")
    for f in FALLACIES
]
ax3a.legend(handles=legend_handles, loc="lower left",
            fontsize=8, framealpha=0.7, bbox_to_anchor=(-0.35, -0.18))

# ── 3B: Rate comparison: FALSE_INFO vs all others grouped ────────────────────
ax3b = axes[1]
ax3b.set_facecolor(PANEL)

fi_rate    = COMBINED["FALSE_INFORMATION"]["rate"]
other_f    = [f for f in FALLACIES if f != "FALSE_INFORMATION"]
other_rates= [COMBINED[f]["rate"] for f in other_f]
other_avg  = np.mean(other_rates)

bar_labels = ["FALSE\nINFORMATION", "Others\n(avg)"]
bar_vals   = [fi_rate, other_avg]
bar_colors = [PALETTE["FALSE_INFORMATION"], "#4a5568"]

b = ax3b.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="none",
             width=0.45)
for bar, val in zip(b, bar_vals):
    ax3b.text(bar.get_x() + bar.get_width()/2, val + 0.25,
              f"{val:.2f}%", ha="center", va="bottom",
              fontsize=13, fontweight="bold", color=TEXT)

ax3b.set_ylabel("Hallucination Rate (%)", fontsize=10)
ax3b.set_title("B  ·  FALSE_INFORMATION vs\nAverage of All Others",
               fontsize=11, fontweight="bold", pad=8)
ax3b.grid(axis="y", alpha=0.4)
ratio = fi_rate / other_avg
ax3b.text(0.5, max(bar_vals)*0.55,
          f"{ratio:.1f}×\nmore effective",
          ha="center", fontsize=12, color="#ffd166",
          fontweight="bold")

# ── 3C: Turn-1 capture rate comparison ───────────────────────────────────────
ax3c = axes[2]
ax3c.set_facecolor(PANEL)

t1_rates = []
for f in FALLACIES:
    c = COMBINED[f]
    t1_rates.append(100 * c["t1"] / c["eligible"] if c["eligible"] else 0)

bars_c = ax3c.barh(
    labels_short, t1_rates,
    color=[PALETTE[f] for f in FALLACIES],
    edgecolor="none", height=0.6
)
ax3c.invert_yaxis()
ax3c.set_xlabel("% of eligible questions fooled at Turn 1", fontsize=10)
ax3c.set_title("C  ·  Turn-1 Capture Rate\n(hallucination on first attack)",
               fontsize=11, fontweight="bold", pad=8)
ax3c.grid(axis="x", alpha=0.4)

for bar, val, f in zip(bars_c, t1_rates, FALLACIES):
    n = COMBINED[f]["t1"]
    ax3c.text(val + 0.1, bar.get_y() + bar.get_height()/2,
              f"  {val:.1f}% ({n})", va="center", fontsize=9, color=TEXT)

plt.tight_layout(pad=1.5)
plt.savefig("results/fig3_deepdive.png",
            dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved fig3_deepdive.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4  —  Token Cost & Efficiency
# ══════════════════════════════════════════════════════════════════════════════

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig4.suptitle("Token Cost Analysis  (combined runs)", fontsize=14,
              fontweight="bold", color=TEXT, y=1.02)

# ── 4A: Token usage breakdown ────────────────────────────────────────────────
ax4a = axes4[0]
ax4a.set_facecolor(PANEL)

runs = ["Run 1\n(100 Q)", "Run 2\n(587 Q)", "Combined\n(687 Q)"]
gpt4o_tokens     = [2_118_071, 12_854_182, 14_972_253]
gpt4o_mini_tokens= [2_179_600, 13_059_320, 15_238_920]

x4 = np.arange(len(runs))
w4 = 0.38
ax4a.bar(x4 - w4/2, [t/1e6 for t in gpt4o_tokens],     w4,
         label="GPT-4o (Attacker)",   color="#7c83ff", edgecolor="none")
ax4a.bar(x4 + w4/2, [t/1e6 for t in gpt4o_mini_tokens], w4,
         label="GPT-4o-mini (Subject)", color="#ff4c6a", edgecolor="none")

ax4a.set_xticks(x4); ax4a.set_xticklabels(runs, fontsize=10)
ax4a.set_ylabel("Tokens (millions)", fontsize=10)
ax4a.set_title("A  ·  Token Usage by Model & Run", fontsize=11, fontweight="bold", pad=8)
ax4a.legend(fontsize=9)
ax4a.grid(axis="y", alpha=0.4)

for bars, vals in [
    (ax4a.containers[0], gpt4o_tokens),
    (ax4a.containers[1], gpt4o_mini_tokens)
]:
    for bar, val in zip(bars, vals):
        ax4a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.1,
                  f"{val/1e6:.1f}M",
                  ha="center", va="bottom", fontsize=8.5, color=TEXT)

# ── 4B: Cost per hallucination ────────────────────────────────────────────────
ax4b = axes4[1]
ax4b.set_facecolor(PANEL)

# Tokens per hallucination: total session tokens / total hallucinations
# Run 1: 4.3M tokens, 32 hallucinations  → 134k tokens/hallucination
# Run 2: 25.9M tokens, 132 hallucinations → 196k tokens/hallucination
# Combined: 30.2M tokens, 164 hallucinations → 184k tokens/hallucination

runs2  = ["Run 1\n(100 Q)", "Run 2\n(587 Q)", "Combined"]
tph    = [4_297_671/32, 25_913_502/132, (4_297_671+25_913_502)/(32+132)]
colors4b = ["#7c83ff", "#ff4c6a", "#ffd166"]

bars4b = ax4b.bar(runs2, [t/1000 for t in tph], color=colors4b, edgecolor="none", width=0.45)
for bar, val in zip(bars4b, tph):
    ax4b.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.5,
              f"{val/1000:.0f}K\ntokens",
              ha="center", va="bottom", fontsize=10, color=TEXT, fontweight="bold")

ax4b.set_ylabel("Tokens per Hallucination (thousands)", fontsize=10)
ax4b.set_title("B  ·  Tokens Spent per Successful Hallucination", fontsize=11, fontweight="bold", pad=8)
ax4b.grid(axis="y", alpha=0.4)

plt.tight_layout(pad=1.5)
plt.savefig("results/fig4_token_cost.png",
            dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved fig4_token_cost.png")

print("\nAll figures saved successfully.")