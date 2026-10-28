"""fig1_flow.svg - STROBE-style cohort flow, counts read from wp1_strobe_funnel.csv."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from paths import OUT

f = pd.read_csv(OUT / "wp1_strobe_funnel.csv")
n = {r.step.strip(): (int(r.n), int(r.patients)) for r in f.itertuples()}
def fmt(k):
    a, b = n[k]; return f"{a:,} / {b:,}"

main = [
    ("MIMIC-IV-ECG v1.0\n" + fmt("ECGs in MIMIC-IV-ECG v1.0"), None),
    ("Serum potassium within ±60 min\n" + fmt("serum K within +-60 min"), "no paired K draw"),
    ("Age ≥ 18\n" + fmt("age >= 18"), "1 ECG under 18"),
    ("No ventricular pacing\n" + fmt("no ventricular pacing"), "4,797 paced ECGs"),
    ("Non-haemolysed specimen\n" + fmt("non-haemolysed specimen"), "699 haemolysed\n(sensitivity stratum)"),
    ("Signal QC passed (§7.1)\n" + fmt("signal QC passed (7.1)"), "938 QC failures"),
]
pair = [
    ("Consecutive within-subject ECG–K pairs\n" + fmt("Consecutive within-subject ECG-K pairs"), None),
    ("Distinct laboratory draws\n" + fmt("distinct laboratory draws"), "12,163 shared-K artefact"),
    ("Δt ≤ 24 h\n" + fmt("dt <= 24 h"), "107,161 pairs Δt > 24 h"),
    ("No CPR / cardioversion between\nCOHORT_DELTA — " + fmt("no CPR/cardioversion -> COHORT_DELTA"),
     "269 acute cardiac events"),
]
leaf = ["COHORT_NULL_CHANGE\n|ΔK| < 0.15, no dialysis\n" + fmt("COHORT_NULL_CHANGE") + " patients",
        "COHORT_HD_PAIRS\nHD between the ECGs\n" + fmt("COHORT_HD_PAIRS") + " patients",
        "COHORT_CROSSSECTIONAL\naudit patients removed\n" + fmt("COHORT_CROSSSECTIONAL") + " patients"]

fig, ax = plt.subplots(figsize=(9.6, 12))
ax.set_xlim(0, 10); ax.set_ylim(0, 20.6); ax.axis("off")

def box(x, y, w, h, txt, fc="#ebf4ff", ec="#2b6cb0", fs=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)

y = 19.3
seq = main + pair
for i, (txt, excl) in enumerate(seq):
    box(2.5, y, 4.9, 1.02, txt)
    if excl:
        box(7.7, y + 0.06, 2.2, 0.9, "excluded\n" + excl, fc="#fff5f5", ec="#c53030", fs=7.0)
        ax.add_patch(FancyArrowPatch((7.4, y + 0.51), (7.7, y + 0.51), arrowstyle="-|>",
                                     mutation_scale=9, color="#c53030", lw=1.0))
    if i < len(seq) - 1:
        ax.add_patch(FancyArrowPatch((4.95, y), (4.95, y - 0.42), arrowstyle="-|>",
                                     mutation_scale=11, color="#4a5568", lw=1.1))
    if i == len(main) - 1:
        ax.text(4.95, y - 0.30, "pair construction", ha="center", fontsize=7.5,
                style="italic", color="#4a5568")
    y -= 1.44

y += 1.44
for k, t in enumerate(leaf):
    x = 0.25 + k * 3.35
    ax.add_patch(FancyArrowPatch((4.95, y), (x + 1.5, y - 1.02), arrowstyle="-|>",
                                 mutation_scale=10, color="#4a5568", lw=1.0))
    box(x, y - 2.05, 3.0, 1.0, t, fc="#f0fff4", ec="#2f855a", fs=7.4)

ax.text(4.95, 20.3, "HD-DELTA — cohort flow (fallback design)", ha="center",
        fontsize=11.5, weight="bold")
ax.text(4.95, 19.98, "MIMIC-IV v3.1 + MIMIC-IV-ECG v1.0 · potassium", ha="center",
        fontsize=8.5, color="#4a5568")
fig.tight_layout(); fig.savefig(OUT / "figS1_flow.svg"); plt.close(fig)
print("wrote outputs/fig1_flow.svg")
