"""
Draw the SCoPE pipeline figure referenced as figure/Overall.png.

Two bands, matching Algorithm 1 in the manuscript:
  Pass 1 (construction) -- the four-step cycle repeated once per stage,
      with the environment transition closing the loop.
  Pass 2 (repair)       -- score / flip / re-simulate / keep-best, repeated
      K times on the assembled schedule.

Grayscale-safe: fills are light grays, emphasis is by border weight, not hue.
Journals still print this page in black and white.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure", "Overall.png")

FILL_LEARNED = "#dcdcdc"   # learned component
FILL_FIXED = "#f4f4f4"     # non-learned component (auction, environment)
EDGE = "#222222"

# Figure coordinate system: 10 wide, 6.4 tall.
fig, ax = plt.subplots(figsize=(9.0, 5.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.4)
ax.axis("off")


def box(x, y, w, h, title, body, fill, lw=1.3, fontsize=8.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=lw, edgecolor=EDGE, facecolor=fill, zorder=3,
        )
    )
    ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center",
            fontsize=fontsize + 0.6, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h / 2 - 0.20, body, ha="center", va="center",
            fontsize=fontsize, linespacing=1.35, zorder=4)


def arrow(x1, y1, x2, y2, style="-|>", rad=0.0, lw=1.3, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=13, linewidth=lw, linestyle=ls,
            color=EDGE, connectionstyle=f"arc3,rad={rad}", zorder=6,
        )
    )


def feedback(x_from, y_box_edge, x_to, y_rail, ls=(0, (4, 3)), lw=1.2):
    """Loop-back routed through empty space rather than across the boxes:
    drop out of the last box, run along a clear rail, rise into the first."""
    ax.plot([x_from, x_from], [y_box_edge, y_rail],
            color=EDGE, linewidth=lw, linestyle=ls, zorder=6)
    ax.plot([x_from, x_to], [y_rail, y_rail],
            color=EDGE, linewidth=lw, linestyle=ls, zorder=6)
    arrow(x_to, y_rail, x_to, y_box_edge, lw=lw, ls=ls)


def band(x, y, w, h, label):
    ax.add_patch(
        Rectangle((x, y), w, h, linewidth=0.9, edgecolor="#999999",
                  facecolor="none", linestyle=(0, (5, 4)), zorder=1)
    )
    ax.text(x + 0.12, y + h - 0.20, label, ha="left", va="center",
            fontsize=9.0, fontweight="bold", color="#444444", zorder=4)


# ----------------------------------------------------------------- Pass 1
band(0.10, 3.22, 9.80, 3.08, "Pass 1  Construction:  one forward pass per stage,  t = 1 ... T")

BY, BH, BW = 4.34, 1.30, 2.22
xs = [0.36, 2.78, 5.20, 7.62]

box(xs[0], BY, BW, BH, "Encode",
    "heterogeneous\nweapon-target graph\nL message-passing rounds", FILL_LEARNED)
box(xs[1], BY, BW, BH, "Communicate",
    "self-attention across\nthe M weapons", FILL_LEARNED)
box(xs[2], BY, BW, BH, "Commit in parallel",
    "M pointer policies,\none forward pass:\nfire or hold", FILL_LEARNED)
box(xs[3], BY, BW, BH, "Auction",
    "assign targets among\nthe firing weapons", FILL_FIXED)

for i in range(3):
    arrow(xs[i] + BW, BY + BH / 2, xs[i + 1], BY + BH / 2)

# Environment transition closing the construction loop, routed below the row.
feedback(xs[3] + BW / 2, BY, xs[0] + BW / 2, 3.86)
ax.text(5.0, 3.52, "environment step:  apply hits,  spend ammunition,  "
                   "start reload clocks,  advance time windows",
        ha="center", va="center", fontsize=8.0, style="italic", color="#333333")

# Legend, kept clear of the row and of the band title.
ax.text(6.30, 6.03, "learned", fontsize=7.8, color="#333333", ha="left", va="center")
ax.add_patch(Rectangle((7.02, 5.94), 0.32, 0.18, facecolor=FILL_LEARNED,
                       edgecolor=EDGE, linewidth=0.9, zorder=3))
ax.text(7.62, 6.03, "not learned", fontsize=7.8, color="#333333", ha="left", va="center")
ax.add_patch(Rectangle((8.72, 5.94), 0.32, 0.18, facecolor=FILL_FIXED,
                       edgecolor=EDGE, linewidth=0.9, zorder=3))

# -------------------------------------------------- hand-off between passes
arrow(5.00, 3.22, 5.00, 2.86, lw=1.6)
ax.text(5.16, 3.04, "complete, feasible schedule", ha="left", va="center",
        fontsize=8.4, style="italic")

# ----------------------------------------------------------------- Pass 2
band(0.10, 0.16, 9.80, 2.66, "Pass 2  Learned repair:  K edits,  best schedule retained")

RY, RH, RW = 0.62, 1.30, 2.22
rxs = [0.36, 2.78, 5.20, 7.62]

box(rxs[0], RY, RW, RH, "Score slots",
    "repair policy rates\nall T x M\nstage-weapon slots", FILL_LEARNED)
box(rxs[1], RY, RW, RH, "Flip one slot",
    "highest-scoring slot\nswitches fire and hold", FILL_LEARNED)
box(rxs[2], RY, RW, RH, "Re-simulate",
    "replay the episode;\nfeasibility recomputed,\nnot patched", FILL_FIXED)
box(rxs[3], RY, RW, RH, "Keep best",
    "return the best\nschedule seen;\nnever worse than the\nconstructed one", FILL_FIXED)

for i in range(3):
    arrow(rxs[i] + RW, RY + RH / 2, rxs[i + 1], RY + RH / 2)

feedback(rxs[3] + RW / 2, RY + RH, rxs[0] + RW / 2, 2.22)
ax.text(5.0, 2.42, "repeat until the edit budget K is spent", ha="center",
        va="center", fontsize=8.0, style="italic", color="#333333")

fig.tight_layout(pad=0.3)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, facecolor="white")
print("wrote", OUT)
