import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

THEME = {
 "light": dict(surf="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e0dfda",
               cat=["#2a78d6", "#eb6834"], ramp=["#9adcff", "#5594e5", "#11549f"]),
 "dark":  dict(surf="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a37",
               cat=["#3987e5", "#d95926"], ramp=["#99dbff", "#4f8dde", "#004792"]),
}
LW = 11.0          # bar thickness in points -- thin marks

def on(bg):
    """Ink that stays legible on `bg`: value labels must never ride on colour alone."""
    def lin(c):
        c = int(c, 16) / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    y = (0.2126 * lin(bg[1:3]) + 0.7152 * lin(bg[3:5]) + 0.0722 * lin(bg[5:7]))
    return "#0b0b0b" if (y + 0.05) / 0.05 > 1.05 / (y + 0.05) else "#ffffff"

def rbar(ax, x0, x1, y, color, lw=LW, z=3):
    """A bar drawn as a capped line so both data-ends are rounded."""
    ax.plot([x0, x1], [y, y], linewidth=lw, color=color, solid_capstyle="round",
            zorder=z, clip_on=False)

def frame(t, figsize, title, subtitle, xlabel, legend):
    p = THEME[t]
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(p["surf"]); ax.set_facecolor(p["surf"])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("bottom")
    ax.tick_params(axis="x", colors=p["ink2"], labelsize=11, length=0, pad=7)
    ax.tick_params(axis="y", colors=p["ink"], labelsize=11.5, length=0, pad=8)
    ax.xaxis.grid(True, color=p["grid"], linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=p["ink2"], fontsize=10.5, labelpad=16)
    fig.text(0.035, 0.972, title, color=p["ink"], fontsize=15.5,
             fontweight="bold", va="top", ha="left")
    fig.text(0.035, 0.895, subtitle, color=p["ink2"], fontsize=11.5,
             va="top", ha="left", linespacing=1.45)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=9,
                      markerfacecolor=c, markeredgecolor="none", label=l)
               for l, c in legend]
    lg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.032, 0.775),
                    frameon=False, ncol=len(legend), fontsize=11.5,
                    handletextpad=0.5, columnspacing=1.8)
    for txt in lg.get_texts(): txt.set_color(p["ink"])
    return fig, ax, p

# ---------- chart 1
TRACKS = ["DAPT", "SFT", "RLVR", "DPO", "VLM", "STaR"]
V04  = {"DAPT": 26767, "SFT": 15666, "VLM": 1845, "STaR": 0, "DPO": 0, "RLVR": 0}
V052 = {"DAPT": 8928,  "SFT": 5386,  "VLM": 2552, "STaR": 2103, "DPO": 2490, "RLVR": 5386}

def chart_split(t):
    pal = THEME[t]
    fig, ax, p = frame(
        t, (9.6, 5.0),
        "Routing removed two thirds of the training rows and added three tracks",
        "Documents the generator judged amendment-exposed no longer produce DAPT or SFT rows;\n"
        "they go to a retrieval corpus instead. Preference and verifier tracks are new in v0.5.2.",
        "rows in the training split",
        [("v0.4", pal["cat"][0]), ("v0.5.2", pal["cat"][1])])
    y = np.arange(len(TRACKS))[::-1] * 1.0
    off = 0.21
    for yi, name in zip(y, TRACKS):
        a, b = V04[name], V052[name]
        if a:
            rbar(ax, 0, a, yi + off, p["cat"][0])
            ax.text(a + 500, yi + off, f"{a:,}", va="center", ha="left",
                    fontsize=10.5, color=p["ink2"])
        else:
            ax.text(250, yi + off, "not produced", va="center", ha="left",
                    fontsize=10, color=p["ink2"], style="italic")
        rbar(ax, 0, b, yi - off, p["cat"][1])
        ax.text(b + 500, yi - off, f"{b:,}", va="center", ha="left",
                fontsize=10.5, color=p["ink2"])
    ax.set_yticks(y); ax.set_yticklabels(TRACKS)
    ax.set_xlim(0, 31500); ax.set_ylim(-0.7, len(TRACKS) - 0.3)
    ticks = [10000, 20000, 30000]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{v:,}" for v in ticks])
    fig.subplots_adjust(left=0.105, right=0.975, top=0.615, bottom=0.135)
    fig.savefig(f"doc/split-composition-{t}.png", facecolor=p["surf"])
    plt.close(fig)

# ---------- chart 2
ROWS = [("uc1 safety", 23, 45, 89), ("uc2 rebar spec", 104, 5, 41),
        ("uc4 faithfulness", 25, 63, 72), ("uc5 incident", 13, 5, 100),
        ("uc6 verdict", 138, 328, 344), ("uc7 requirement", 54, 26, 48),
        ("sft", 62, 136, 196)]

def chart_routing(t):
    pal = THEME[t]
    fig, ax, p = frame(
        t, (9.6, 5.2),
        "Most scored items come from documents withheld from training",
        "Share of each track's items by the route the generator gave their source document.\n"
        "uc2 is the exception: two thirds of its threshold items still train the model.",
        "share of a track's items, %",
        [("train", pal["ramp"][0]), ("both", pal["ramp"][1]), ("retrieve", pal["ramp"][2])])
    y = np.arange(len(ROWS))[::-1] * 1.0
    left = np.zeros(len(ROWS)); GAP = 0.6
    for i in range(3):
        vals = np.array([100 * r[i+1] / sum(r[1:]) for r in ROWS])
        ax.barh(y, vals, 0.40, left=left, color=p["ramp"][i], zorder=3, linewidth=0)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v >= 8:
                ax.text(l + v/2, y[yi], f"{v:.0f}", ha="center", va="center",
                        fontsize=10, color=on(p["ramp"][i]))
        left = left + vals + GAP
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}   n={sum(r[1:])}" for r in ROWS])
    ax.set_xlim(0, 102.5); ax.set_ylim(-0.7, len(ROWS) - 0.3)
    ax.set_xticks([25, 50, 75, 100])
    fig.subplots_adjust(left=0.275, right=0.975, top=0.615, bottom=0.135)
    fig.savefig(f"doc/routing-mix-{t}.png", facecolor=p["surf"])
    plt.close(fig)

for t in ("light", "dark"):
    chart_split(t); chart_routing(t)
print("ok")


# ---------- chart 3: the passage test
BOOKS = [
    ("uc2 rebar spec", 0.9667, 0.1000),
    ("sft",         0.9500, 0.1406),
    ("uc1 safety",      0.9294, 0.1529),
    ("probe",    0.9281, 0.1125),
    ("uc4 faithfulness",   0.9062, 0.1375),
    ("uc7 requirement",    0.7109, 0.0234),
    ("uc5 incident",       0.4572, 0.0107),
    ("uc6 verdict",        0.4136, 0.3951),
]

def chart_books(t):
    pal = THEME[t]
    fig, ax, p = frame(
        t, (9.6, 5.4),
        "Withholding the clause collapses every track except one",
        "Same items, scored with and without the passage the answer was mined from.\n"
        "uc6 moves 0.019 (McNemar p=0.45) and sits below the 0.500 a constant answer scores.",
        "score",
        [("open book", pal["cat"][0]), ("closed book", pal["cat"][1])])
    y = np.arange(len(BOOKS))[::-1] * 1.0
    off = 0.21
    for yi, (name, a, b) in zip(y, BOOKS):
        rbar(ax, 0, a, yi + off, p["cat"][0])
        ax.text(a + 0.012, yi + off, f"{a:.3f}", va="center", ha="left",
                fontsize=10.5, color=p["ink2"])
        rbar(ax, 0, b, yi - off, p["cat"][1])
        ax.text(b + 0.012, yi - off, f"{b:.3f}", va="center", ha="left",
                fontsize=10.5, color=p["ink2"])
    # the number a constant answer scores on uc6's balanced two-label set
    ax.plot([0.5, 0.5], [-0.62, 0.62], linewidth=1.6, color=p["ink2"],
            linestyle=(0, (4, 3)), zorder=4)
    ax.text(0.515, -0.66, "0.500  constant answer", fontsize=9.5,
            color=p["ink2"], va="top", ha="left")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in BOOKS])
    ax.set_xlim(0, 1.08); ax.set_ylim(-1.05, len(BOOKS) - 0.3)
    ax.set_xticks([0.25, 0.50, 0.75, 1.0])
    fig.subplots_adjust(left=0.215, right=0.975, top=0.645, bottom=0.125)
    fig.savefig(f"doc/open-vs-closed-v052-{t}.png", facecolor=p["surf"])
    plt.close(fig)


for t in ("light", "dark"):
    chart_books(t)
print("ok books")


# ---------- chart 4: retrieval tracks corpus coverage
RAG = [  # track, coverage, closed, rag, open
    ("uc4 faithfulness",   1.000, 0.1375, 0.3937, 0.9062),
    ("uc1 safety",      0.637, 0.0828, 0.2566, 0.8134),
    ("uc7 requirement",    0.570, 0.0234, 0.1562, 0.7109),
    ("uc5 incident",       0.186, 0.0107, 0.0105, 0.4572),
    ("uc2 rebar spec", 0.047, 0.1000, 0.0800, 0.9667),
]

def chart_rag(t):
    pal = THEME[t]
    fig, ax, p = frame(
        t, (9.6, 5.2),
        "Retrieval helps exactly as far as the corpus reaches",
        "The baseline searches the held-out chunks; half the items were mined from documents\n"
        "that are not in it. Where the answer is there, retrieval recovers part of the open-book gap.",
        "score",
        [("closed book", pal["ramp"][0]), ("retrieved", pal["ramp"][1]),
         ("open book", pal["ramp"][2])])
    y = np.arange(len(RAG))[::-1] * 1.0
    offs = (0.26, 0.0, -0.26)
    for yi, row in zip(y, RAG):
        name, cov, cl, rg, op = row
        for off, val, col in zip(offs, (cl, rg, op), p["ramp"]):
            rbar(ax, 0, val, yi + off, col, lw=8.0)
            ax.text(val + 0.012, yi + off, f"{val:.3f}", va="center", ha="left",
                    fontsize=9.5, color=p["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}\ngold in corpus {r[1]:.2f}" for r in RAG])
    ax.set_xlim(0, 1.06); ax.set_ylim(-0.75, len(RAG) - 0.25)
    ax.set_xticks([0.25, 0.50, 0.75, 1.0])
    fig.subplots_adjust(left=0.245, right=0.975, top=0.615, bottom=0.125)
    fig.savefig(f"doc/rag-coverage-{t}.png", facecolor=p["surf"])
    plt.close(fig)


for t in ("light", "dark"):
    chart_rag(t)
print("ok rag")


# ---------- chart 5: category weighting, training against evaluation
# train% , eval% , retrieve% , open-book score
CATS = [
    ("Safety and disaster",        13.2, 37.7, 49, 0.632),
    ("General construction",        9.4, 11.8, 63, 0.661),
    ("Structures and facilities",  10.9, 10.1, 63, 0.758),
    ("Equipment and services",      9.1,  9.6, 47, 0.648),
    ("Design standards (KDS)",     15.4,  8.2, 32, 0.879),
    ("Quality and inspection",      4.9,  7.9, 57, 0.716),
    ("Contract and cost",           1.3,  4.1, 80, 0.725),
    ("Building and architecture",   2.7,  3.7, 65, 0.642),
    ("Specifications (KCS)",       12.7,  3.1, 22, 0.878),
    ("Research reports",            5.0,  0.2,  0, 1.000),
]

def chart_cats(t):
    pal = THEME[t]
    fig, ax, p = frame(
        t, (9.6, 5.8),
        "Training and evaluation still do not weight the same categories",
        "Safety and disaster is 13% of the training rows and 38% of the scored items;\n"
        "specifications and research reports run the other way. Both come from one corpus.",
        "share of rows / items, %",
        [("training rows", pal["cat"][0]), ("scored items", pal["cat"][1])])
    y = np.arange(len(CATS))[::-1] * 1.0
    off = 0.21
    for yi, (name, tr, ev, _, _) in zip(y, CATS):
        rbar(ax, 0, tr, yi + off, p["cat"][0])
        ax.text(tr + 0.5, yi + off, f"{tr:.1f}", va="center", ha="left",
                fontsize=10, color=p["ink2"])
        rbar(ax, 0, ev, yi - off, p["cat"][1])
        ax.text(ev + 0.5, yi - off, f"{ev:.1f}", va="center", ha="left",
                fontsize=10, color=p["ink2"])
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in CATS])
    ax.set_xlim(0, 41); ax.set_ylim(-0.7, len(CATS) - 0.3)
    ax.set_xticks([10, 20, 30, 40])
    fig.subplots_adjust(left=0.265, right=0.975, top=0.655, bottom=0.115)
    fig.savefig(f"doc/catdist-v052-{t}.png", facecolor=p["surf"])
    plt.close(fig)


for t in ("light", "dark"):
    chart_cats(t)
print("ok cats")
