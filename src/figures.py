"""
Figures for the paper, drawn from the study's own CSV tables.

Every function here takes a DataFrame the study already saved and writes a PNG
and a PDF.  Nothing is computed that the table does not hold, so each figure
has a table twin with the exact values (``report`` writes both).

HOUSE RULES (and why)
---------------------
* One job per colour.  Layers and models are *identities*: they take fixed
  hues, assigned by name, never by rank -- "classifier" is the same blue in
  every figure whether or not another layer is present.  Seen/unseen is an
  *ordered* contrast and takes two steps of one hue.  AUROC is a *magnitude*
  and takes a single-hue ramp.
* The palette was checked for colour-vision deficiency before it was used
  (adjacent CVD delta-E 9.1, normal-vision 22.9 on this surface).  Two of its
  hues sit below 3:1 contrast on a light page, so identity never rests on
  colour alone: every multi-series figure has a legend AND direct labels.
* Text is ink, never the series colour.  A coloured mark beside the text
  carries identity.
* No second y-axis, anywhere.  Two measures on different scales get two
  panels.
* Grid and axes are hairlines one step off the surface; the data is the only
  thing allowed to be loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ---- palette (validated; see module docstring) ---------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"          # de-emphasis: "missed", "not applicable"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_LIGHT = "#86b6ef"       # ordinal partner of BLUE (step 250)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# identity -> colour.  Fixed by NAME so a figure with fewer layers does not
# repaint the ones that remain.
LAYER_COLOUR: Dict[str, str] = {
    "classifier": BLUE,
    "novelty": ORANGE,
    "behaviour": AQUA,
}
MODEL_COLOUR: Dict[str, str] = {"xgb": BLUE, "rf": ORANGE, "lgbm": AQUA, "mlp": YELLOW}
BUDGET_COLOUR = [BLUE, ORANGE, AQUA]

ANY_ATTACK = "__ANY_ATTACK__"


def _style(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path.with_suffix(".png")


def _pct(v: float) -> str:
    if v != v:
        return "n/a"
    return f"{100 * v:.1f}%" if (0 < v < 0.1 or 0.9 < v < 1) else f"{100 * v:.0f}%"


def _class_order(df: pd.DataFrame, by: str = "n") -> List[str]:
    d = df[df["class"] != ANY_ATTACK].drop_duplicates("class")
    return d.sort_values(by, ascending=True)["class"].tolist()


# =====================================================================
# E5 -- which layer catches what
# =====================================================================


def fig_layer_attribution(
    attr: pd.DataFrame,
    out_path: Path,
    title: str,
    layers: Sequence[str] = ("classifier", "novelty", "behaviour"),
    benign_label: str = "BENIGN",
) -> Optional[Path]:
    """Per attack class: the share of its flows caught by each layer alone, by
    two or more layers, or by none.  Horizontal 100% stacked bars."""
    d = attr[attr["class"] != benign_label].copy()
    if d.empty:
        return None

    def bucket(row) -> str:
        if row["n_layers"] == 0:
            return "missed"
        if row["n_layers"] == 1:
            return row["layers"]
        return "two or more layers"

    d["bucket"] = d.apply(bucket, axis=1)
    present = set(d.loc[d["share"] > 0, "bucket"])
    order = [b for b in (*layers, "two or more layers", "missed") if b in present]
    colour = {**LAYER_COLOUR, "two or more layers": YELLOW, "missed": NEUTRAL}
    piv = (d.groupby(["class", "bucket"])["share"].sum().unstack(fill_value=0.0)
           .reindex(columns=order, fill_value=0.0))
    totals = d.drop_duplicates("class").set_index("class")["class_total"]
    piv = piv.loc[totals.sort_values().index]

    fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(piv) + 1.4))
    _style(ax, "x")
    y = np.arange(len(piv))
    left = np.zeros(len(piv))
    for b in order:
        vals = piv[b].to_numpy()
        ax.barh(y, vals, left=left, height=0.42, color=colour[b], label=b,
                edgecolor=SURFACE, linewidth=1.6)
        left += vals
    ax.set_ylim(-0.6, len(piv) - 0.4)
    missed = piv["missed"].to_numpy() if "missed" in piv.columns else np.zeros(len(piv))
    detected = 1.0 - missed
    for yi, det in zip(y, detected):
        ax.text(1.015, yi, f"{_pct(det)} caught", va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  (n={int(totals[c]):,})" for c in piv.index], fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the class's flows", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(order), frameon=False,
              fontsize=8, labelcolor=INK2, handlelength=1.0, columnspacing=1.2)
    return _save(fig, out_path)


# =====================================================================
# E2 -- seen versus unseen
# =====================================================================


def fig_seen_unseen(df: pd.DataFrame, out_path: Path, title: str) -> Optional[Path]:
    """Dumbbell per attack class: detection when the class was in training
    (dark) and when it was held out (light)."""
    d = df.dropna(subset=["det_seen", "det_unseen"], how="all").copy()
    if d.empty:
        return None
    d = d.sort_values("det_unseen", ascending=True)
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(d) + 1.5))
    _style(ax, "x")
    for yi, (_, r) in zip(y, d.iterrows()):
        if r["det_seen"] == r["det_seen"] and r["det_unseen"] == r["det_unseen"]:
            ax.plot([r["det_unseen"], r["det_seen"]], [yi, yi], color=AXIS, linewidth=1.6,
                    solid_capstyle="round", zorder=1)
    # light marker larger and underneath, dark marker smaller and on top: when
    # the two rates coincide both are still visible, as a ring round a dot
    ax.scatter(d["det_unseen"], y, s=78, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=1.4,
               zorder=3, label="class held out (unseen)")
    ax.scatter(d["det_seen"], y, s=30, color="#184f95", edgecolor=SURFACE, linewidth=1.0,
               zorder=4, label="class in training (seen)")
    for yi, (_, r) in zip(y, d.iterrows()):
        if r["det_unseen"] == r["det_unseen"]:
            ax.text(1.03, yi, _pct(r["det_unseen"]), va="center", ha="left", fontsize=8, color=INK)
    ax.text(1.03, len(d) - 0.35, "unseen", va="bottom", ha="left", fontsize=7.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  (n={int(n):,})" for c, n in zip(d["class"], d["n_total"])],
                       fontsize=8, color=INK)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("flows detected", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8,
              labelcolor=INK2, handletextpad=0.3, columnspacing=1.6)
    return _save(fig, out_path)


# =====================================================================
# E1 -- protocol effect
# =====================================================================


def fig_protocol_effect(df: pd.DataFrame, out_path: Path, title: str,
                        value: str = "detection_rate") -> Optional[Path]:
    """One line per model across split protocols: what the same model scores
    when only the split changes."""
    d = df.copy()
    if d.empty:
        return None
    protocols = [p for p in ("random", "blocked", "crossday") if p in set(d["protocol"])]
    models = list(dict.fromkeys(d["model"]))
    x = np.arange(len(protocols))

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    _style(ax, "y")
    spare = [YELLOW, AQUA, ORANGE, BLUE]
    for m in models:
        col = MODEL_COLOUR.get(m) or spare.pop()
        vals = [d[(d["model"] == m) & (d["protocol"] == p)][value].mean() for p in protocols]
        ax.plot(x, vals, color=col, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=m, solid_capstyle="round")
        if vals and vals[-1] == vals[-1]:
            ax.text(x[-1] + 0.08, vals[-1], f"{m}  {_pct(vals[-1])}", va="center", ha="left",
                    fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=8.5, color=INK)
    ax.set_xlim(-0.25, len(protocols) - 1 + 0.75)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("attack flows detected", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=24)
    if len(models) > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(models), frameon=False,
                  fontsize=8, labelcolor=INK2)
    return _save(fig, out_path)


# =====================================================================
# E6 -- threshold transfer
# =====================================================================


def fig_threshold_transfer(df: pd.DataFrame, out_path: Path, title: str) -> Optional[Path]:
    """Observed test false-alarm rate against the budget it was aimed at, per
    calibration strategy.

    Each budget is a short connector from its target (a tick in ink) to what
    was observed (a dot in the budget's colour).  The length of the connector
    is the miss; a dot sitting on its tick means the budget held.
    """
    d = df[df["class"] == ANY_ATTACK].copy()
    if d.empty:
        return None
    models = list(dict.fromkeys(d["model"]))
    strategies = list(dict.fromkeys(d["strategy"]))[::-1]
    budgets = sorted(d["budget"].unique())
    dodge = np.linspace(0.24, -0.24, len(budgets)) if len(budgets) > 1 else np.array([0.0])

    fig, axes = plt.subplots(1, len(models),
                             figsize=(3.7 * len(models) + 0.6, 0.62 * len(strategies) + 1.7),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        _style(ax, "x")
        ax.set_xscale("log")
        y = np.arange(len(strategies), dtype=float)
        for bi, b in enumerate(budgets):
            col = BUDGET_COLOUR[bi % len(BUDGET_COLOUR)]
            sub = d[(d["model"] == m) & (d["budget"] == b)].set_index("strategy").reindex(strategies)
            obs = sub["observed_fpr"].clip(lower=1e-5).to_numpy()
            yy = y + dodge[bi]
            ax.hlines(yy, np.minimum(obs, b), np.maximum(obs, b), color=AXIS, linewidth=1.6,
                      zorder=1)
            ax.scatter(np.full(len(yy), b), yy, marker="|", s=90, color=INK, linewidth=1.2,
                       zorder=2, label="the budget aimed at" if bi == 0 else None)
            ax.scatter(obs, yy, s=44, color=col, edgecolor=SURFACE, linewidth=1.4, zorder=3,
                       label=f"aimed at {100 * b:g}%")
        ax.set_yticks(y)
        ax.set_yticklabels(strategies, fontsize=8, color=INK)
        ax.set_ylim(-0.6, len(strategies) - 0.4)
        ax.set_xlabel("false-alarm rate observed on the test day", fontsize=8, color=INK2)
        ax.set_title(m, fontsize=9, color=INK2, loc="left")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(which="both", length=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [i for i, l in enumerate(labels) if l != "the budget aimed at"] + \
            [i for i, l in enumerate(labels) if l == "the budget aimed at"]
    # title, then one legend row, then the panels: fixed bands so that nothing
    # can land on a panel title
    h = fig.get_figheight()
    fig.subplots_adjust(top=1.0 - 0.95 / h)
    fig.suptitle(title, fontsize=10, color=INK, x=0.02, ha="left", y=1.0 - 0.10 / h)
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
               bbox_to_anchor=(0.02, 1.0 - 0.32 / h), ncol=len(order), frameon=False,
               fontsize=8, labelcolor=INK2, handletextpad=0.2, columnspacing=1.4)
    return _save(fig, out_path)


# =====================================================================
# E4 -- behaviour-layer threshold sweep
# =====================================================================


def fig_threshold_sweep(df: pd.DataFrame, out_path: Path, title: str,
                        deployed: Optional[float] = None) -> Optional[Path]:
    """Two panels over the port threshold: what is caught, and what it costs.

    Two panels, not two y-axes: detection is a share and the cost is a count,
    and aligning two unrelated scales on one plot invents a relationship.
    """
    if df.empty:
        return None
    d = df.sort_values("port_threshold")
    # only classes the layer covers at some threshold: a line that sits on
    # zero throughout hides the ones under it and says nothing
    det_cols = [c for c in d.columns if c.startswith("det__") and d[c].max() > 0.01]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2], "hspace": 0.18})
    for ax in (a1, a2):
        _style(ax, "y")
        ax.set_xscale("log")
    cols = [BLUE, ORANGE, AQUA]
    for c, col in zip(det_cols[:3], cols):
        name = c.replace("det__", "")
        a1.plot(d["port_threshold"], d[c], color=col, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.3, label=name, solid_capstyle="round")
    a1.set_ylim(-0.02, 1.02)
    a1.set_yticks([0, 0.5, 1.0])
    a1.set_yticklabels(["0%", "50%", "100%"])
    a1.set_ylabel("flows covered", fontsize=8, color=INK2)
    if len(det_cols) > 1:
        a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8,
                  labelcolor=INK2)
    elif len(det_cols) == 1:
        # one series: the axis label names it, no legend box needed
        a1.set_ylabel(f"{det_cols[0].replace('det__', '')}\nflows covered", fontsize=8, color=INK2)
    a1.set_title(title, fontsize=10, color=INK, loc="left", pad=24 if len(det_cols) > 1 else 8)

    a2.plot(d["port_threshold"], d["benign_groups_per_day"], color=INK2, linewidth=2, marker="o",
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.3, solid_capstyle="round")
    a2.set_ylabel("benign-labelled\nsource-windows\nflagged per day", fontsize=8, color=INK2)
    a2.set_xlabel("distinct destination ports per source per window (threshold)", fontsize=8,
                  color=INK2)
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    for ax in (a1, a2):
        ax.tick_params(which="both", length=0)
    if deployed:
        # drawn last: a reference line added to an empty log axis collapses it
        for ax in (a1, a2):
            ax.axvline(deployed, color=AXIS, linewidth=0.8, zorder=0)
        top = a2.get_ylim()[1]
        a2.text(deployed, top, " deployed threshold", fontsize=7.5, color=MUTED, ha="left",
                va="top")
    return _save(fig, out_path)


# =====================================================================
# E3 -- separability heatmap
# =====================================================================


def fig_auc_heatmap(df: pd.DataFrame, out_path: Path, title: str,
                    value: str = "auroc") -> Optional[Path]:
    """Detector x attack class, one hue, darker = better separated from benign.

    0.5 (chance) sits at the light end on purpose: a detector that cannot tell
    a class from benign should fade into the page.  Exact values are in the
    table that ships with the figure.
    """
    d = df[df["class"] != ANY_ATTACK]
    if d.empty:
        return None
    piv = d.pivot_table(index="detector", columns="class", values=value, aggfunc="mean")
    piv = piv.loc[:, piv.mean().sort_values(ascending=False).index]
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    cmap.set_bad(SURFACE)

    fig, ax = plt.subplots(figsize=(0.62 * piv.shape[1] + 2.4, 0.42 * piv.shape[0] + 1.6))
    im = ax.imshow(np.clip(piv.to_numpy(dtype=float), 0.5, 1.0), cmap=cmap, vmin=0.5, vmax=1.0,
                   aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=40, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # surface gaps between cells, not borders
    ax.set_xticks(np.arange(-0.5, piv.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, piv.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[0.5, 0.75, 1.0])
    cb.ax.set_yticklabels(["0.5 or below", "0.75", "1.0"], fontsize=7.5, color=INK2)
    cb.outline.set_visible(False)
    cb.set_label(value.upper(), fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=10)
    return _save(fig, out_path)
