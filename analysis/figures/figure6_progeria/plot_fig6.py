#!/usr/bin/env python3
"""Plot HGPS likelihood fits against the reconstructed individual-record KM curve."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "results/progeria"
FIGURE = ROOT / "Figures/Figure6/Fig6.png"


def km(records):
    """Kaplan-Meier survival with Greenwood log-log pointwise intervals."""
    age = np.asarray([r['age'] for r in records])
    death = np.asarray([r['death'] for r in records], dtype=bool)
    t, s, lo, hi = [0.], [1.], [1.], [1.]
    survival, greenwood = 1., 0.
    for event_age in sorted(set(age[death])):
        n = int(np.sum(age >= event_age))
        d = int(np.sum((age == event_age) & death))
        survival *= 1 - d / n
        if n > d:
            greenwood += d / (n * (n - d))
        if 0 < survival < 1 and greenwood > 0:
            se = np.sqrt(greenwood) / abs(np.log(survival))
            z = np.log(-np.log(survival))
            lower = np.exp(-np.exp(z + 1.96 * se))
            upper = np.exp(-np.exp(z - 1.96 * se))
        else:
            lower = upper = survival
        t.append(event_age)
        s.append(survival)
        lo.append(lower)
        hi.append(upper)
    return tuple(np.asarray(values) for values in (t, s, lo, hi))

mpl.rcParams.update({"font.family": "Arial", "font.size": 16,
                     "axes.labelsize": 19, "axes.labelweight": "normal",
                     "xtick.labelsize": 16.5, "ytick.labelsize": 16.5,
                     "legend.fontsize": 12, "axes.linewidth": 1.3})

COLORS = {
    "eta": "#008489", "beta": "#2B5685", "Xc": "#D37610", "epsilon": "#DEA322",
    "Xc+epsilon": "#C53334", "Xc+eta": "#008489", "Xc+beta": "#2B5685",
    "eta+beta": "#684392", "epsilon+eta": "#48A3A2", "epsilon+beta": "#687FAD",
}
SYMBOL = {"eta": r"$\eta$", "beta": r"$\beta$", "epsilon": r"$\epsilon$", "Xc": r"$X_c$"}


def style(ax, norm=False):
    ax.set_ylim(-.015, 1.04)
    ax.set_xlim(0, 1.95 if norm else 30)
    ax.set_xticks([0, .5, 1, 1.5] if norm else [0, 10, 20, 30])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel("Age / median lifespan" if norm else "Age (years)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=1.3, length=6, labelsize=16.5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("normal")
    ax.xaxis.label.set_fontweight("normal")


def main():
    fits = {r["model"]: r for r in json.loads((HERE / "results.json").read_text())["models"]}
    curves = {}
    with (HERE / "survival_curves.csv").open() as f:
        for row in csv.DictReader(f):
            curves.setdefault(row["model"], []).append((float(row["age"]), float(row["survival"])))
    for key in curves:
        curves[key] = np.asarray(curves[key])
    with (ROOT / "data/hgps/records.csv").open() as f:
        records = list(csv.DictReader(f))
    for r in records:
        r["age"] = float(r["age"])
        r["death"] = int(r["death"])
    t, s, lo, hi = km(records)
    median = float(t[np.flatnonzero(s <= .5)[0]])
    with (HERE / "period_survival.csv").open() as f:
        sw = [r for r in csv.DictReader(f) if r["country"] == "SWE"]
    swx = np.asarray([float(r["age_over_median"]) for r in sw])
    swy = np.asarray([float(r["survival"]) for r in sw])

    fig, axes = plt.subplots(1, 3, figsize=(20.5, 6.3),
                             gridspec_kw={"width_ratios": [1, 1.18, 1.35], "wspace": .27})
    for ax, scale in zip(axes, [median, 1, 1]):
        ax.fill_between(t / scale, lo, hi, step="post", color="#555555", alpha=.19, lw=0)
        ax.step(t / scale, s, where="post", color="#171717", lw=2.8,
                label="HGPS observed", zorder=10)
        style(ax, norm=(scale != 1))
    axes[0].set_ylabel("Survival")
    axes[0].plot(swx, swy, "--", color="#555555", lw=2.5, label="Sweden 2019")
    axes[0].legend(loc="lower left", frameon=False, fontsize=13.5)

    for name in ["eta", "beta", "Xc", "epsilon"]:
        x = curves[name]
        label = f"{SYMBOL[name]} ({fits[name]['factors'][name]:.3g}×; ΔAIC {fits[name]['delta_aic']:.1f})"
        axes[1].plot(x[:, 0], x[:, 1], lw=2.4, color=COLORS[name], label=label)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -.23),
                   frameon=False, fontsize=12, handlelength=2, borderaxespad=0)

    pair_order = sorted(["Xc+epsilon", "Xc+eta", "Xc+beta", "eta+beta",
                         "epsilon+eta", "epsilon+beta"], key=lambda n: fits[n]["aic"])
    for name in pair_order:
        x = curves[name]
        label = ", ".join(f"{SYMBOL[p]} {fits[name]['factors'][p]:.3g}×" + ('*' if p in fits[name].get('boundary', []) else '') for p in name.split("+"))
        label += f"  (ΔAIC {fits[name]['delta_aic']:.1f})"
        axes[2].plot(x[:, 0], x[:, 1], lw=2.7 if name == "Xc+beta" else 2.0,
                     color=COLORS[name], label=label,
                     zorder=7 if name == "Xc+beta" else 3)
    axes[2].legend(loc="upper left", bbox_to_anchor=(0, -.23),
                   frameon=False, fontsize=12, borderaxespad=0,
                   labelspacing=.35, handlelength=2)
    for ax, letter in zip(axes, "abc"):
        ax.text(-.085, 1.025, letter, transform=ax.transAxes, fontsize=27, fontweight="normal")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=280, bbox_inches="tight")
    plt.close(fig)
    print(FIGURE)


if __name__ == "__main__":
    main()
