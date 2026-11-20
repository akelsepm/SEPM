"""Chapter 9 figures: per-system gap curves (final configuration vs
literature forms, 300 K) in square side-by-side pairs, plus the
Gamma-residual hump spectrum of all nine systems (VCA).

Reads the final grids in ../hidden_files/ternary_<sys>_300k.json and
the targets in ternary_targets.py. Output:
../files/book-bolum-01/ternary_data/fig_ch9_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ternary_targets as TT

HID = Path(__file__).parent.parent / "hidden_files"
FIG = Path(__file__).parent.parent / "files" / "book-bolum-01" / "ternary_data"
FIG.mkdir(parents=True, exist_ok=True)
TEAL, GOLD, GRAY, RED = "#0f766e", "#b98a1f", "#44403c", "#b91c1c"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})


def grid(sysname):
    rows = json.loads((HID / f"ternary_{sysname.lower()}_300k.json").read_text())
    x = np.array([r["x"] for r in rows])
    return x, rows


def panel(ax, sysname, title):
    x, rows = grid(sysname)
    tgt = TT.SYSTEMS[sysname]
    xf = np.linspace(0, 1, 200)
    eg = np.array([r["edges"]["Eg_Gamma_eV"] for r in rows])
    ex = np.array([r["edges"]["E_X_eV"] for r in rows])
    el = np.array([r["edges"]["E_L_eV"] for r in rows])
    ax.plot(x, eg, "o-", color=TEAL, lw=1.4, ms=3.5, label="Γ (ours)")
    ax.plot(x, ex, "o-", color=GOLD, lw=1.4, ms=3.5, label="X (ours)")
    ax.plot(x, el, "o-", color=GRAY, lw=1.4, ms=3.5, label="L (ours)")
    ax.plot(xf, tgt["G"](xf, 300), "--", color=TEAL, lw=1.0, alpha=0.55)
    ax.plot(xf, tgt["X"](xf, 300), "--", color=GOLD, lw=1.0, alpha=0.55)
    ax.plot(xf, tgt["L"](xf, 300), "--", color=GRAY, lw=1.0, alpha=0.55)
    ax.set_xlabel("x")
    ax.set_ylabel("valley energy (eV)")
    ax.set_title(title, color=GRAY, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)


PAIRS = [("InGaAs", "AlInAs", "InGaAs — path-limited", "AlInAs — path-limited"),
         ("InGaP", "GaAsP", "InGaP — validated (VS8)", "GaAsP — path-limited"),
         ("AlGaSb", "InGaSb", "AlGaSb — path-limited", "InGaSb — validated (VS8+VA4)"),
         ("GaAsSb", "InAsSb", "GaAsSb — path-limited", "InAsSb — validated (VS8)")]
for i, (s1, s2, t1, t2) in enumerate(PAIRS, 1):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    panel(axes[0], s1, t1)
    panel(axes[1], s2, t2)
    axes[0].legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / f"fig_ch9_pair{i}.png", dpi=200)
    plt.close(fig)
    print("fig pair", i, "ok")

# Gamma hump spectrum (VCA residuals, all nine systems)
import ternary_fit as TF
fig, ax = plt.subplots(figsize=(7.6, 4.2))
styles = {"AlGaAs": "-", "InGaSb": "-", "InGaP": "-", "InAsSb": "-",
          "InGaAs": "--", "AlInAs": "--", "GaAsP": "--", "AlGaSb": "--",
          "GaAsSb": "--"}
for s in ["AlGaAs", "InGaSb", "InGaP", "InAsSb", "GaAsP", "InGaAs",
          "GaAsSb", "AlInAs", "AlGaSb"]:
    if s == "AlGaAs":
        import fit_bowing as FB
        res = [1000 * (FB.edges_b(x, 300, 0.0)["Eg_Gamma_eV"] - FB.LIT[300]["G"](x))
               for x in TF.XS]
    else:
        r = TF.resid(s, 300, {})
        res = r["G"]
    ax.plot(TF.XS, res, styles[s], lw=1.3, marker="o", ms=3, label=s)
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_xlabel("x")
ax.set_ylabel("Gamma residual (meV)")
ax.set_title("VCA Gamma-path residual in nine systems (300 K)", color=GRAY, fontsize=10)
ax.legend(fontsize=7.5, ncol=3, loc="best")
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(FIG / "fig_ch9_humps.png", dpi=200)
plt.close(fig)
print("humps ok")
