"""Chapter 7 atlas figures: per material (300 K) two SQUARE panels —
dense bands (X-Gamma-L) and residual lollipops in sigma units.
Residuals: pool anchor observables vs the verified target set
(fit_pool.TARGETS with the DOS-valley swap applied as in the fits;
base sigmas, not the narrow-gap fitting profile)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fit_pool as FP

HERE = Path(__file__).parent
FIG = HERE.parent / "files" / "book-bolum-01"
DATA = FIG / "atlas_data"
TEAL, GRAY, RED, GOLD = "#0f766e", "#44403c", "#b91c1c", "#b98a1f"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRAY,
                     "text.color": GRAY, "axes.labelcolor": GRAY,
                     "xtick.color": GRAY, "ytick.color": GRAY})

LABELS = {"Eg": "$E_\\Gamma$", "E_L": "$E_L$", "E_X": "$E_X$", "Dso": "$\\Delta_{so}$",
          "me100": "$m_e[100]$", "mlh100": "$m_{lh}[100]$", "mhh100": "$m_{hh}[100]$",
          "me111": "$m_e[111]$", "mlh111": "$m_{lh}[111]$", "mhh111": "$m_{hh}[111]$",
          "dip_L": "dip$_L$", "mt_L": "$m_{tL}$", "ml_Lmin": "$m_{lL}$",
          "mt_X": "$m_{tX}$", "ml_X": "$m_{lX}$",
          "mDOS_L": "$m^{DOS}_L$", "mDOS_X": "$m^{DOS}_X$"}


def targets_for(mat, T):
    tg = dict(FP.TARGETS[(mat, T)]["targets"])
    if mat in FP.DOS_VALLEY:
        for k in ("mt_L", "ml_Lmin", "mt_X", "ml_X"):
            tg.pop(k, None)
        tg.update(FP.DOS_VALLEY[mat])
    return tg


def zscores(mat, T):
    e = json.loads((HERE / "params_pool" / f"{mat}_{T}K" / "params.json").read_text())
    o = e["observables"]
    tg = targets_for(mat, T)
    out = []
    for k in ["Eg", "E_L", "E_X", "Dso", "me100", "mlh100", "mhh100",
              "me111", "mlh111", "mhh111", "dip_L", "mt_L", "ml_Lmin",
              "mt_X", "ml_X", "mDOS_L", "mDOS_X"]:
        if k not in tg or k not in o:
            continue
        v, s = tg[k]
        z = (o[k] - v) / s if k in ("Eg", "E_L", "E_X", "Dso", "dip_L") \
            else (o[k] / v - 1.0) / s
        out.append((k, z))
    return out


def bands_fig(mat):
    d = np.load(DATA / f"{mat}_300K_bands.npz")
    x, E, ticks, vbm = d["x"], d["E"] - float(d["vbm"]), d["ticks"], float(d["vbm"])
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for b in range(E.shape[0]):
        ax.plot(x, E[b], color=TEAL, lw=1.1)
    ax.axhline(0, color=GOLD, lw=0.9, ls="--")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["X", "$\\Gamma$", "L"])
    for t in ticks[1:-1]:
        ax.axvline(t, color=GRAY, lw=0.6, ls=":")
    ax.set_xlim(x[0], x[-1])
    top = max(E[8]) + 0.8
    ax.set_ylim(-5.2, min(top, 6.5))
    ax.set_ylabel("$E - E_{VBM}$ (eV)")
    ax.set_title(f"{mat}, 300 K", color=GRAY)
    fig.tight_layout()
    fig.savefig(FIG / f"fig_atlas_{mat}_bands.png", dpi=200)
    plt.close(fig)


def resid_fig(mat, T=300):
    zs = zscores(mat, T)
    labels = [LABELS[k] for k, _ in zs]
    vals = np.array([z for _, z in zs])
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    y = np.arange(len(zs))[::-1]
    cap = 8.0
    vc = np.clip(vals, -cap, cap)
    ax.hlines(y, 0, vc, color=TEAL, lw=1.4)
    ax.plot(vc, y, "o", color=TEAL, ms=5)
    over = np.abs(vals) > cap
    if over.any():
        ax.plot(vc[over], y[over], marker=">", color=RED, ms=8)
    ax.axvline(0, color=GRAY, lw=0.8)
    for s in (-1, 1):
        ax.axvline(s, color=GOLD, lw=0.7, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(-cap - 0.4, cap + 0.4)
    ax.set_xlabel("residual / $\\sigma$")
    ax.set_title(f"{mat}, {T} K — residuals", color=GRAY)
    fig.tight_layout()
    fig.savefig(FIG / f"fig_atlas_{mat}_resid{T}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    for mat in ["GaAs", "InAs", "GaSb", "AlSb", "InSb", "AlAs", "InP", "GaP"]:
        bands_fig(mat)
        resid_fig(mat, 300)
        resid_fig(mat, 77)
        print(mat, "sekiller tamam")
