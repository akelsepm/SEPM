"""
Chapter 8 figures: AlGaAs validation panels (square, book palette).
Data: the final ternary grids (hidden_files/ternary_algaas_*.json,
bowing active) + the T scan; the no-bowing comparison curve is
computed live by toggling pool.BOWING off.
Output: files/book-bolum-01/ternary_data/fig_ch8_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fit_pool
import pool as POOL
from bulk_properties import Bulk

HERE = Path(__file__).parent
HID = HERE.parent / "hidden_files"
FIG = HERE.parent / "files" / "book-bolum-01" / "ternary_data"
FIG.mkdir(parents=True, exist_ok=True)
TEAL, GRAY, RED, GOLD = "#0f766e", "#44403c", "#b91c1c", "#b98a1f"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRAY,
                     "text.color": GRAY, "axes.labelcolor": GRAY,
                     "xtick.color": GRAY, "ytick.color": GRAY})

G300 = {r["x"]: r for r in json.loads((HID / "ternary_algaas_300k.json").read_text())}
G77 = {r["x"]: r for r in json.loads((HID / "ternary_algaas_77k.json").read_text())}
XS = np.array(sorted(G300))
LIT300 = {"G": lambda x: 1.425 + 1.155 * x + 0.37 * x ** 2,
          "X": lambda x: 1.900 + 0.125 * x + 0.143 * x ** 2,
          "L": lambda x: 1.708 + 0.642 * x}
LIT77 = {"G": lambda x: 1.519 + 1.155 * x + 0.37 * x ** 2 - 0.0114,
         "X": lambda x: 1.981 + 0.124 * x + 0.144 * x ** 2 - 0.0097,
         "L": lambda x: 1.815 + 0.69 * x - 0.0128}
xf = np.linspace(0, 1, 200)


def gaps_panel(grid, lit, title, fname):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for key, col, lab in (("Eg_Gamma_eV", TEAL, r"$E_\Gamma$ (ours)"),
                          ("E_L_eV", GOLD, r"$E_L$ (ours)"),
                          ("E_X_eV", RED, r"$E_X$ (ours)")):
        ax.plot(XS, [grid[x]["edges"][key] for x in XS], "-", color=col,
                lw=1.8, label=lab)
    for key, lk in (("Eg_Gamma_eV", "G"), ("E_L_eV", "L"), ("E_X_eV", "X")):
        ax.plot(xf, lit[lk](xf), "--", color=GRAY, lw=1.0, alpha=0.75)
    ax.plot([], [], "--", color=GRAY, lw=1.0, label="literature forms")
    ax.set_xlabel("x  (Al$_x$Ga$_{1-x}$As)")
    ax.set_ylabel("valley energy (eV)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


gaps_panel(G300, LIT300, "300 K: valleys and literature forms", "fig_ch8_gaps300.png")
gaps_panel(G77, LIT77, "77 K: valleys and literature forms", "fig_ch8_gaps77.png")

# ---- residuals, two square panels side by side
fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
for ax, grid, lit, title in ((axes[0], G300, LIT300, "300 K residuals (meV)"),
                             (axes[1], G77, LIT77, "77 K residuals (meV)")):
    for key, lk, col, lab, off in (("Eg_Gamma_eV", "G", TEAL, r"$\Gamma$", -0.012),
                                   ("E_L_eV", "L", GOLD, "L", 0.0),
                                   ("E_X_eV", "X", RED, "X", 0.012)):
        r = np.array([1000 * (grid[x]["edges"][key] - lit[lk](x)) for x in XS])
        ax.vlines(XS + off, 0, r, color=col, lw=1.2)
        ax.plot(XS + off, r, "o", color=col, ms=4, label=lab)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(-0.05, 1.05)
fig.tight_layout()
fig.savefig(FIG / "fig_ch8_resid.png", dpi=200)
plt.close(fig)

# ---- bowing before/after on E_X (300 K)
saved = dict(POOL.BOWING)
POOL.BOWING = {}
ex_nobow = []
for x in XS:
    mat = POOL.get_params("AlGaAs", x=float(x), T=300)
    fit_pool.setup_basis(mat["a0"])
    ex_nobow.append(Bulk(mat).edges()["E_X_eV"])
POOL.BOWING = saved
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot(xf, LIT300["X"](xf), "--", color=GRAY, lw=1.1, label="VMR $E_X(x)$")
ax.plot(XS, ex_nobow, "-", color=GRAY, lw=1.4, alpha=0.55, label="no bowing")
ax.plot(XS, [G300[x]["edges"]["E_X_eV"] for x in XS], "-", color=TEAL, lw=1.9,
        label="with bowing (final)")
ax.set_xlabel("x")
ax.set_ylabel("$E_X$ (eV)")
ax.set_title("VA4 bowing layer: before / after")
ax.legend(fontsize=8, frameon=False)
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(FIG / "fig_ch8_bowing.png", dpi=200)
plt.close(fig)

# ---- m_e(x) and Delta_so(x) at 300 K
fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
ax = axes[0]
ax.plot(XS, [G300[x]["m100"]["m_e"] for x in XS], "o-", color=TEAL, lw=1.6, label="ours")
ax.plot(xf, 0.063 + 0.083 * xf, "--", color=GRAY, lw=1.1, label="Ioffe: 0.063+0.083x")
ax.plot(xf, 0.067 + 0.083 * xf, ":", color=GRAY, lw=1.1, label="VMR (low T): 0.067+0.083x")
ax.set_xlabel("x"); ax.set_ylabel("$m_e$ ($m_0$)")
ax.set_title("Electron mass"); ax.legend(fontsize=8, frameon=False); ax.set_xlim(0, 1)
ax = axes[1]
ax.plot(XS, [G300[x]["edges"]["Delta_so_eV"] for x in XS], "o-", color=TEAL, lw=1.6, label="ours")
ax.plot(xf, 0.341 - 0.061 * xf, "--", color=GRAY, lw=1.1, label="VMR: 0.341−0.061x")
ax.plot(xf, 0.340 - 0.040 * xf, ":", color=GRAY, lw=1.1, label="Adachi: 0.340−0.040x")
ax.set_xlabel("x"); ax.set_ylabel(r"$\Delta_{so}$ (eV)")
ax.set_title("Spin-orbit splitting"); ax.legend(fontsize=8, frameon=False); ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(FIG / "fig_ch8_mass_so.png", dpi=200)
plt.close(fig)

# ---- T behaviour at x = 0.3 (Gamma track)
ts = json.loads((HID / "ternary_algaas_x03_Tscan.json").read_text())
TT = np.array([r["T_K"] for r in ts])
EG = np.array([r["edges"]["Eg_Gamma_eV"] for r in ts])
Tf = np.linspace(0, 300, 200)
varshni = 1.8988 - 0.0005405 * Tf ** 2 / (Tf + 204)
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot(Tf, varshni, "--", color=GRAY, lw=1.1, label="Ioffe–Varshni prescription")
ax.plot(TT, EG, "o-", color=TEAL, lw=1.6, label="ours (parameter interpolation)")
ax.set_xlabel("T (K)"); ax.set_ylabel("$E_\\Gamma$ (eV)")
ax.set_title("x = 0.3: temperature behaviour")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig_ch8_T.png", dpi=200)
plt.close(fig)

print("figs:", sorted(p.name for p in FIG.glob("fig_ch8_*.png")))
