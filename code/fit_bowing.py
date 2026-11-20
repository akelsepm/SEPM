"""
Alloy-layer fit (Chapter 8 package, 2026-11-19): one bowing
coefficient for AlGaAs on the VA4 form factor, fitted by least
squares to the VMR E_X(x) curve at 300 K on the FROZEN v3+v3 anchor
pair. The term b * x * (1 - x) vanishes at both endpoints, so the
binary anchors are untouched by construction (Kazim's loop idea,
PoC in files/Katman-3-Bowing-Denemesi-01.md). The 77 K path is a
transfer test, not a fit input.

Writes ../hidden_files/bowing_algaas.json with the coefficient and
the before/after validation metrics at both temperatures.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import fit_pool
from bulk_properties import Bulk
from pool import get_params

HID = Path(__file__).parent.parent / "hidden_files"
XS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
LIT = {
    300: {"G": lambda x: 1.425 + 1.155 * x + 0.37 * x ** 2,
          "X": lambda x: 1.900 + 0.125 * x + 0.143 * x ** 2,
          "L": lambda x: 1.708 + 0.642 * x},
    77: {"G": lambda x: 1.519 + 1.155 * x + 0.37 * x ** 2 - 0.0114,
         "X": lambda x: 1.981 + 0.124 * x + 0.144 * x ** 2 - 0.0097,
         "L": lambda x: 1.815 + 0.69 * x - 0.0128},
}


def edges_b(x, T, b):
    mat = get_params("AlGaAs", x=float(x), T=T)
    m2 = {k: (dict(v) if isinstance(v, dict) else v)
          for k, v in mat.items() if k != "provenance"}
    m2["VA_RY"][4] += b * x * (1 - x)
    fit_pool.setup_basis(m2["a0"])
    return Bulk(m2).edges()


def metrics(T, b):
    lit = LIT[T]
    g, xx, ll = [], [], []
    for x in XS:
        e = edges_b(x, T, b)
        g.append(1000 * (e["Eg_Gamma_eV"] - lit["G"](x)))
        xx.append(1000 * (e["E_X_eV"] - lit["X"](x)))
        if x <= 0.5:
            ll.append(1000 * (e["E_L_eV"] - lit["L"](x)))
    g, xx, ll = map(np.array, (g, xx, ll))
    return {"G_rms": float(np.sqrt(np.mean(g ** 2))),
            "G_at_03": float(g[2]), "G_max_045": float(np.max(np.abs(g[XS <= 0.45]))),
            "X_rms": float(np.sqrt(np.mean(xx ** 2))),
            "X_at_03": float(xx[2]),
            "L_rms_05": float(np.sqrt(np.mean(ll ** 2)))}


def main():
    resid = lambda b: np.array([edges_b(x, 300, b[0])["E_X_eV"] - LIT[300]["X"](x)
                                for x in XS])
    sol = least_squares(resid, [0.0], xtol=1e-13, ftol=1e-13)
    b = float(sol.x[0])
    out = {"system": "AlGaAs", "key": "VA4", "b_Ry": b,
           "fitted_on": "E_X(x) vs VMR (2001), 300 K, x = 0.1..0.9",
           "anchors": "GaAs v3 + AlAs v3 (frozen pair)",
           "before": {str(T): metrics(T, 0.0) for T in (300, 77)},
           "after": {str(T): metrics(T, b) for T in (300, 77)}}
    (HID / "bowing_algaas.json").write_text(json.dumps(out, indent=1))
    print(f"b(VA4, AlGaAs) = {b:+.5f} Ry")
    for T in (300, 77):
        bf, af = out["before"][str(T)], out["after"][str(T)]
        print(f"{T}K: X rms {bf['X_rms']:.0f} -> {af['X_rms']:.0f} meV | "
              f"L rms {bf['L_rms_05']:.0f} -> {af['L_rms_05']:.0f} | "
              f"G(.3) {bf['G_at_03']:+.1f} -> {af['G_at_03']:+.1f} | "
              f"Gmax(.45) {bf['G_max_045']:.1f} -> {af['G_max_045']:.1f}")
    print("saved hidden_files/bowing_algaas.json")


if __name__ == "__main__":
    main()
