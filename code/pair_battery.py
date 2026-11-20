"""
Pair acceptance battery (Chapter 8 package, 2026-11-19): the
GaAs+AlAs pair is judged as a unit under Kazim's Gamma-priority
rule — the Gamma track must not degrade beyond a few meV versus the
delivered v3/v3 path; among the pairs that pass, the X/L tracks
decide. Candidates: GaAs {v3 pool anchor, ridge lambda=1} x
AlAs {v3 pool anchor, batch-round v4 fit}, at 300 K and 77 K.

Metrics on the alloy path (VCA, pool._lerp machinery by hand so
candidate anchors can be substituted):
  * Gamma: max |Eg - lit| on x in [0.1, 0.45] and [0.1, 0.9], and at 0.3
  * X: rms |E_X - lit| on [0.1, 0.9]; crossover x_c
  * L: rms |E_L - lit| on [0.1, 0.5] (range of the literature forms)
Literature: 300 K Lee (Gamma), VMR (X), Casey-Panish (L);
77 K Ioffe prescriptions (0 K forms minus Varshni shifts).
Output: ../hidden_files/pair_battery.json + printed table.
"""
import json
from pathlib import Path

import numpy as np

import fit_pool
from bulk_properties import Bulk
from pool import binary_params

HERE = Path(__file__).parent
HID = HERE.parent / "hidden_files"

LIT = {
    300: {"G": lambda x: 1.425 + 1.155 * x + 0.37 * x ** 2,
          "X": lambda x: 1.900 + 0.125 * x + 0.143 * x ** 2,
          "L": lambda x: 1.708 + 0.642 * x},
    77: {"G": lambda x: 1.519 + 1.155 * x + 0.37 * x ** 2 - 0.0114,
         "X": lambda x: 1.981 + 0.124 * x + 0.144 * x ** 2 - 0.0097,
         "L": lambda x: 1.815 + 0.69 * x - 0.0128},
}
XS = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9])


def from_fit_json(path):
    d = json.loads(Path(path).read_text())
    p = d["params"]
    return {"VS_RY": {q: p[f"VS{q}"] for q in (3, 4, 8, 11)},
            "VA_RY": {q: p[f"VA{q}"] for q in (3, 4, 8, 11)},
            "L_SO": p["L"], "BETA": p["beta"]}


def strip(mat):
    return {k: mat[k] for k in ("a0", "VS_RY", "VA_RY", "L_SO", "BETA")}


def mix(ga, al, x):
    return {"a0": ga["a0"] + x * (al["a0"] - ga["a0"]),
            "VS_RY": {q: ga["VS_RY"][q] + x * (al["VS_RY"][q] - ga["VS_RY"][q])
                      for q in ga["VS_RY"]},
            "VA_RY": {q: ga["VA_RY"][q] + x * (al["VA_RY"][q] - ga["VA_RY"][q])
                      for q in ga["VA_RY"]},
            "L_SO": ga["L_SO"] + x * (al["L_SO"] - ga["L_SO"]),
            "BETA": ga["BETA"] + x * (al["BETA"] - ga["BETA"])}


def evaluate(ga, al, T):
    lit = LIT[T]
    rows = []
    for x in XS:
        mat = mix(ga, al, float(x))
        fit_pool.setup_basis(mat["a0"])
        e = Bulk(mat).edges()
        rows.append((float(x), e["Eg_Gamma_eV"], e["E_X_eV"], e["E_L_eV"]))
    g = np.array([1000 * (r[1] - lit["G"](r[0])) for r in rows])
    xx = np.array([1000 * (r[2] - lit["X"](r[0])) for r in rows])
    ll = np.array([1000 * (r[3] - lit["L"](r[0])) for r in rows if r[0] <= 0.5])
    xs = np.array([r[0] for r in rows])
    d = np.array([r[1] - r[2] for r in rows])
    xc = None
    for i in range(len(d) - 1):
        if d[i] < 0 and d[i + 1] >= 0:
            xc = float(xs[i] - d[i] * (xs[i + 1] - xs[i]) / (d[i + 1] - d[i]))
    i3 = int(np.where(xs == 0.3)[0][0])
    return {"G_max_045": float(np.max(np.abs(g[xs <= 0.45]))),
            "G_max_09": float(np.max(np.abs(g))),
            "G_at_03": float(g[i3]),
            "X_rms": float(np.sqrt(np.mean(xx ** 2))),
            "X_at_03": float(xx[i3]),
            "L_rms_05": float(np.sqrt(np.mean(ll ** 2))),
            "xc": xc}


def main():
    out = {}
    for T in (300, 77):
        ga_v3 = strip(binary_params("GaAs", T))
        al_v3 = strip(binary_params("AlAs", T))
        ga_r = from_fit_json(HID / f"fit_ridge_GaAs_{T}K_lam1.0.json")
        ga_r["a0"] = ga_v3["a0"]
        al_v4 = from_fit_json(HID / f"fit_pool_AlAs_{T}K.json")
        al_v4["a0"] = al_v3["a0"]
        for gname, ga in (("GaAs-v3", ga_v3), ("GaAs-ridge1", ga_r)):
            for aname, al in (("AlAs-v3", al_v3), ("AlAs-v4", al_v4)):
                m = evaluate(ga, al, T)
                out[f"{T}K {gname}+{aname}"] = m
                print(f"{T}K {gname:12s}+{aname:8s}: "
                      f"Gmax(.45)={m['G_max_045']:5.1f} Gmax(.9)={m['G_max_09']:5.1f} "
                      f"G(.3)={m['G_at_03']:+6.1f} | Xrms={m['X_rms']:5.1f} "
                      f"X(.3)={m['X_at_03']:+6.1f} | Lrms={m['L_rms_05']:5.1f} | "
                      f"xc={m['xc'] if m['xc'] is None else round(m['xc'], 3)}",
                      flush=True)
    (HID / "pair_battery.json").write_text(json.dumps(out, indent=1))
    print("saved hidden_files/pair_battery.json")


if __name__ == "__main__":
    main()
