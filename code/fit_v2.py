"""
Fit v2 - classic mask + L/X valley-shape targets (Kazim's instruction,
2026-11-19: add the curvature targets NOW, not at Layer 4).

Free parameters: VS3, VS8, VS11, VA3, VA11, L_so  (beta fixed 1/0.9).
Targets: the Layer-1 ten (edges, Delta_so, Gamma masses) PLUS
  * dip depth  E_L - E_min(Lambda line)      -> 0      (sigma 3 meV)
  * m_t(L)                                    -> 0.075  (sigma 7%)
  * m_l at the Lambda-line minimum            -> 1.9    (sigma 10%)
  * m_t(X)                                    -> 0.23   (sigma 7%)
  * m_l(X) one-sided interior curvature       -> 1.3    (sigma 20%)
Literature valley masses: Vurgaftman et al. (2001) GaAs values.
Usage: python3 fit_v2.py <Material> <T>   (only GaAs 300 wired so far;
the target table below is the extension point for the pool campaign.)
"""
import json
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution

import fit_gaas_300k as F

U = F.U                      # 2 pi / a  [m^-1]
H22 = F.HBAR2_OVER_2M0       # hbar^2 / 2 m0 [eV m^2]
KG, KL, KX = F.K_GAMMA, F.K_L, F.K_X

TARGETS = {
    # name: (target, sigma)  -- masses: sigma is relative
    "Eg": (1.424, 0.010), "E_L": (1.708, 0.010), "E_X": (1.900, 0.010),
    "Dso": (0.341, 0.005),
    "me100": (0.067, 0.03), "mlh100": (0.090, 0.03), "mhh100": (0.350, 0.03),
    "me111": (0.067, 0.03), "mlh111": (0.078, 0.03), "mhh111": (0.893, 0.03),
    "dip_L": (0.0, 0.003),
    "mt_L": (0.075, 0.07), "ml_Lmin": (1.9, 0.10),
    "mt_X": (0.23, 0.07), "ml_X": (1.3, 0.20),
}
MASS_KEYS = {"me100", "mlh100", "mhh100", "me111", "mlh111", "mhh111",
             "mt_L", "ml_Lmin", "mt_X", "ml_X"}


def _pair_mean_e(k, V, L, beta):
    w = F.eigs(k, V, L, beta)
    return 0.5 * (w[8] + w[9])


def observables(p6):
    VS3, VS8, VS11, VA3, VA11, L = p6
    V = F.build_V([VS3, 0.0, VS8, VS11, VA3, 0.0, 0.0, VA11])
    beta = F.BETA if hasattr(F, "BETA") else 1.0 / 0.9
    eg = F.eigs(KG, V, L, beta)
    el = F.eigs(KL, V, L, beta)
    ex = F.eigs(KX, V, L, beta)
    vbm = eg[7]
    o = {"Eg": eg[8] - vbm, "E_L": el[8] - vbm, "E_X": ex[8] - vbm,
         "Dso": eg[7] - eg[3]}
    me, mlh, mhh = F.parabolic_masses(V, L, beta, [1, 0, 0])
    o.update({"me100": me, "mlh100": mlh, "mhh100": mhh})
    me, mlh, mhh = F.parabolic_masses(V, L, beta, [1, 1, 1])
    o.update({"me111": me, "mlh111": mlh, "mhh111": mhh})
    # --- L valley shape: Lambda-line scan near L
    axis = np.array([1.0, 1.0, 1.0])
    ts = np.linspace(0.40, 0.50, 11)
    Es = np.array([F.eigs(t * U * axis, V, L, beta)[8] for t in ts])
    i = int(np.argmin(Es))
    o["dip_L"] = (el[8] - Es[i])
    tmin = ts[i]
    h = 0.01
    dk = h * U * np.sqrt(3.0)
    if 0 < i < len(ts) - 1:
        a = (Es[i + 1] - 2 * Es[i] + Es[i - 1]) / dk ** 2
    else:  # minimum at L itself: one-sided interior curvature
        e_in = F.eigs((0.50 - h) * U * axis, V, L, beta)[8]
        a = (e_in - el[8]) / dk ** 2  # slope-contaminated; dip term guards
    o["ml_Lmin"] = abs(H22 / a) if a > 0 else 50.0
    # --- transverse masses (symmetric, pair mean, delta = 0.02 U)
    dl = 0.02
    for name, k0, tdir in (("mt_L", KL, np.array([1.0, -1.0, 0.0]) / np.sqrt(2)),
                           ("mt_X", KX, np.array([0.0, 1.0, 0.0]))):
        e0 = _pair_mean_e(k0, V, L, beta)
        ep = _pair_mean_e(k0 + dl * U * tdir, V, L, beta)
        em = _pair_mean_e(k0 - dl * U * tdir, V, L, beta)
        a = (ep + em - 2 * e0) / (2 * (dl * U) ** 2)
        o[name] = abs(H22 / a) if a > 0 else 50.0
    # --- X longitudinal: one-sided interior curvature (mirror symmetry)
    e0 = _pair_mean_e(KX, V, L, beta)
    ein = _pair_mean_e(KX - dl * U * np.array([1.0, 0.0, 0.0]), V, L, beta)
    a = (ein - e0) / (dl * U) ** 2
    o["ml_X"] = abs(H22 / a) if a > 0 else 50.0
    return o


def cost(p6):
    o = observables(p6)
    c = 0.0
    for k, (t, s) in TARGETS.items():
        sig = s * t if k in MASS_KEYS else s
        c += ((o[k] - t) / sig) ** 2
    return c


BOUNDS = [(-0.35, -0.15), (-0.05, 0.12), (0.0, 0.13),
          (0.0, 0.10), (0.0, 0.12), (2000.0, 8000.0)]
INCUMBENT = [-0.27580, 0.04202, 0.06032, 0.02748, 0.05437, 4580.31]


def main():
    t0 = time.time()
    print("incumbent cost:", round(cost(INCUMBENT), 3), flush=True)
    init = np.tile(INCUMBENT, (60, 1))
    rng = np.random.default_rng(7)
    for j, (lo, hi) in enumerate(BOUNDS):
        init[1:, j] = rng.uniform(lo, hi, 59)
    res = differential_evolution(cost, BOUNDS, init=init, maxiter=250,
                                 tol=1e-7, polish=True, seed=7, workers=1,
                                 updating="immediate")
    o = observables(res.x)
    out = {"params": {"VS3": res.x[0], "VS8": res.x[1], "VS11": res.x[2],
                      "VA3": res.x[3], "VA11": res.x[4], "L_so": res.x[5],
                      "beta": 1.0 / 0.9},
           "cost": float(res.fun), "observables": {k: float(v) for k, v in o.items()},
           "targets": {k: list(v) for k, v in TARGETS.items()},
           "seconds": round(time.time() - t0, 1)}
    path = "../hidden_files/fit_v2_gaas_300k.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("saved:", path)


if __name__ == "__main__":
    main()
