"""
Pool campaign fit — rich scheme (51 PW + full mask + beta free),
generalised to any zinc-blende binary and temperature.

    python3 fit_pool.py GaAs 77
    python3 fit_pool.py InAs 300

Targets live in TARGETS below, one row per (material, T). Campaign
rule: a row marked verified=False is a WORKING row (plan-table values);
its fit is exploratory and must not be registered as a locked anchor
until the row is checked line-by-line against Vurgaftman et al. (2001)
(or the Ioffe database reproduction of it).
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import fit_gaas_300k as F

# ---------------------------------------------------------------- targets
# GaAs rows: verified (Katman 1-2 + Varshni for 77 K; valley energies at
# 77 K use the Gamma Varshni shift — flagged in the fit report).
# Antimonide rows: WORKING values from files/Malzeme-Havuzu-Plani.md.
def _row(a0, Eg, EL, EX, Dso, me, g1, g2, g3, mtL, mlL, mtX, mlX, verified):
    hh100 = 1.0 / (g1 - 2 * g2); lh100 = 1.0 / (g1 + 2 * g2)
    hh111 = 1.0 / (g1 - 2 * g3); lh111 = 1.0 / (g1 + 2 * g3)
    return {"a0": a0, "verified": verified, "targets": {
        "Eg": (Eg, 0.010), "E_L": (EL, 0.010), "E_X": (EX, 0.010),
        "Dso": (Dso, 0.005),
        "me100": (me, 0.03), "mlh100": (lh100, 0.03), "mhh100": (hh100, 0.03),
        "me111": (me, 0.03), "mlh111": (lh111, 0.03), "mhh111": (hh111, 0.03),
        "dip_L": (0.0, 0.003),
        "mt_L": (mtL, 0.07), "ml_Lmin": (mlL, 0.10),
        "mt_X": (mtX, 0.07), "ml_X": (mlX, 0.20)}}

TARGETS = {
    ("GaAs", 300): _row(5.653e-10, 1.424, 1.708, 1.900, 0.341,
                        0.067, 6.98, 2.06, 2.93, 0.075, 1.9, 0.19, 1.9, True),
    ("GaAs", 77): _row(5.648e-10, 1.508, 1.804, 1.970, 0.341,
                       0.067, 6.98, 2.06, 2.93, 0.075, 1.9, 0.19, 1.9, True),
    ("InAs", 300): _row(6.0583e-10, 0.354, 1.08, 1.37, 0.39,
                        0.023, 20.0, 8.5, 9.2, 0.07, 1.87, 0.16, 0.9, False),
    ("InAs", 77): _row(6.052e-10, 0.407, 1.13, 1.42, 0.39,
                       0.023, 20.0, 8.5, 9.2, 0.07, 1.87, 0.16, 0.9, False),
    ("GaSb", 300): _row(6.0959e-10, 0.726, 0.81, 1.03, 0.76,
                        0.0412, 13.4, 4.7, 6.0, 0.11, 0.95, 0.22, 1.51, True),
    ("GaSb", 77): _row(6.090e-10, 0.800, 0.888, 1.126, 0.76,
                       0.0412, 13.4, 4.7, 6.0, 0.11, 0.95, 0.22, 1.51, True),
    ("AlSb", 300): _row(6.1355e-10, 2.300, 2.20, 1.62, 0.676,
                        0.14, 5.18, 1.19, 1.97, 0.10, 1.0, 0.19, 1.1, False),
    ("AlSb", 77): _row(6.130e-10, 2.37, 2.24, 1.66, 0.676,
                       0.14, 5.18, 1.19, 1.97, 0.10, 1.0, 0.19, 1.1, False),
    ("InSb", 300): _row(6.4794e-10, 0.170, 0.68, 1.00, 0.81,
                        0.0135, 34.8, 15.5, 16.5, 0.08, 1.0, 0.20, 0.9, False),
    ("InSb", 77): _row(6.473e-10, 0.228, 0.74, 1.06, 0.81,
                       0.0135, 34.8, 15.5, 16.5, 0.08, 1.0, 0.20, 0.9, False),
    ("AlAs", 300): _row(5.6611e-10, 3.03, 2.36, 2.16, 0.28,
                        0.15, 3.76, 0.82, 1.42, 0.11, 1.0, 0.19, 1.1, False),
    ("AlAs", 77): _row(5.657e-10, 3.09, 2.42, 2.21, 0.28,
                       0.15, 3.76, 0.82, 1.42, 0.11, 1.0, 0.19, 1.1, False),
    ("InP", 300): _row(5.8687e-10, 1.344, 1.93, 2.19, 0.11,
                       0.0795, 5.08, 1.60, 2.10, 0.09, 1.2, 0.20, 1.0, False),
    ("InP", 77): _row(5.864e-10, 1.418, 2.00, 2.34, 0.11,
                      0.0795, 5.08, 1.60, 2.10, 0.09, 1.2, 0.20, 1.0, False),
    ("GaP", 300): _row(5.4505e-10, 2.78, 2.60, 2.26, 0.08,
                       0.09, 4.05, 0.49, 1.25, 0.15, 1.2, 0.22, 1.12, False),
    ("GaP", 77): _row(5.447e-10, 2.86, 2.67, 2.33, 0.08,
                      0.09, 4.05, 0.49, 1.25, 0.15, 1.2, 0.22, 1.12, False),
}
MASS_KEYS = {"me100", "mlh100", "mhh100", "me111", "mlh111", "mhh111",
             "mt_L", "ml_Lmin", "mt_X", "ml_X", "mDOS_L", "mDOS_X"}

# Valley-mass verification (2026-11-19, Ioffe NSM pages): for these
# materials the archive lists only density-of-states valley masses, so
# the separate ml/mt targets are replaced by DOS-mass targets.
DOS_VALLEY = {"InAs": {"mDOS_L": (0.29, 0.07), "mDOS_X": (0.64, 0.07)},
              "InSb": {"mDOS_L": (0.25, 0.07)},
              "InP": {"mDOS_L": (0.63, 0.07), "mDOS_X": (0.66, 0.07)}}
NAMES = ["VS3", "VS4", "VS8", "VS11", "VA3", "VA4", "VA8", "VA11", "L", "beta"]

# Literature seeds: Chelikowsky & Cohen, PRB 14, 556 (1976), Table VIII;
# AlSb from Cohen & Bergstresser, PR 141, 789 (1966), Table II.
# (VS3, VS8, VS11, VA3, VA4, VA11) in Ry; VS4 = VA8 = 0 by symmetry.
LIT_SEEDS = {
    "GaP":  (-0.230, 0.020, 0.057, 0.100, 0.070, 0.025),
    "GaSb": (-0.220, 0.005, 0.045, 0.040, 0.030, 0.000),
    "InP":  (-0.235, 0.000, 0.053, 0.080, 0.060, 0.030),
    "InAs": (-0.230, 0.000, 0.045, 0.055, 0.045, 0.010),
    "InSb": (-0.200, -0.010, 0.044, 0.044, 0.030, 0.015),
    "AlSb": (-0.21, 0.02, 0.06, 0.06, 0.04, 0.02),
}


def setup_basis(a0):
    """Rebuild every a0-dependent global of the machinery at 51 PW."""
    F.A0 = a0
    vecs = []
    for n in itertools.product(range(-3, 4), repeat=3):
        q2 = sum(x * x for x in n)
        if q2 <= 11 and (all(x % 2 == 0 for x in n) or all(x % 2 == 1 for x in n)):
            vecs.append(n)
    vecs.sort(key=lambda v: (sum(x * x for x in v), v))
    F.G_INT = vecs; F.NB = len(vecs)
    F.GI = np.array(vecs, dtype=int)
    F.GV = F.GI.astype(float) * (2 * np.pi / a0)
    F.DN = F.GI[:, None, :] - F.GI[None, :, :]
    F.Q2 = np.sum(F.DN ** 2, axis=2)
    F.PHASE = (np.pi / 4.0) * np.sum(F.DN, axis=2)
    F.DG = F.GV[:, None, :] - F.GV[None, :, :]
    F.CROSS0 = np.cross(F.DG, F.GV[None, :, :])
    F.U = 2 * np.pi / a0
    F.K_GAMMA = np.zeros(3)
    F.K_L = 0.5 * F.U * np.array([1.0, 1.0, 1.0])
    F.K_X = F.U * np.array([1.0, 0.0, 0.0])


def observables10(p10, U, H22):
    V = F.build_V(p10[:8])
    L, beta = p10[8], p10[9]
    KG, KL, KX = F.K_GAMMA, F.K_L, F.K_X
    eg = F.eigs(KG, V, L, beta); el = F.eigs(KL, V, L, beta)
    ex = F.eigs(KX, V, L, beta)
    vbm = eg[7]
    o = {"Eg": eg[8] - vbm, "E_L": el[8] - vbm, "E_X": ex[8] - vbm,
         "Dso": eg[7] - eg[3]}
    me, mlh, mhh = F.parabolic_masses(V, L, beta, [1, 0, 0])
    o.update({"me100": me, "mlh100": mlh, "mhh100": mhh})
    me, mlh, mhh = F.parabolic_masses(V, L, beta, [1, 1, 1])
    o.update({"me111": me, "mlh111": mlh, "mhh111": mhh})
    axis = np.array([1.0, 1.0, 1.0])
    ts = np.linspace(0.40, 0.50, 11)
    Es = np.array([F.eigs(t * U * axis, V, L, beta)[8] for t in ts])
    i = int(np.argmin(Es))
    o["dip_L"] = el[8] - Es[i]

    def pair_mean(k):
        w = F.eigs(k, V, L, beta)
        return 0.5 * (w[8] + w[9])

    # Valley longitudinal masses: SYMMETRIC second difference at the
    # valley minimum, a = (E+ + E- - 2E0) / (2 h^2), m = H22 / a
    # (Kazim decision 2026-11-19, diagnosis in
    # files/Katman-3-Egrilik-Tahmincisi-Tani.md: the old one-sided
    # estimators measured the anticrossing shoulder, not curvature;
    # the symmetric estimator is window-stable, e.g. GaAs ml_X = 1.06
    # for h = 0.005..0.04, and agrees with bulk_properties.valley_masses).
    h_t = 0.01
    dk_L = h_t * U * np.sqrt(3.0)
    t_min = ts[i]
    a = (pair_mean((t_min + h_t) * U * axis)
         + pair_mean((t_min - h_t) * U * axis)
         - 2 * pair_mean(t_min * U * axis)) / (2 * dk_L ** 2)
    o["ml_Lmin"] = abs(H22 / a) if a > 0 else 50.0

    dl = 0.02
    for name, k0, tdir in (("mt_L", KL, np.array([1.0, -1.0, 0.0]) / np.sqrt(2)),
                           ("mt_X", KX, np.array([0.0, 1.0, 0.0]))):
        e0 = pair_mean(k0)
        a = (pair_mean(k0 + dl * U * tdir) + pair_mean(k0 - dl * U * tdir)
             - 2 * e0) / (2 * (dl * U) ** 2)
        o[name] = abs(H22 / a) if a > 0 else 50.0
    e0 = pair_mean(KX)
    a = (pair_mean(KX + dl * U * np.array([1.0, 0.0, 0.0]))
         + pair_mean(KX - dl * U * np.array([1.0, 0.0, 0.0]))
         - 2 * e0) / (2 * (dl * U) ** 2)
    o["ml_X"] = abs(H22 / a) if a > 0 else 50.0
    o["mDOS_L"] = (16.0 * o["ml_Lmin"] * o["mt_L"] ** 2) ** (1.0 / 3.0)
    o["mDOS_X"] = (9.0 * o["ml_X"] * o["mt_X"] ** 2) ** (1.0 / 3.0)
    return o


def main():
    material, T = sys.argv[1], int(sys.argv[2])
    row = TARGETS[(material, T)]
    tg = dict(row["targets"])
    if material == "AlAs":
        # Satellite-weight variant (Kazim B-package, 2026-11-19): AlAs
        # is indirect; its Gamma and X edges became load-bearing
        # through AlGaAs, so their sigmas are tightened (edges 10 -> 5
        # meV, ml_X 0.20 -> 0.10, mt_X 0.07 -> 0.03).
        tg = {k: (v[0], 0.005 if k in ("Eg", "E_X") else
                  (0.10 if k == "ml_X" else
                   (0.03 if k == "mt_X" else v[1])))
              for k, v in tg.items()}
    if material in ("InAs", "InSb"):
        # Narrow-gap weighting variant (documented): the parabolic
        # Gamma mass of a narrow-gap material is an edge-of-validity
        # target (Kane coupling ties it to the gap); prioritise the
        # gap itself (sigma 5 meV) and relax mass sigmas x2.
        tg = {k: (v[0], 0.005 if k == "Eg" else (v[1] * 2 if k in MASS_KEYS else v[1]))
              for k, v in tg.items()}
        print(f"{material}: narrow-gap agirlik profili uygulaniyor", flush=True)
    if material in DOS_VALLEY:
        for k in ("mt_L", "ml_Lmin", "mt_X", "ml_X"):
            tg.pop(k, None)
        tg.update(DOS_VALLEY[material])
        print(f"{material}: DOS-kutle vadi hedefleri (dogrulama 2026-11-19)",
              flush=True)
    setup_basis(row["a0"])
    U, H22 = F.U, F.HBAR2_OVER_2M0
    keys = list(tg)

    def residuals(p10):
        o = observables10(p10, U, H22)
        r = []
        for k in keys:
            t, s = tg[k]
            sig = s * t if k in MASS_KEYS else s
            r.append((o[k] - t) / sig)
        return np.array(r)

    seed = np.array([-0.29139, -0.03745, 0.07922, 0.0,
                     0.14049, 0.04996, -0.0495, 0.01698, 2139.59, 1.17143])
    if len(sys.argv) > 3:  # seed override: pool entry name or lit:<Material>
        if sys.argv[3].startswith("lit:"):
            v = LIT_SEEDS[sys.argv[3][4:]]
            seed = np.array([v[0], 0.0, v[1], v[2], v[3], v[4], 0.0, v[5],
                             2139.59, 1.17143])
        else:
            sp = json.loads((Path(__file__).parent / "params_pool" / sys.argv[3]
                             / "params.json").read_text())["params_Ry_and_scalars"]
            seed = np.array([sp["VS3"], sp["VS4"], sp["VS8"], sp["VS11"],
                             sp["VA3"], sp["VA4"], sp["VA8"], sp["VA11"],
                             sp["L"], sp["beta"]])
        print(f"seed override: {sys.argv[3]}", flush=True)
    lo = np.array(F.BOUNDS_LO, dtype=float); hi = np.array(F.BOUNDS_HI, dtype=float)
    seed = np.minimum(np.maximum(seed, lo + 1e-9 * (1 + np.abs(lo))),
                      hi - 1e-9 * (1 + np.abs(hi)))  # keep x0 strictly feasible
    t0 = time.time()
    print(f"{material} {T}K seed cost:", round(float(np.sum(residuals(seed) ** 2)), 1),
          flush=True)
    s = least_squares(residuals, seed, bounds=(F.BOUNDS_LO, F.BOUNDS_HI),
                      method="trf", xtol=1e-10, ftol=1e-10, gtol=1e-10,
                      max_nfev=400)
    o = observables10(s.x, U, H22)
    out = {"material": material, "T_K": T, "verified_targets": row["verified"],
           "scheme": "51PW + full mask + beta free",
           "params": {n: float(v) for n, v in zip(NAMES, s.x)},
           "cost": float(2 * s.cost),
           "observables": {k: float(v) for k, v in o.items()},
           "targets": {k: list(v) for k, v in tg.items()},
           "seconds": round(time.time() - t0, 1)}
    path = f"../hidden_files/fit_pool_{material}_{T}K.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
