"""
EPM form-factor FIT ENGINE v1 (standing tool, Layer 1.5)
=========================================================
Config-driven generalisation of fit_gaas_300k.py. One engine, many
materials / temperatures: each job is a CONFIG dict (lattice constant,
basis shells, target table with sigmas, free-parameter mask, bounds,
starts). Results are registered into the parameter POOL:

    code/params_pool/<ConfigId>/
        params.json   - locked/fitted parameter set + provenance
        report.md     - observables vs targets, residuals, notes
        bands.png     - band profile along L-Gamma-X
    code/params_pool/INDEX.md   - the pool registry

Physics core = fit_gaas_300k.py (verified mirror Hamiltonian).
Usage:
    python3 fit_engine.py verify GaAs_300K     # observables + register
    python3 fit_engine.py fit    GaAs_300K     # multi-start fit + register
"""

import itertools
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import least_squares

import fit_gaas_300k as F

POOL = pathlib.Path(__file__).resolve().parent / "params_pool"
NAMES = ["VS3", "VS4", "VS8", "VS11", "VA3", "VA4", "VA8", "VA11", "L", "beta"]

# ---------------------------------------------------------------------
# CONFIGS - add one entry per material/temperature job.
# targets: key -> (value, sigma); masses use relative sigma (fraction).
# ---------------------------------------------------------------------
CONFIGS = {
    "GaAs_300K": {
        "a0": 5.653e-10,
        "basis_q2": [0, 3, 4],                    # 15 plane waves
        "targets": {
            "Eg_Gamma": (1.424, 0.010), "Eg_L": (1.708, 0.010),
            "Eg_X": (1.900, 0.010), "Delta_so": (0.341, 0.005),
            "m_e_100": (0.067, 0.03), "m_lh_100": (0.090, 0.03),
            "m_hh_100": (0.350, 0.03), "m_e_111": (0.067, 0.03),
            "m_lh_111": (0.078, 0.03), "m_hh_111": (0.893, 0.03),
        },
        "target_source": "Vurgaftman et al., J. Appl. Phys. 89, 5815 (2001), 300 K",
        # classic mask, beta fixed at 1/0.9 (locked decision 2026-11-18)
        "free": ["VS3", "VS8", "VS11", "VA3", "VA11", "L"],
        "fixed": {"VS4": 0.0, "VA4": 0.0, "VA8": 0.0, "beta": 1.0 / 0.9},
        "start": {"VS3": -0.248, "VS8": 0.0, "VS11": 0.0907,
                  "VA3": 0.090, "VA11": 0.014, "L": 3900.0},
        "bounds": {"VS3": (-0.45, -0.05), "VS8": (-0.10, 0.15),
                   "VS11": (0.0, 0.20), "VA3": (0.0, 0.20),
                   "VA11": (0.0, 0.15), "L": (500.0, 9000.0)},
        "locked_params": {"VS3": -0.27580, "VS4": 0.0, "VS8": 0.04202,
                          "VS11": 0.06032, "VA3": 0.02748, "VA4": 0.0,
                          "VA8": 0.0, "VA11": 0.05437, "L": 4580.31,
                          "beta": 1.0 / 0.9},
        "note": "Locked set B (Kazim, 2026-11-18). See Fit-Raporu-01.",
    },
}


def setup(cfg):
    """Rebuild the physics module's structure globals for this config."""
    F.A0 = cfg["a0"]
    F.U = 2 * np.pi / F.A0
    vecs = []
    for n in itertools.product(range(-3, 4), repeat=3):
        q2 = sum(x * x for x in n)
        if q2 in cfg["basis_q2"] and (all(x % 2 == 0 for x in n)
                                      or all(x % 2 == 1 for x in n)):
            vecs.append(n)
    vecs.sort(key=lambda v: (sum(x * x for x in v), v))
    F.G_INT = vecs
    F.NB = len(vecs)
    F.GI = np.array(vecs, dtype=int)
    F.GV = F.GI.astype(float) * F.U
    F.DN = F.GI[:, None, :] - F.GI[None, :, :]
    F.Q2 = np.sum(F.DN**2, axis=2)
    F.PHASE = (np.pi / 4.0) * np.sum(F.DN, axis=2)
    F.DG = F.GV[:, None, :] - F.GV[None, :, :]
    F.CROSS0 = np.cross(F.DG, F.GV[None, :, :])
    F.K_L = 0.5 * F.U * np.array([1.0, 1.0, 1.0])
    F.K_X = F.U * np.array([1.0, 0.0, 0.0])
    F.TARGETS = cfg["targets"]
    F.KEYS = list(cfg["targets"])


def params_vector(d):
    return np.array([d[n] for n in NAMES], dtype=float)


def residuals(p10):
    obs = F.observables(p10)
    r = []
    for k in F.KEYS:
        t, s = F.TARGETS[k]
        r.append((obs[k] - t) / (s * t) if k.startswith("m_")
                 else (obs[k] - t) / s)
    return np.array(r)


def run_fit(cfg):
    free = cfg["free"]
    idx = [NAMES.index(n) for n in free]
    base_d = dict(cfg["fixed"])
    base_d.update(cfg["start"])
    base = params_vector({n: base_d.get(n, 0.0) for n in NAMES})

    def res(y):
        p = base.copy(); p[idx] = y; return residuals(p)
    lo = np.array([cfg["bounds"][n][0] for n in free])
    hi = np.array([cfg["bounds"][n][1] for n in free])
    best = None
    rng = np.random.default_rng(7)
    starts = [base[idx]]
    for _ in range(3):
        starts.append(np.clip(base[idx] * (1 + rng.normal(0, 0.2, len(idx))),
                              lo + 1e-9, hi - 1e-9))
    for x0 in starts:
        s = least_squares(res, x0, bounds=(lo, hi), method="trf",
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        if best is None or s.cost < best.cost:
            best = s
    p = base.copy(); p[idx] = best.x
    return p, float(2 * best.cost)


def band_figure(cfg_id, p10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = F.build_V(p10[:8])
    E = []
    for i in range(21):
        n = np.array([0.5 - i / 20.0] * 3) if i <= 10 else np.array([(i - 10) / 10.0, 0, 0])
        E.append(np.sort(np.linalg.eigvalsh(F.hamiltonian(n * F.U, V, p10[8], p10[9]))))
    E = np.array(E)
    x = np.concatenate([np.linspace(-10, 0, 11), np.linspace(1, 10, 10)])
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=200)
    for b in range(min(16, E.shape[1])):
        ax.plot(x, E[:, b] - E[10][7], lw=0.7, color="#0f766e")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlim(-10, 10); ax.set_ylim(-6, 5)
    ax.set_xticks([-10, 0, 10]); ax.set_xticklabels(["L [111]", "Γ [000]", "X [100]"])
    ax.set_ylabel("Energy − VBM(Γ)  [eV]")
    ax.set_title(f"{cfg_id} — fitted band profile (pool entry)")
    ax.grid(alpha=0.25); fig.tight_layout()
    out = POOL / cfg_id / "bands.png"
    fig.savefig(out)
    return out


def register(cfg_id, cfg, p10, cost, mode):
    d = POOL / cfg_id
    d.mkdir(parents=True, exist_ok=True)
    obs = F.observables(p10)
    params = {n: float(v) for n, v in zip(NAMES, p10)}
    (d / "params.json").write_text(json.dumps(
        {"config": cfg_id, "a0_m": cfg["a0"], "basis_q2": cfg["basis_q2"],
         "params_Ry_and_scalars": params, "cost": cost, "mode": mode,
         "target_source": cfg["target_source"], "note": cfg.get("note", "")},
        indent=1))
    lines = [f"# Pool entry — {cfg_id}", "",
             f"- Mode: {mode} · cost = {cost:.3f}",
             f"- Targets: {cfg['target_source']}",
             f"- Basis: G² shells {cfg['basis_q2']} ({F.NB} PW), a0 = {cfg['a0']:.4e} m",
             "", "| Quantity | Fit | Target | Diff |", "|---|---|---|---|"]
    for k in F.KEYS:
        t, _ = F.TARGETS[k]
        lines.append(f"| {k} | {obs[k]:.4f} | {t:.3f} | {obs[k] - t:+.4f} |")
    lines += ["", "Parameters (Ry; L, beta dimensionless):",
              json.dumps(params, indent=1), "", cfg.get("note", "")]
    (d / "report.md").write_text("\n".join(lines))
    band_figure(cfg_id, p10)
    # registry
    idx_path = POOL / "INDEX.md"
    rows = {}
    if idx_path.exists():
        for line in idx_path.read_text().splitlines():
            if line.startswith("| ") and "---" not in line and "Config" not in line:
                rows[line.split("|")[1].strip()] = line
    rows[cfg_id] = (f"| {cfg_id} | {cost:.2f} | {mode} | "
                    f"{cfg['target_source']} |")
    body = ["# Parameter Pool — INDEX", "",
            "Every fitted/locked EPM parameter set lives here with its data.",
            "", "| Config | Cost | Mode | Target source |", "|---|---|---|---|"]
    body += [rows[k] for k in sorted(rows)]
    idx_path.write_text("\n".join(body) + "\n")
    return obs


def main():
    mode, cfg_id = sys.argv[1], sys.argv[2]
    cfg = CONFIGS[cfg_id]
    setup(cfg)
    if mode == "verify":
        p = params_vector(cfg["locked_params"])
        cost = float(residuals(p) @ residuals(p))
    elif mode == "fit":
        p, cost = run_fit(cfg)
    else:
        raise SystemExit("mode must be verify|fit")
    obs = register(cfg_id, cfg, p, cost, mode)
    print(f"{cfg_id} [{mode}] cost = {cost:.3f}")
    for k in F.KEYS:
        print(f"  {k:10s} {obs[k]:9.4f}  target {F.TARGETS[k][0]:8.3f}")
    print("registered in pool:", POOL / cfg_id)


if __name__ == "__main__":
    main()
