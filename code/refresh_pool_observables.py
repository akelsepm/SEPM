"""
Refresh every pool entry's stored observables + cost under the
CURRENT measurement ruler (symmetric valley estimator in
fit_pool.observables10, 2026-11-19) and the same target transforms
the fits used (DOS-valley swap; narrow-gap profile for InAs/InSb;
AlAs satellite tightening). Parameters are NOT touched — this is a
measurement refresh only, so no delivered alloy number changes.
Backup of the pre-refresh pool: hidden_files/params_pool_pre_refresh_obs/.
"""
import json
import shutil
from pathlib import Path

import numpy as np

import fit_gaas_300k as F
import fit_pool
from fit_pool import DOS_VALLEY, MASS_KEYS, NAMES, TARGETS, observables10

HERE = Path(__file__).parent
POOL = HERE / "params_pool"
BAK = HERE.parent / "hidden_files" / "params_pool_pre_refresh_obs"
if not BAK.exists():
    shutil.copytree(POOL, BAK)
    print("backup:", BAK)


def fit_targets(mat, T):
    tg = dict(TARGETS[(mat, T)]["targets"])
    if mat == "AlAs":
        tg = {k: (v[0], 0.005 if k in ("Eg", "E_X") else
                  (0.10 if k == "ml_X" else
                   (0.03 if k == "mt_X" else v[1])))
              for k, v in tg.items()}
    if mat in ("InAs", "InSb"):
        tg = {k: (v[0], 0.005 if k == "Eg" else (v[1] * 2 if k in MASS_KEYS else v[1]))
              for k, v in tg.items()}
    if mat in DOS_VALLEY:
        for k in ("mt_L", "ml_Lmin", "mt_X", "ml_X"):
            tg.pop(k, None)
        tg.update(DOS_VALLEY[mat])
    return tg


import re

for entry in sorted(POOL.iterdir()):
    pj = entry / "params.json"
    if not pj.exists() or not re.fullmatch(
            r"(GaAs|InAs|GaSb|AlSb|InSb|AlAs|InP|GaP)_(77|300)K", entry.name):
        continue  # archive entries (e.g. GaAs_300K_classic) are skipped
    d = json.loads(pj.read_text())
    mat, T = entry.name.rsplit("_", 1)
    T = int(T[:-1])
    sp = d["params_Ry_and_scalars"]
    p10 = np.array([sp[k] for k in NAMES])
    fit_pool.setup_basis(d["a0_m"])
    o = observables10(p10, F.U, F.HBAR2_OVER_2M0)
    tg = fit_targets(mat, T)
    cost = 0.0
    for k, (t, s) in tg.items():
        sig = s * t if k in MASS_KEYS else s
        cost += ((o[k] - t) / sig) ** 2
    d["observables"] = {k: float(v) for k, v in o.items()}
    d["cost"] = float(cost)
    d["obs_ruler"] = "symmetric valley estimator (2026-11-19 refresh)"
    pj.write_text(json.dumps(d, indent=1))
    print(f"{entry.name}: cost {cost:6.1f}  ml_X "
          f"{o.get('ml_X', float('nan')):.2f}  ml_L {o.get('ml_Lmin', float('nan')):.2f}")
print("done: observables re-measured for all 16 anchors (parameters unchanged)")
