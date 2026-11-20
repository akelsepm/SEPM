"""Chapter 9 alloy-layer driver v2 (generalizes fit_bowing.py).

Per system: VCA baseline residuals against ternary_targets, a
form-factor sensitivity scan, then a Gamma-first bowing fit with at
most two coefficients and one alternation. Registration gate
(decided before seeing the results, Chapter 8 discipline):
  * Gamma rms at 300 K <= 25 meV after the fit, AND
  * the 300 K coefficient must TRANSFER: Gamma rms at 77 K no worse
    than the VCA value + 10 meV, AND
  * each fitted mid-x term |b|/4 stays <= 60% of the channel's mean
    endpoint form-factor magnitude (the term must not rival the
    potential it corrects).
Systems failing the gate are recorded as path-limited: their
would-be coefficients are documented but NOT registered in pool.py.
Writes hidden_files/bowing_<system>.json per system.

Usage: python3 ternary_fit.py InGaAs [more systems...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import fit_pool
import ternary_targets as TT
from bulk_properties import Bulk
from pool import POOL_DIR, get_params

HID = Path(__file__).parent.parent / "hidden_files"
XS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
CHANNELS = ["VS3", "VS4", "VS8", "VS11", "VA3", "VA4", "VA8", "VA11"]
KEY = {"G": "Eg_Gamma_eV", "X": "E_X_eV", "L": "E_L_eV", "Dso": "Delta_so_eV"}
BOUNDS = ([-0.15], [0.15])


def edges_with(system, x, T, bow):
    mat = get_params(system, x=float(x), T=T)
    m2 = {k: (dict(v) if isinstance(v, dict) else v)
          for k, v in mat.items() if k != "provenance"}
    for key, b in bow.items():
        m2[key[:2] + "_RY"][int(key[2:])] += b * x * (1 - x)
    fit_pool.setup_basis(m2["a0"])
    return Bulk(m2).edges()


def resid(system, T, bow):
    tgt = TT.SYSTEMS[system]
    out = {v: [] for v in "GXL"}
    out["Dso"] = []
    for x in XS:
        e = edges_with(system, x, T, bow)
        for v in "GXL":
            out[v].append(1000 * (e[KEY[v]] - tgt[v](x, T)))
        out["Dso"].append(1000 * (e["Delta_so_eV"] - tgt["Dso"](x, T)))
    return {v: np.array(a) for v, a in out.items()}


def rms(res):
    return {v: float(np.sqrt(np.mean(a ** 2))) for v, a in res.items()}


def scan(system, x=0.4, db=0.01):
    rows = {}
    for ch in CHANNELS:
        ep = edges_with(system, x, 300, {ch: db})
        em = edges_with(system, x, 300, {ch: -db})
        rows[ch] = {v: (ep[KEY[v]] - em[KEY[v]]) / (2 * db) for v in "GXL"}
    return rows


def pick_gamma_lever(rows):
    best, score = None, -1e9
    for ch, d in rows.items():
        s = abs(d["G"]) - 1.0 * max(abs(d["X"]), abs(d["L"]))
        if abs(d["G"]) >= 1.0 and s > score:
            best, score = ch, s
    return best


def pick_sat_lever(rows, valley):
    """Satellite lever: strong on the valley, weak on Gamma."""
    cands = [(abs(d[valley]), ch) for ch, d in rows.items()
             if abs(d[valley]) >= 2.0 and abs(d["G"]) <= 0.25 * abs(d[valley])]
    return max(cands)[1] if cands else None


def fit_on(system, valley, ch, bow0):
    f = lambda b: np.array([edges_with(system, x, 300, {**bow0, ch: b[0]})[KEY[valley]]
                            - TT.SYSTEMS[system][valley](x, 300) for x in XS])
    r = least_squares(f, [bow0.get(ch, 0.0)], bounds=BOUNDS)
    return float(r.x[0])


def term_ratio(system, ch, b):
    """|b|/4 relative to the channel's mean endpoint |FF| (300 K)."""
    from pool import _split_ternary
    binA, binB = _split_ternary(system)
    vals = []
    for bn in (binA, binB):
        d = json.loads((POOL_DIR / f"{bn}_300K" / "params.json").read_text())
        vals.append(abs(d["params_Ry_and_scalars"][ch]))
    return (abs(b) / 4) / (np.mean(vals) + 1e-12)


def crossover(system, bow, T=300):
    xs = np.linspace(0, 1, 101)
    prev, xc = None, None
    for x in xs:
        e = edges_with(system, x, T, bow)
        fund = min(e["Eg_Gamma_eV"], e["E_X_eV"], e["E_L_eV"])
        cur = "G" if fund == e["Eg_Gamma_eV"] else ("X" if fund == e["E_X_eV"] else "L")
        if prev == "G" and cur != "G":
            xc = (float(x), cur)
            break
        prev = cur
    return xc


def run(system):
    print(f"=== {system} ===", flush=True)
    base300 = rms(resid(system, 300, {}))
    base77 = rms(resid(system, 77, {}))
    print("  VCA rms300:", {k: round(v, 1) for k, v in base300.items()},
          "| rms77:", {k: round(v, 1) for k, v in base77.items()}, flush=True)
    rows = scan(system)
    bow, steps = {}, []
    r_now = dict(base300)
    g_ch = None
    if r_now["G"] >= 15.0:
        g_ch = pick_gamma_lever(rows)
        if g_ch:
            bow[g_ch] = fit_on(system, "G", g_ch, bow)
            steps.append(f"Gamma lever {g_ch}: b={bow[g_ch]:+.5f} Ry")
            r_now = rms(resid(system, 300, bow))
            print(f"  {steps[-1]} -> rmsG {r_now['G']:.1f}", flush=True)
    sat = max(("X", "L"), key=lambda v: r_now[v])
    s_ch = None
    if r_now[sat] >= 25.0:
        s_ch = pick_sat_lever(rows, sat)
        if s_ch and s_ch not in bow:
            bow[s_ch] = fit_on(system, sat, s_ch, bow)
            steps.append(f"{sat} lever {s_ch}: b={bow[s_ch]:+.5f} Ry")
            r2 = rms(resid(system, 300, bow))
            print(f"  {steps[-1]} -> rms{sat} {r2[sat]:.1f}, rmsG {r2['G']:.1f}", flush=True)
            if r2["G"] > r_now["G"] + 5.0 and g_ch:  # one alternation
                bow[g_ch] = fit_on(system, "G", g_ch, bow)
                steps.append(f"Gamma refit {g_ch}: b={bow[g_ch]:+.5f} Ry")
                r2 = rms(resid(system, 300, bow))
                print(f"  {steps[-1]} -> rmsG {r2['G']:.1f}", flush=True)
            r_now = r2
    fin300 = r_now
    fin77 = rms(resid(system, 77, bow))
    ratios = {ch: term_ratio(system, ch, b) for ch, b in bow.items()}
    gate = bool(bow) and fin300["G"] <= 25.0 \
        and fin77["G"] <= base77["G"] + 10.0 \
        and all(r <= 0.60 for r in ratios.values())
    xc_t = TT.SYSTEMS[system]["crossover"]
    xc_b = crossover(system, {}) if xc_t else None
    xc_f = crossover(system, bow) if xc_t else None
    out = {"system": system, "bowing": bow, "steps": steps,
           "term_ratios": ratios, "gate_pass": gate,
           "rms300_vca": base300, "rms300_final": fin300,
           "rms77_vca": base77, "rms77_final": fin77,
           "scan_x0.4": rows,
           "crossover_target": xc_t, "crossover_vca": xc_b,
           "crossover_final": xc_f,
           "note": TT.NOTES[system]}
    (HID / f"bowing_{system}.json").write_text(json.dumps(out, indent=1))
    print(f"  FINAL bow={ {k: round(v,5) for k,v in bow.items()} } gate={gate} "
          f"rms300={ {k: round(v,1) for k,v in fin300.items()} } "
          f"rms77={ {k: round(v,1) for k,v in fin77.items()} } xc: {xc_b} -> {xc_f} (target {xc_t})",
          flush=True)
    return out


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(s)
    print("ALL DONE", flush=True)
