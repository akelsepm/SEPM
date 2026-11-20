"""
Parameter pool API — the dataset as a function: params(system, x, T).

Anchors are FITTED binary sets at the two reference temperatures
(77 K and 300 K), stored in params_pool/<Binary>_<T>K/params.json.
Everything else is interpolation of the PARAMETERS (never of finished
observables), so the result is always a usable Hamiltonian input:

  * Temperature: linear in T between the two anchors, per parameter.
    Inside [77, 300] -> flag "T-interpolated"; outside -> the same line
    extrapolated, flag "T-EXTRAPOLATED" (unvalidated by construction).
  * Composition: for a ternary A_x B_{1-x} C, virtual-crystal
    interpolation of the parameters, linear in x; the lattice constant
    follows Vegard's law (the same linear rule). A per-parameter
    bowing hook (BOWING below) carries the alloy layer's own fitted
    coefficients; AlGaAs/VA4 is active, everything else is zero.

Book rule (Kazim, 2026-11-19): the book publishes ONLY the anchors —
binaries at 77/300 K, compared against literature. Interpolated sets
are program-side and always carry their provenance flag.

Usage:
    from pool import get_params
    p = get_params("GaAs", T=300)            # anchor
    p = get_params("GaAs", T=200)            # T-interpolated
    p = get_params("AlGaAs", x=0.30, T=200)  # alloy + T interpolation
The returned dict matches bulk_properties.MATERIALS entries
(a0, VS_RY, VA_RY, L_SO, BETA) plus a "provenance" record.
"""
import json
from pathlib import Path

POOL_DIR = Path(__file__).parent / "params_pool"
ANCHOR_TEMPS = (77, 300)
CATIONS = ("Al", "Ga", "In")
ANIONS = ("As", "Sb", "P")
# Alloy-layer bowing coefficients, fitted per system against alloy
# data (never against the binaries; the term b * x * (1 - x) vanishes
# at both endpoints, so anchors are untouched by construction).
# AlGaAs/VA4: fitted 2026-11-19 to the VMR E_X(x) curve at 300 K on
# the frozen v3+v3 pair (code/fit_bowing.py; X rms 56 -> 12 meV,
# 77 K transfer 61 -> 19 meV; Gamma cost <= 7 meV at x = 0.3).
BOWING = {("AlGaAs", "VA4"): 0.00862}


def _load_anchor(binary, T):
    path = POOL_DIR / f"{binary}_{T}K" / "params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"anchor missing: {binary} at {T} K ({path}) — fit it first "
            f"(campaign order in files/Malzeme-Havuzu-Plani.md)")
    d = json.loads(path.read_text())
    p = d["params_Ry_and_scalars"]
    return {"a0": d["a0_m"],
            "VS_RY": {q: p[f"VS{q}"] for q in (3, 4, 8, 11)},
            "VA_RY": {q: p[f"VA{q}"] for q in (3, 4, 8, 11)},
            "L_SO": p["L"], "BETA": p["beta"]}


def _lerp(pa, pb, f):
    out = {"a0": pa["a0"] + f * (pb["a0"] - pa["a0"]),
           "VS_RY": {q: pa["VS_RY"][q] + f * (pb["VS_RY"][q] - pa["VS_RY"][q])
                     for q in pa["VS_RY"]},
           "VA_RY": {q: pa["VA_RY"][q] + f * (pb["VA_RY"][q] - pa["VA_RY"][q])
                     for q in pa["VA_RY"]},
           "L_SO": pa["L_SO"] + f * (pb["L_SO"] - pa["L_SO"]),
           "BETA": pa["BETA"] + f * (pb["BETA"] - pa["BETA"])}
    return out


def binary_params(binary, T):
    """Locked-scheme parameter set of a binary at temperature T."""
    lo, hi = ANCHOR_TEMPS
    if T == lo:
        p = _load_anchor(binary, lo); flag = "anchor-77K"
    elif T == hi:
        p = _load_anchor(binary, hi); flag = "anchor-300K"
    else:
        pa, pb = _load_anchor(binary, lo), _load_anchor(binary, hi)
        f = (T - lo) / (hi - lo)
        p = _lerp(pa, pb, f)
        flag = "T-interpolated" if lo < T < hi else "T-EXTRAPOLATED"
    p["provenance"] = {"system": binary, "x": None, "T": T, "flag": flag}
    return p


def _split_ternary(name):
    """'AlGaAs' -> ('Al', 'Ga', 'As'); the FIRST cation carries x."""
    for an in ANIONS:
        if name.endswith(an):
            stem = name[: -len(an)]
            for c1 in CATIONS:
                if stem.startswith(c1):
                    c2 = stem[len(c1):]
                    if c2 in CATIONS:
                        return c1, c2, an
    return None


def get_params(system, x=None, T=300):
    """Parameter set for a binary ('GaAs') or ternary ('AlGaAs', x =
    fraction of the FIRST cation). T in kelvin."""
    sp = _split_ternary(system)
    if sp is None:  # binary
        if x not in (None, 0.0, 1.0):
            raise ValueError("x given for a binary system")
        return binary_params(system, T)
    if x is None:
        raise ValueError(f"x is required for ternary {system}")
    c1, c2, an = sp
    pA = binary_params(f"{c1}{an}", T)
    pB = binary_params(f"{c2}{an}", T)
    p = _lerp(pB, pA, x)
    bowed = []
    for (sysname, key), b in BOWING.items():
        if sysname == system and b:
            section, shell = key[:2] + "_RY", int(key[2:])
            p[section][shell] += b * x * (1 - x)
            bowed.append(key)
    flags = {pA["provenance"]["flag"], pB["provenance"]["flag"]}
    p["provenance"] = {"system": system, "x": x, "T": T,
                       "flag": "VCA-x" + ("+" + "+".join(sorted(flags))
                                          if flags else "")
                       + ("+bowing(" + ",".join(bowed) + ")" if bowed else "")}
    return p


def missing_anchors():
    out = []
    for c in CATIONS:
        for an in ANIONS:
            b = f"{c}{an}"
            for T in ANCHOR_TEMPS:
                if not (POOL_DIR / f"{b}_{T}K" / "params.json").exists():
                    out.append(f"{b}_{T}K")
    return out


if __name__ == "__main__":
    p = get_params("GaAs", T=300)
    print("GaAs @300K (anchor):", p["VS_RY"], "L =", p["L_SO"],
          "|", p["provenance"]["flag"])
    print("Missing anchors (the campaign will fill these):")
    print(" ", ", ".join(missing_anchors()))
