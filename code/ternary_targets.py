"""Literature targets for the Chapter 9 ternary wave (8 systems).

Sources: Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815
(2001) recommended values (full text) and the Ioffe NSM archive,
compiled 2026-11-19/20 in
~/workspace/research_notes/ternary-wave1-literature-targets-20261120-0127/report.md
(read that report for per-value provenance and caveats).

CONVENTION (ours, = pool.py): x is the fraction of the FIRST-named
constituent: InGaAs = In_x Ga_1-x As, GaAsP = Ga As_x P_1-x, etc.
Every curve below is already transcribed into this convention
(several sources use the reversed variable; the transcriptions were
endpoint-checked against both binaries).

Construction: for each valley, E(x, T) = (1-x) E_B(T) + x E_A(T)
- C x (1-x), with binary endpoints E(T) from Vurgaftman's 0 K values
and his Varshni parameters (the report's 300 K polynomials are this
same construction evaluated at 300 K). Delta_so and m_e curves are
the paper's (temperature-independent) recommended polynomials.
EXCEPTIONS, flagged per system in NOTES:
  * InGaSb / InAsSb X and L: Vurgaftman tabulates no InSb X/L
    Varshni parameters, so no 300 K form exists in the paper. These
    targets are CONSTRUCTED: our Chapter-7-verified binary endpoints
    (Ioffe-based) at each T + the paper's bowing C.
  * AlGaSb Gamma: the paper's bowing is composition-dependent,
    C(x) = -0.044 + 1.22 x (x = Al fraction = our x).
  * InGaP: C(Gamma) = 0.65 eV is the DISORDERED alloy; CuPt-ordered
    samples sit ~100 meV lower and are not a target.
"""
import numpy as np

# Binary Varshni data from Vurgaftman Tables I-IX: valley ->
# (E0 [eV], alpha [meV/K], beta [K]); "coth"/"linear" are special.
VARSHNI = {
    "GaAs": {"G": (1.519, 0.5405, 204), "X": (1.981, 0.460, 204),
             "L": (1.815, 0.605, 204)},
    "AlAs": {"G": (3.099, 0.885, 530), "X": (2.240, 0.700, 530),
             "L": (2.460, 0.605, 204)},
    "InAs": {"G": (0.417, 0.276, 93), "X": (1.433, 0.276, 93),
             "L": (1.133, 0.276, 93)},
    "GaP": {"G": "coth", "X": (2.350, 0.5771, 372),
            "L": (2.720, 0.5771, 372)},
    "InP": {"G": (1.4236, 0.363, 162), "X": "linear",
            "L": (2.014, 0.363, 162)},
    "GaSb": {"G": (0.812, 0.417, 140), "X": (1.141, 0.475, 94),
             "L": (0.875, 0.597, 140)},
    "AlSb": {"G": (2.386, 0.420, 140), "X": (1.696, 0.390, 140),
             "L": (2.329, 0.580, 140)},
    "InSb": {"G": (0.235, 0.320, 170), "X": None, "L": None},
}


def _e_bin(binary, valley, T):
    v = VARSHNI[binary][valley]
    if v == "coth":      # GaP Gamma, Vurgaftman's form
        return 2.886 + 0.1081 * (1 - 1 / np.tanh(164.0 / T))
    if v == "linear":    # InP X: E = 2.384 - 3.7e-4 T
        return 2.384 - 3.7e-4 * T
    E0, a, b = v
    return E0 - a * T * T / (T + b) / 1000.0


def _curve(binA, binB, valley, C):
    """Alloy curve E(x, T); x = fraction of binA. C may be a
    callable C(x) (AlGaSb Gamma)."""
    def f(x, T=300):
        e = (1 - x) * _e_bin(binB, valley, T) + x * _e_bin(binA, valley, T)
        c = C(x) if callable(C) else C
        return e - c * x * (1 - x)
    return f


def _constructed(binA, binB, endsA, endsB, C):
    """X/L target from verified endpoints {T: E} + constant bowing."""
    def f(x, T=300):
        return (1 - x) * endsB[T] + x * endsA[T] - C * x * (1 - x)
    return f


def _poly(*coefs):
    """Polynomial in our x: sum coefs[i] * x**i (300 K; also used at
    77 K for Delta_so / m_e, which carry no T model)."""
    def f(x, T=300):
        return sum(c * x ** i for i, c in enumerate(coefs))
    return f


def _lin(e_at_0, e_at_1):
    def f(x, T=300):
        return (1 - x) * e_at_0 + x * e_at_1
    return f


# Verified binary endpoints (Chapter 7, Ioffe-based) for the two
# constructed-target systems. InAs endpoints from Vurgaftman-Varshni
# (consistent with Ioffe at 300 K: X 1.370, L 1.070).
_VER = {  # (binary, valley) -> {T: E}
    ("GaSb", "X"): {300: 1.030, 77: 1.126}, ("GaSb", "L"): {300: 0.810, 77: 0.888},
    ("InSb", "X"): {300: 1.000, 77: 1.000}, ("InSb", "L"): {300: 0.680, 77: 0.740},
    ("InAs", "X"): {300: 1.370, 77: 1.423}, ("InAs", "L"): {300: 1.070, 77: 1.123},
}

SYSTEMS = {
    "InGaAs": {  # x = In
        "G": _curve("InAs", "GaAs", "G", 0.477),
        "X": _curve("InAs", "GaAs", "X", 1.40),
        "L": _curve("InAs", "GaAs", "L", 0.33),
        "Dso": _poly(0.341, -0.101, 0.15),
        "me": _poly(0.067, -0.0501, 0.0091),
        "a": _lin(5.65325, 6.0583),
        "crossover": None,
    },
    "AlInAs": {  # x = Al
        "G": _curve("AlAs", "InAs", "G", 0.70),
        "X": _curve("AlAs", "InAs", "X", 0.0),
        "L": _curve("AlAs", "InAs", "L", 0.0),
        "Dso": _poly(0.390, -0.260, 0.15),
        "me": _poly(0.026, 0.075, 0.049),
        "a": _lin(6.0583, 5.6611),
        "crossover": ("X", 0.64),
    },
    "InGaP": {  # x = In; DISORDERED alloy targets
        "G": _curve("InP", "GaP", "G", 0.65),
        "X": _curve("InP", "GaP", "X", 0.20),
        "L": _curve("InP", "GaP", "L", 1.03),
        "Dso": _poly(0.080, 0.028),
        "me": _poly(0.130, -0.1015, 0.051),
        "a": _lin(5.4505, 5.8697),
        "crossover": ("X", 0.30),   # Gamma-X at x_Ga 0.70; Gamma-L at x_In 0.33
    },
    "GaAsP": {  # x = As
        "G": _curve("GaAs", "GaP", "G", 0.19),
        "X": _curve("GaAs", "GaP", "X", 0.24),
        "L": _curve("GaAs", "GaP", "L", 0.16),
        "Dso": _poly(0.080, 0.261),
        "me": _lin(0.130, 0.067),   # endpoints only; no bowing recommended
        "a": _lin(5.4505, 5.65325),
        "crossover": ("X", 0.55),   # indirect for P fraction > 0.45 (0 K)
    },
    "AlGaSb": {  # x = Al
        "G": _curve("AlSb", "GaSb", "G", lambda x: -0.044 + 1.22 * x),
        "X": _curve("AlSb", "GaSb", "X", 0.0),
        "L": _curve("AlSb", "GaSb", "L", 0.0),
        "Dso": _poly(0.760, -0.384, 0.30),
        "me": _lin(0.039, 0.140),   # endpoints only; no bowing tabulated
        "a": _lin(6.0959, 6.1355),
        "crossover": ("X", 0.43),   # recommended-set value; exp. Gamma-L claims 0.23-0.27
    },
    "InGaSb": {  # x = In
        "G": _curve("InSb", "GaSb", "G", 0.415),
        "X": _constructed("InSb", "GaSb", _VER[("InSb", "X")], _VER[("GaSb", "X")], 0.33),
        "L": _constructed("InSb", "GaSb", _VER[("InSb", "L")], _VER[("GaSb", "L")], 0.40),
        "Dso": _poly(0.760, -0.050, 0.10),
        "me": _poly(0.039, -0.0347, 0.0092),
        "a": _lin(6.0959, 6.4794),
        "crossover": None,
    },
    "GaAsSb": {  # x = As
        "G": _curve("GaAs", "GaSb", "G", 1.43),
        "X": _curve("GaAs", "GaSb", "X", 1.20),
        "L": _curve("GaAs", "GaSb", "L", 1.20),
        "Dso": _poly(0.760, -1.019, 0.60),
        "me": _poly(0.0393, -0.0021, 0.0258),   # NSM mass formula
        "a": _lin(6.0959, 5.65325),
        "crossover": None,   # direct throughout; miscibility gap ~0.2-0.8 Sb (suppl.)
    },
    "InAsSb": {  # x = As
        "G": _curve("InAs", "InSb", "G", 0.67),
        "X": _constructed("InAs", "InSb", _VER[("InAs", "X")], _VER[("InSb", "X")], 0.60),
        "L": _constructed("InAs", "InSb", _VER[("InAs", "L")], _VER[("InSb", "L")], 0.60),
        "Dso": _poly(0.810, -1.620, 1.20),
        "me": _poly(0.0135, -0.0225, 0.035),
        "a": _lin(6.4794, 6.0583),
        "crossover": None,   # direct throughout; gap minimum near x_As 0.37
    },
}

NOTES = {
    "InGaAs": "VMR full set; NSM formulas (reversed variable) agree on Gamma bowing (~0.43-0.48).",
    "AlInAs": "VMR only (no NSM section). E_L not tabulated: linear (C=0). Crossover Gamma-X 0.64 (exp. 0.68).",
    "InGaP": "DISORDERED alloy. CuPt ordering lowers the gap ~100 meV; ordered samples are not a target. C(E_L)=1.03 as extracted (range column garbled in the source PDF).",
    "GaAsP": "VMR only (no NSM section). m_e: no bowing recommended (endpoints only).",
    "AlGaSb": "Gamma bowing composition-dependent. Recommended set crosses Gamma-X at ~0.43; experimental Gamma-L claims 0.23/0.27 are NOT reproduced by the set.",
    "InGaSb": "X/L targets CONSTRUCTED (verified endpoints + VMR bowing); no NSM gap formulas.",
    "GaAsSb": "VMR Gamma bowing 1.43 vs NSM 1.2 (NSM formula valid only Sb<0.3). Miscibility gap ~0.2-0.8 Sb at 600 C (supplementary sources).",
    "InAsSb": "X/L targets CONSTRUCTED (verified endpoints + VMR bowing). Miscibility/phase separation reported 0.2<x_Sb<0.6 (supplementary).",
}

if __name__ == "__main__":
    # Endpoint sanity: every curve must return its binary endpoints.
    ends = {"InGaAs": ("GaAs", "InAs"), "AlInAs": ("InAs", "AlAs"),
            "InGaP": ("GaP", "InP"), "GaAsP": ("GaP", "GaAs"),
            "AlGaSb": ("GaSb", "AlSb"), "InGaSb": ("GaSb", "InSb"),
            "GaAsSb": ("GaSb", "GaAs"), "InAsSb": ("InSb", "InAs")}
    for s, spec in SYSTEMS.items():
        b0, b1 = ends[s]
        for v in "GXL":
            if (b0, v) in _VER or (b1, v) in _VER:
                continue
            for x, b in ((0.0, b0), (1.0, b1)):
                got = spec[v](x, 300)
                want = _e_bin(b, v, 300)
                assert abs(got - want) < 1e-9, (s, v, x, got, want)
        print(s, "endpoints OK; Eg(0.5,300) =", round(spec["G"](0.5, 300), 4))
    print("InGaSb X(0.5,300) constructed:", round(SYSTEMS["InGaSb"]["X"](0.5, 300), 4),
          "| InAsSb G min ~", round(min(SYSTEMS["InAsSb"]["G"](x / 50, 300) for x in range(51)), 4))
