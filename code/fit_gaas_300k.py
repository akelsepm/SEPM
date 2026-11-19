"""
EPM GaAs 300 K fit package (Layer 1.5)
======================================
Goal (Kazim's design, 2026-11-18): find the form-factor pool + L (+ beta)
that reproduces the 300 K literature values, using the same strategy he
used originally - gaps + spin-orbit splitting + PARABOLIC effective-mass
fits inside an optimisation loop.

Free parameters : VS(3,4,8,11), VA(3,4,8,11) [Ry], L_SO, beta (m1 = m0/beta)
Targets (Vurgaftman et al., J. Appl. Phys. 89, 5815 (2001), GaAs @ 300 K):
  Eg(Gamma) = 1.424 eV   Delta_so = 0.341 eV   m_e = 0.067 m0
  E_L = 1.708 eV and E_X = 1.900 eV are the L / X CONDUCTION MINIMA
  measured from the GAMMA valence-band maximum (Vurgaftman convention) -
  NOT the local gaps at L / X. (Corrected 2026-11-18 after the first run
  fitted local gaps; the notebook's good-looking local L gap turned out
  to be two ~0.78 eV edge errors cancelling.)
  Luttinger gamma1=6.98, gamma2=2.06, gamma3=2.93  ->
  m_hh[100]=0.350, m_lh[100]=0.090, m_hh[111]=0.893, m_lh[111]=0.078

The Hamiltonian builder is the verified mirror (max dev. 1.5e-13 eV),
re-implemented vectorised and parameterised. The mirror file stays frozen.
"""

import json
import pathlib
import time

import numpy as np
from scipy.optimize import differential_evolution, least_squares

# ---------------- fixed structure (same as mirror) ----------------
A0 = 5.653e-10
RY_TO_EV = 13.6059
G_INT = [
    (0, 0, 0),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2),
]
HBAR_SI = 1.0545717e-34
HBAR_EVS = 6.58211928e-16
M0 = 9.10938291e-31
C_LIGHT = 299792458.0
EV_J = 1.602176634e-19
HBAR2_OVER_2M0 = HBAR_SI**2 / (2 * M0) / EV_J          # eV m^2  (~3.81e-20)

NB = len(G_INT)
GI = np.array(G_INT, dtype=int)
GV = GI.astype(float) * (2 * np.pi / A0)
DN = GI[:, None, :] - GI[None, :, :]                    # (15,15,3) integer
Q2 = np.sum(DN**2, axis=2)                              # (15,15)
PHASE = (np.pi / 4.0) * np.sum(DN, axis=2)              # G.tau phase
DG = GV[:, None, :] - GV[None, :, :]                    # G_i - G_j
CROSS0 = np.cross(DG, GV[None, :, :])                   # (Gi-Gj) x Gj

SIGMA = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]], complex),
         np.array([[1, 0], [0, -1]], complex)]
SPIN = [np.array([1, 0], complex), np.array([0, 1], complex)]
SIGMA_SS = np.zeros((2, 2, 3), complex)
for s in range(2):
    for sp in range(2):
        for a in range(3):
            SIGMA_SS[s, sp, a] = SPIN[s].conj() @ SIGMA[a] @ SPIN[sp]

QS = [3, 4, 8, 11]
QIDX = {q: n for n, q in enumerate(QS)}


def build_V(p):
    """p = [VS3,VS4,VS8,VS11, VA3,VA4,VA8,VA11] in Ry -> V matrix (eV)."""
    vs = np.zeros(Q2.shape)
    va = np.zeros(Q2.shape)
    for q, n in QIDX.items():
        m = Q2 == q
        vs[m] = p[n] * RY_TO_EV
        va[m] = p[4 + n] * RY_TO_EV
    return vs * np.cos(PHASE) - 1j * va * np.sin(PHASE)


def hamiltonian(k, V, L_so, beta):
    m1 = M0 / beta
    coeff2 = HBAR_SI * HBAR_EVS / (2 * m1)
    coeff1 = 1j * HBAR_SI**2 * L_so / (4 * m1**2 * C_LIGHT**2)
    kG = GV + k
    kin = coeff2 * np.sum(kG**2, axis=1)
    cross = CROSS0 + np.cross(DG, k)                    # (Gi-Gj) x (k+Gj)
    so = coeff1 * V[:, :, None, None] * np.einsum("ija,tsa->ijts", cross, SIGMA_SS)
    A = V.copy()
    A[np.diag_indices(NB)] += kin
    H = np.zeros((NB, 2, NB, 2), complex)
    H[:, 0, :, 0] = A + so[:, :, 0, 0]
    H[:, 0, :, 1] = so[:, :, 0, 1]
    H[:, 1, :, 0] = so[:, :, 1, 0]
    H[:, 1, :, 1] = A + so[:, :, 1, 1]
    return H.transpose(0, 1, 2, 3).reshape(2 * NB, 2 * NB)


def eigs(k, V, L_so, beta, vectors=False):
    H = hamiltonian(np.asarray(k, float), V, L_so, beta)
    if vectors:
        return np.linalg.eigh(H)
    return np.linalg.eigvalsh(H)


# ---------------- observables ----------------
U = 2 * np.pi / A0
K_GAMMA = np.zeros(3)
K_L = 0.5 * U * np.array([1.0, 1.0, 1.0])
K_X = U * np.array([1.0, 0.0, 0.0])
DELTAS = np.array([0.01, 0.02, 0.03, 0.04])


def parabolic_masses(V, L_so, beta, direction):
    """Parabolic masses near Gamma along `direction`.

    CBM: single-state overlap tracking. Holes: the Gamma8 quartet is
    degenerate at Gamma, so individual states are arbitrary mixtures -
    instead the whole quartet SUBSPACE is tracked (projector weight),
    the 4 branch energies are split into their two Kramers pairs, and
    each pair is fitted. Returns (m_e, m_light, m_heavy) in m0."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    w0, v0 = eigs(K_GAMMA, V, L_so, beta, vectors=True)
    quartet = v0[:, 4:8]                                # Gamma8 subspace
    cbm_prev = v0[:, 8]
    e_cb = [w0[8]]
    pair_hi, pair_lo = [w0[7]], [w0[7]]                 # both start at VBM
    for dl in DELTAS:
        w, v = eigs(dl * U * d, V, L_so, beta, vectors=True)
        j = int(np.argmax(np.abs(v.conj().T @ cbm_prev)**2))
        e_cb.append(w[j])
        cbm_prev = v[:, j]
        weight = np.sum(np.abs(quartet.conj().T @ v)**2, axis=0)
        idx = np.argsort(weight)[-4:]                   # Gamma8-derived states
        es = np.sort(w[idx])                            # two Kramers pairs
        pair_lo.append(0.5 * (es[0] + es[1]))
        pair_hi.append(0.5 * (es[2] + es[3]))
    kk = np.concatenate([[0.0], DELTAS]) * U

    def mass(energies):
        a = np.polyfit(kk**2, np.array(energies), 1)[0]
        return abs(HBAR2_OVER_2M0 / a)
    mh = sorted([mass(pair_lo), mass(pair_hi)])
    return mass(e_cb), mh[0], mh[1]


def observables(p10):
    """p10 = [VS..., VA..., L, beta] -> dict of fitted quantities."""
    V = build_V(p10[:8])
    L_so, beta = p10[8], p10[9]
    eg = eigs(K_GAMMA, V, L_so, beta)
    el = eigs(K_L, V, L_so, beta)
    ex = eigs(K_X, V, L_so, beta)
    me100, ml100, mh100 = parabolic_masses(V, L_so, beta, [1, 0, 0])
    me111, ml111, mh111 = parabolic_masses(V, L_so, beta, [1, 1, 1])
    return {
        # E_L / E_X: conduction minima relative to VBM(Gamma) - see header.
        "Eg_Gamma": eg[8] - eg[7], "Eg_L": el[8] - eg[7], "Eg_X": ex[8] - eg[7],
        "Delta_so": eg[7] - eg[3],
        "m_e_100": me100, "m_lh_100": ml100, "m_hh_100": mh100,
        "m_e_111": me111, "m_lh_111": ml111, "m_hh_111": mh111,
    }


TARGETS = {
    "Eg_Gamma": (1.424, 0.010), "Eg_L": (1.708, 0.010), "Eg_X": (1.900, 0.010),
    "Delta_so": (0.341, 0.005),
    "m_e_100": (0.067, 0.03), "m_lh_100": (0.090, 0.03), "m_hh_100": (0.350, 0.03),
    "m_e_111": (0.067, 0.03), "m_lh_111": (0.078, 0.03), "m_hh_111": (0.893, 0.03),
}
KEYS = list(TARGETS)


def residuals(p10):
    obs = observables(p10)
    r = []
    for k in KEYS:
        t, s = TARGETS[k]
        if k.startswith("m_"):
            r.append((obs[k] - t) / (s * t))            # relative, sigma = 3%
        else:
            r.append((obs[k] - t) / s)                  # absolute, sigma in eV
    return np.array(r)


def cost(p10):
    r = residuals(p10)
    return float(r @ r)


# ---------------- starts ----------------
NOTEBOOK = np.array([-0.248, 0.0, 0.0, 0.0907, 0.090, 0.0, 0.0, 0.014, 3900.0, 1 / 0.9])
CHELIKOWSKY = np.array([-0.24845, 0.0, 0.03309, 0.09066,
                        0.07016, 0.0, 0.0, 0.01429, 3900.0, 1 / 0.9])
BOUNDS_LO = np.array([-0.45, -0.10, -0.10, 0.00, 0.00, -0.05, -0.05, 0.00, 500.0, 0.95])
BOUNDS_HI = np.array([-0.05, 0.10, 0.15, 0.20, 0.20, 0.05, 0.05, 0.06, 9000.0, 1.30])


def run():
    t0 = time.time()
    print("cost(notebook set)   =", round(cost(NOTEBOOK), 2))
    print("cost(CC start)       =", round(cost(CHELIKOWSKY), 2))
    results = []
    rng = np.random.default_rng(20261118)
    starts = [NOTEBOOK, CHELIKOWSKY]
    for s in range(4):
        jitter = NOTEBOOK * (1 + rng.normal(0, 0.15, 10))
        jitter[8] = NOTEBOOK[8] * (1 + rng.normal(0, 0.3))
        jitter[9] = NOTEBOOK[9] * (1 + rng.normal(0, 0.05))
        starts.append(np.clip(jitter, BOUNDS_LO + 1e-6, BOUNDS_HI - 1e-6))
    for i, x0 in enumerate(starts):
        sol = least_squares(residuals, x0, bounds=(BOUNDS_LO, BOUNDS_HI),
                            method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=400)
        results.append(sol)
        print(f"LS start {i}: cost = {sol.cost * 2:.3f}  ({time.time() - t0:.0f}s)")
    best = min(results, key=lambda s: s.cost)
    print("--- differential evolution cross-check (seeded at best LS) ---")
    de = differential_evolution(cost, list(zip(BOUNDS_LO, BOUNDS_HI)),
                                seed=42, maxiter=120, popsize=12, tol=1e-9,
                                init="sobol", polish=True,
                                x0=best.x)
    print(f"DE cost = {de.fun:.3f}")
    cand = de.x if de.fun < 2 * best.cost else best.x
    print("\n=== BEST SET ===")
    names = ["VS3", "VS4", "VS8", "VS11", "VA3", "VA4", "VA8", "VA11", "L", "beta"]
    for n, v, nb_, cc in zip(names, cand, NOTEBOOK, CHELIKOWSKY):
        print(f"  {n:5s} = {v:12.5f}    (notebook {nb_:10.4f} | CC {cc:10.4f})")
    obs = observables(cand)
    print("\n=== OBSERVABLES vs TARGETS (300 K) ===")
    for k in KEYS:
        t, _ = TARGETS[k]
        print(f"  {k:10s} fit = {obs[k]:9.4f}   target = {t:8.3f}   diff = {obs[k] - t:+.4f}")
    print("\nfinal cost =", round(cost(cand), 3), f"  total {time.time() - t0:.0f}s")
    out = {"params": dict(zip(names, [float(x) for x in cand])),
           "observables": {k: float(v) for k, v in obs.items()},
           "targets": {k: TARGETS[k][0] for k in KEYS},
           "cost": cost(cand)}
    p = pathlib.Path(__file__).resolve().parent.parent / "hidden_files" / "fit_gaas_300k_result.json"
    p.write_text(json.dumps(out, indent=1))
    print("saved:", p)


if __name__ == "__main__":
    run()
