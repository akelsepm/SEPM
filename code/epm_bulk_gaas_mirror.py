"""
EPM bulk GaAs - Python mirror v0 (faithful translation, Layer 1)
================================================================
Source: Mathematica notebook
  Local_EPM_GaAs_at_300K_with_Spin_Orbit_04.03.2014.nb  (Mathematica 7.0)

Rules of the mirror (agreed 2026-11-18):
  * Same conventions, same basis, same parameters as the notebook.
    No "improvements" in v0 - deviations are measured, not hidden.
  * ALL physical parameters live in the single block below.
  * The Mathematica original stays the reference.

Conventions mirrored from the notebook:
  * G vectors are handled as INTEGER triples n (units of 2*pi/a0);
    |G_i - G_j|^2 is then an exact integer -> the form-factor switch
    is exact, as in the notebook.
  * V(dn) = VS(|dn|^2) * cos(dn . tau_red) - i * VA(|dn|^2) * sin(dn . tau_red)
    with tau_red = (pi/4) * (1,1,1)   [because tau = (a0/8)(1,1,1) and
    G = (2*pi/a0) n  =>  G . tau = (pi/4) * sum(n)].
  * Kinetic coefficient  coeff2 = hbar_SI * hbar_eVs / (2 * m1)   [eV * m^2]
    (the notebook's mixed-unit trick: one hbar in J*s, one in eV*s,
     |k+G|^2 in m^-2 -> energy directly in eV.)
  * Spin-orbit coefficient coeff1 = i * hbar_SI^2 * L / (4 * m1^2 * c^2)
    Hso(s,s') = coeff1 * ( <s|sigma|s'> . [(G_i - G_j) x (k + G_j)] )
                * V(G_i - G_j)
  * Hamiltonian assembly order = ArrayFlatten of 15x15 blocks of 2x2:
    basis index = i*2 + s   (i = plane wave, s = spin).
  * k path: p = 0..20;  p <= 10: k = (2pi/a0)*(0.5 - p/20)*(1,1,1)  [L->Gamma]
                        p >  10: k = (2pi/a0)*((p-10)/10, 0, 0)     [Gamma->X]
"""

import json
import pathlib

import numpy as np

# =====================================================================
# PARAMETER BLOCK  (the single intervention point - mirror of notebook)
# =====================================================================
A0 = 5.653e-10              # lattice constant [m]  (GaAs, ~300 K)
RY_TO_EV = 13.6059          # notebook's conversion factor

# Symmetric / antisymmetric form factors [Ry], keyed by |dn|^2
VS_RY = {3: -0.248, 4: 0.0, 8: 0.0, 11: 0.0907}
VA_RY = {3:  0.090, 4: 0.0, 8: 0.0, 11: 0.014}

# Plane-wave basis: integer triples n (units of 2*pi/a0), notebook order
G_INT = [
    (0, 0, 0),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2),
]

# Physical constants, exactly as typed in the notebook
HBAR_SI = 1.0545717e-34     # J s
HBAR_EVS = 6.58211928e-16   # eV s
M0 = 9.10938291e-31         # kg
M1 = 0.9 * M0               # notebook's scaled mass (Dente-Tilton lineage, beta = M0/M1)
C_LIGHT = 299792458.0       # m/s
L_SO = 3900.0               # empirical spin-orbit enhancement (fitted, per Kazim)

# =====================================================================
# Derived quantities
# =====================================================================
TWOPi_A = 2.0 * np.pi / A0
G_VEC = np.array(G_INT, dtype=float) * TWOPi_A          # (15,3) in m^-1
G_INT_ARR = np.array(G_INT, dtype=int)
TAU_RED = np.pi / 4.0                                    # G.tau = TAU_RED * sum(dn)

COEFF2 = HBAR_SI * HBAR_EVS / (2.0 * M1)                 # eV m^2
COEFF1 = 1j * HBAR_SI**2 * L_SO / (4.0 * M1**2 * C_LIGHT**2)

SIGMA = [
    np.array([[0, 1], [1, 0]], dtype=complex),           # sigma_x
    np.array([[0, -1j], [1j, 0]], dtype=complex),        # sigma_y
    np.array([[1, 0], [0, -1]], dtype=complex),          # sigma_z
]
SPIN = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
# SIGMA_SS[s, s'] = (<s|sigma_x|s'>, <s|sigma_y|s'>, <s|sigma_z|s'>)
SIGMA_SS = np.zeros((2, 2, 3), dtype=complex)
for s in range(2):
    for sp in range(2):
        for a in range(3):
            SIGMA_SS[s, sp, a] = SPIN[s].conj() @ SIGMA[a] @ SPIN[sp]


def form_factor(dn):
    """V(G_i - G_j) in eV for integer difference triple dn."""
    q2 = int(dn @ dn)
    vs = VS_RY.get(q2, 0.0) * RY_TO_EV
    va = VA_RY.get(q2, 0.0) * RY_TO_EV
    phase = TAU_RED * int(np.sum(dn))
    return vs * np.cos(phase) - 1j * va * np.sin(phase)


# Precompute V matrix (15x15) and difference vectors
NB = len(G_INT)
V_MAT = np.zeros((NB, NB), dtype=complex)
DG = np.zeros((NB, NB, 3))                                # G_i - G_j [m^-1]
for i in range(NB):
    for j in range(NB):
        dn = G_INT_ARR[i] - G_INT_ARR[j]
        V_MAT[i, j] = form_factor(dn)
        DG[i, j] = G_VEC[i] - G_VEC[j]


def hamiltonian(k):
    """30x30 spinor Hamiltonian at k (m^-1), notebook conventions."""
    kG = G_VEC + np.asarray(k, dtype=float)              # k + G_j rows
    kin = COEFF2 * np.sum(kG**2, axis=1)                 # uses G_i row; |k+G_i|^2
    H = np.zeros((2 * NB, 2 * NB), dtype=complex)
    for i in range(NB):
        for j in range(NB):
            # spin-orbit: coeff1 * (sigma_ss . [(G_i-G_j) x (k+G_j)]) * V_ij
            cross = np.cross(DG[i, j], kG[j])
            so = COEFF1 * V_MAT[i, j] * (SIGMA_SS * cross).sum(axis=2)  # (2,2)
            block = so.copy()
            block[0, 0] += kin[i] * (1.0 if i == j else 0.0) + V_MAT[i, j]
            block[1, 1] += kin[i] * (1.0 if i == j else 0.0) + V_MAT[i, j]
            H[2 * i:2 * i + 2, 2 * j:2 * j + 2] = block
    return H


def k_path():
    pts = []
    for p in range(21):
        if p <= 10:
            n = np.array([0.5 - p / 20.0] * 3)
        else:
            n = np.array([(p - 10) / 10.0, 0.0, 0.0])
        pts.append(n * TWOPi_A)
    return pts


def main():
    here = pathlib.Path(__file__).resolve()
    ref_path = here.parent.parent / "hidden_files" / "notebook1_prints.json"
    ref = json.loads(ref_path.read_text())["prints_descending_as_in_notebook"]

    max_dev_all = 0.0
    max_herm = 0.0
    rows = []
    for p, k in enumerate(k_path()):
        H = hamiltonian(k)
        max_herm = max(max_herm, float(np.max(np.abs(H - H.conj().T))))
        ev = np.linalg.eigvalsh((H + H.conj().T) / 2.0)  # ascending
        ref_sorted = np.sort(np.array(ref[p]))
        dev = float(np.max(np.abs(ev - ref_sorted)))
        max_dev_all = max(max_dev_all, dev)
        rows.append((p, dev))
    print(f"Hermiticity: max |H - H^dag| = {max_herm:.3e} eV")
    print(f"Mirror vs notebook: max |dE| over 21 k-points x 30 bands = {max_dev_all:.3e} eV")
    worst = sorted(rows, key=lambda r: -r[1])[:3]
    print("worst k-points (p, max|dE|):", [(p, f"{d:.2e}") for p, d in worst])

    # Headline edges from the mirror itself
    def edges(p):
        ev = np.linalg.eigvalsh((lambda H: (H + H.conj().T) / 2.0)(hamiltonian(k_path()[p])))
        return ev[7], ev[8], ev[8] - ev[7], ev[7] - ev[3]
    for p, name in [(0, "L"), (10, "Gamma"), (20, "X")]:
        vbm, cbm, gap, dso = edges(p)
        print(f"{name}: VBM={vbm:.4f} CBM={cbm:.4f} Eg={gap:.4f}" +
              (f" Delta_so={dso:.4f}" if name == "Gamma" else ""))


if __name__ == "__main__":
    main()
