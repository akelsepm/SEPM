"""
LOCKED Layer-1 parameter set - GaAs bulk @ 300 K  (decision: Kazim, 2026-11-18)
=============================================================================
Classic mask (VS4 = VA4 = VA8 = 0), 15 plane-wave basis, single-L
spin-orbit scheme, beta FIXED at Kazim's Dente-Tilton value 1/0.9
(m1 = 0.9 m0). Fitted against Vurgaftman et al. (2001) 300 K targets:
gaps/valleys <= 3.6 meV, Delta_so 1.6 meV, masses within ~1-8%.
Fit cost 13.68 (beta-free classic: 9.51 - the difference is the price
of keeping beta, measured and accepted).

Source: hidden_files/fit_gaas_300k_classic_beta_fixed.json
Report: files/Fit-Raporu-01-GaAs-300K.md

WARNING: these numbers are bound to THIS scheme (15 PW basis + this SO
form). They do not transfer to another basis - see Report finding 2.
"""

A0 = 5.653e-10                 # lattice constant [m]
RY_TO_EV = 13.6059

VS_RY = {3: -0.27580, 4: 0.0, 8: 0.04202, 11: 0.06032}
VA_RY = {3:  0.02748, 4: 0.0, 8: 0.0,     11: 0.05437}

BETA = 1.0 / 0.9               # kinetic-energy scaling (m1 = M0 / BETA)
L_SO = 4580.31                 # empirical spin-orbit enhancement

# Verification values this set produces (Vurgaftman targets in brackets):
#   Eg(Gamma) = 1.4204 [1.424]   E_L = 1.7104 [1.708]   E_X = 1.8977 [1.900]
#   Delta_so  = 0.3426 [0.341]
#   m_e[100]  = 0.0649 [0.067]   m_lh[100] = 0.0911 [0.090]  m_hh[100] = 0.3594 [0.350]
#   m_e[111]  = 0.0665 [0.067]   m_lh[111] = 0.0855 [0.078]  m_hh[111] = 0.8666 [0.893]
