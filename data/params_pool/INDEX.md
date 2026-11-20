# Parameter Pool — INDEX

Production scheme: 51 plane waves (G^2 <= 11), full form-factor mask, free beta, LOCAL pseudopotential.
Valley targets verified against the Ioffe NSM archive (2026-11-19); GaAs locked;
AlSb/AlAs valley masses unverified (exploratory).
Cohen-Bergstresser (1966) / Chelikowsky-Cohen (1976) form factors are seeds only:
used for GaSb; collapsed for InAs/InSb/InP (incomplete without their nonlocal wells).

Costs and observables were re-measured on 2026-11-19 under the symmetric
valley-curvature estimator (parameters unchanged; see
code/refresh_pool_observables.py). Costs compare only within one material.

| Anchor | Cost | Mode |
|---|---|---|
| GaAs_300K | 58.2 | fit-v3 LOCKED — frozen pair anchor (pair acceptance battery, Ch. 8) |
| GaAs_77K | 69.7 | fit-v3 LOCKED — frozen pair anchor (pair acceptance battery, Ch. 8) |
| InAs_300K | 241.7 | fit-pool R3 (DOS targets, narrow-gap profile) |
| InAs_77K | 276.4 | fit-pool R3 (DOS targets, narrow-gap profile) |
| GaSb_300K | 108.6 | fit-pool (CC76 seed, verified targets) |
| GaSb_77K | 135.0 | fit-pool (CC76 seed, verified targets) |
| AlSb_300K | 128.7 | fit-pool v4 (CB66 seed; batch round, symmetric estimator) |
| AlSb_77K | 148.4 | fit-pool (CB66 seed) EXPLORATORY |
| InSb_300K | 737.7 | fit-pool R4 (InSb_77K seed, verified targets) |
| InSb_77K | 937.1 | fit-pool R4 (InAs seed, verified targets) |
| AlAs_300K | 173.9 | fit-pool EXPLORATORY (batch v4 not registered: E_L trade-off) |
| AlAs_77K | 159.0 | fit-pool EXPLORATORY (v4 reverted 2026-11-19: failed ternary midpoint test) |
| InP_300K | 152.1 | fit-pool (verified targets, own constants) |
| InP_77K | 69.3 | fit-pool v4 (batch round, symmetric estimator) |
| GaP_300K | 126.5 | fit-pool v4 (batch round, symmetric estimator) |
| GaP_77K | 125.3 | fit-pool v4 (batch round, symmetric estimator) |

Alloy layer: pool.py BOWING is active for AlGaAs — VA4, b = +0.00862 Ry,
fitted to the VMR E_X(x) curve at 300 K on the frozen GaAs+AlAs pair
(77 K transfer verified). The term vanishes at both endpoints; anchors
are untouched by construction.

GaAs_300K_classic/: the pedagogical classic set of Chapter 6 (15 PW) — archive.
