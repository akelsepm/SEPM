# Parameter Pool — INDEX

Production scheme: 51 plane waves (G^2 <= 11), full form-factor mask, free beta, LOCAL pseudopotential.
Valley targets verified against the Ioffe NSM archive (2026-11-19); GaAs locked;
AlSb/AlAs valley masses unverified (exploratory).
Cohen-Bergstresser (1966) / Chelikowsky-Cohen (1976) form factors are seeds only:
used for GaSb; collapsed for InAs/InSb/InP (incomplete without their nonlocal wells).

| Anchor | Cost | Mode |
|---|---|---|
| GaAs_300K | 65.9 | fit-v3r2 LOCKED (verified valley targets, 2026-11-19) |
| GaAs_77K | 77.8 | fit-pool LOCKED (verified valley targets, 2026-11-19) |
| InAs_300K | 137.1 | fit-pool R3 (DOS targets, narrow-gap profile) |
| InAs_77K | 172.3 | fit-pool R3 (DOS targets, narrow-gap profile) |
| GaSb_300K | 110.2 | fit-pool (CC76 seed, verified targets) |
| GaSb_77K | 123.9 | fit-pool (CC76 seed, verified targets) |
| AlSb_300K | 178.8 | fit-pool (CB66 seed) |
| AlSb_77K | 132.1 | fit-pool EXPLORATORY |
| InSb_300K | 608.6 | fit-pool R4 (InSb_77K seed, verified targets) |
| InSb_77K | 798.0 | fit-pool R4 (InAs seed, verified targets) |
| AlAs_300K | 114.0 | fit-pool EXPLORATORY |
| AlAs_77K | 106.6 | fit-pool EXPLORATORY |
| InP_300K | 74.8 | fit-pool (verified targets, own constants) |
| InP_77K | 94.9 | fit-pool (verified targets, own constants) |
| GaP_300K | 136.5 | fit-pool (verified targets, own constants) |
| GaP_77K | 137.4 | fit-pool (verified targets, own constants) |

GaAs_300K_classic/: the pedagogical classic set of Chapter 6 (15 PW) — archive.
