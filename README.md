# SEPM — Superlattice Empirical Pseudopotential Method

Open code and data accompanying the monograph in preparation
*Süperörgülerde Ampirik Pseudopotansiyel Yöntemi* (Empirical Pseudopotential
Method for Superlattices) by Kazım Akel.

Everything here is public on purpose: fitted parameter sets, the fit engine,
the property-extraction code and the book chapters (PDF). The only part that
will stay private is the final QCL design stage; that decision belongs to
the end of the project, not to its foundations.

## Contents
- `code/` — Python implementation (plane-wave EPM with spin–orbit,
  classic-mask fits, bulk property extraction). The Mathematica original
  remains the reference implementation; the Python code mirrors it.
- `data/params_pool/` — locked parameter sets per material/temperature,
  each with its fit report, band figure and extracted electronic
  properties (edges, effective masses, Luttinger parameters, DOS).
- `book/` — chapter PDFs (Turkish), produced in order, Chapter 1–6.

## Method in one paragraph
Zinc-blende binaries are fitted in a 15 plane-wave basis with the classic
Cohen–Bergstresser mask (VS4 = VA4 = VA8 = 0), a single spin–orbit scale
and a fixed kinetic scaling beta = 1/0.9 (Magri–Zunger rationale).
Fit targets: band edges (Gamma, L, X), spin–orbit splitting, Gamma
effective masses, and — since v2 — L/X valley shapes (transverse masses,
Lambda-line stationarity, X curvature). Targets are the Vurgaftman et al.
(2001) recommended values at 300 K and 77 K.

## License
Code: MIT. Data and book chapters: CC BY 4.0. See LICENSE files.
