"""Layer-3 ternary worker: VCA observables along a composition grid.

One (system, x, T) point per subprocess (one scheme per process rule).
The point's interpolated parameter set is materialized as a scratch pool
entry, computed through the proven bulk_from_pool path, and the scratch
entry is removed afterwards. Results are JSON lines on stdout.

Usage:
    python3 ternary_worker.py point AlGaAs 0.30 300
    python3 ternary_worker.py grid  AlGaAs 300 0,0.1,...,1.0  out.json
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def point(system, x, T):
    import numpy as np

    from bulk_properties import bulk_from_pool
    from pool import get_params

    p = get_params(system, x=x, T=T)
    tag = f"_{system}_x{int(round(100 * x)):03d}_{T}K"
    entry = HERE / "params_pool" / tag
    entry.mkdir(exist_ok=True)
    doc = {"config": tag, "scheme": "51PW (G^2<=11) + full mask + beta free",
           "a0_m": p["a0"], "basis_q2": [0, 3, 4, 8, 11],
           "params_Ry_and_scalars": {
               **{f"VS{q}": p["VS_RY"][q] for q in (3, 4, 8, 11)},
               **{f"VA{q}": p["VA_RY"][q] for q in (3, 4, 8, 11)},
               "L": p["L_SO"], "beta": p["BETA"]},
           "cost": None, "mode": "VCA-interpolated (NOT fitted)",
           "target_source": "pool interpolation",
           "note": "Scratch entry, removed after the run."}
    (entry / "params.json").write_text(json.dumps(doc, indent=1))
    try:
        blk, mat = bulk_from_pool(tag)
        E = blk.edges()
        g100 = blk.gamma_masses(np.array([1.0, 0.0, 0.0]))
        g111 = blk.gamma_masses(np.array([1.0, 1.0, 1.0]) / np.sqrt(3))
        lut = blk.luttinger(g100, g111)
        vL = blk.valley_masses("L")
        vX = blk.valley_masses("X")
        out = {"system": system, "x": x, "T": T, "a0_A": p["a0"] * 1e10,
               "edges": E, "m100": g100,
               "gamma1": lut["gamma1_from_100"], "gamma2": lut["gamma2"],
               "gamma3": lut["gamma3"],
               "valley_L": {k: vL.get(k) for k in
                            ("stationary_at_point", "m_long", "m_trans",
                             "m_dos_valley")},
               "valley_X": {k: vX.get(k) for k in
                            ("stationary_at_point", "m_long", "m_trans",
                             "m_dos_valley")},
               "provenance": p["provenance"]["flag"]}
        print("RESULT " + json.dumps(out))
    finally:
        (entry / "params.json").unlink(missing_ok=True)
        entry.rmdir()


def grid(system, T, xs, out_path):
    rows = []
    for x in xs:
        r = subprocess.run([sys.executable, __file__, "point", system,
                            str(x), str(T)], capture_output=True, text=True)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
        if not line:
            print(f"x={x}: FAILED\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
            continue
        rows.append(json.loads(line[0][7:]))
        print(f"x={x}: Eg={rows[-1]['edges']['Eg_Gamma_eV']:.4f} "
              f"E_L={rows[-1]['edges']['E_L_eV']:.4f} "
              f"E_X={rows[-1]['edges']['E_X_eV']:.4f}", flush=True)
    Path(out_path).write_text(json.dumps(rows, indent=1))
    print("wrote", out_path, len(rows), "points")


if __name__ == "__main__":
    if sys.argv[1] == "point":
        point(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
    else:
        grid(sys.argv[2], int(sys.argv[3]),
             [float(s) for s in sys.argv[4].split(",")], sys.argv[5])
