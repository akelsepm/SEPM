"""Chapter 7 atlas worker: properties + dense band path for ONE pool anchor.

Usage: python3 atlas_worker.py <PoolName>   (e.g. GaAs_300K)
Writes properties_atlas.json into the pool entry and band data to
../files/book-bolum-01/atlas_data/<PoolName>_bands.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

import bulk_properties as BP

HERE = Path(__file__).parent
OUT = HERE.parent / "files" / "book-bolum-01" / "atlas_data"
OUT.mkdir(parents=True, exist_ok=True)

NB = 14  # spinor bands tracked (of 102)


def band_path(bulk):
    """X -> Gamma -> L path, 201 points per segment, overlap tracking."""
    U = 2 * np.pi / bulk.mat["a0"]
    X = np.array([1.0, 0.0, 0.0])
    G = np.zeros(3)
    L = np.array([0.5, 0.5, 0.5])
    segs = [(X, G), (G, L)]
    xs, es_all, ticks = [], [], [0]
    xoff = 0.0
    prev_vecs = None
    for a, b in segs:
        n = 201
        for i in range(n):
            if i == 0 and xs:
                continue
            t = i / (n - 1)
            k = (a + t * (b - a)) * U
            ev, vec = bulk.eigs(k, vectors=True)
            ev, vec = np.asarray(ev), np.asarray(vec)
            if prev_vecs is not None:
                ov = np.abs(prev_vecs.conj().T @ vec[:, :NB])  # NB x NB
                order = []
                used = set()
                for r in range(NB):
                    c = int(np.argmax(ov[r]))
                    while c in used:
                        ov[r, c] = -1
                        c = int(np.argmax(ov[r]))
                    used.add(c)
                    order.append(c)
                ev_t = ev[order]
                vec_t = vec[:, order]
            else:
                ev_t, vec_t = ev[:NB], vec[:, :NB]
            prev_vecs = vec_t
            xs.append(xoff + t * np.linalg.norm(b - a))
            es_all.append(ev_t)
        xoff += np.linalg.norm(b - a)
        ticks.append(xoff)
    return np.array(xs), np.array(es_all).T, ticks  # (NB, npts)


def main():
    name = sys.argv[1]
    bulk, _mat = BP.bulk_from_pool(name)
    edges = bulk.edges()
    g100 = bulk.gamma_masses(np.array([1.0, 0.0, 0.0]))
    g110 = bulk.gamma_masses(np.array([1.0, 1.0, 0.0]) / np.sqrt(2))
    g111 = bulk.gamma_masses(np.array([1.0, 1.0, 1.0]) / np.sqrt(3))
    lut = BP.Bulk.luttinger(g100, g111)
    vL = bulk.valley_masses("L")
    vX = bulk.valley_masses("X")
    props = {"pool": name, "edges": edges,
             "gamma_masses": {"100": g100, "110": g110, "111": g111},
             "luttinger": lut, "valley_L": vL, "valley_X": vX}
    pp = HERE / "params_pool" / name / "properties_atlas.json"
    pp.write_text(json.dumps(props, indent=2, default=float))
    xs, es, ticks = band_path(bulk)
    np.savez(OUT / f"{name}_bands.npz", x=xs, E=es, ticks=ticks, vbm=bulk.vbm)
    print(name, "ok  Eg=", round(edges["Eg_Gamma_eV"], 4),
          " g1=", round(lut["gamma1_from_100"], 3))


if __name__ == "__main__":
    main()
