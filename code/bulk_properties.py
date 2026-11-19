"""
Bulk electronic properties from a LOCKED EPM parameter set (Layer 2).

Generalised design: a "material" is any parameter set in the locked
scheme (classic mask, 15 PW, single-L SO, beta) plus its lattice
constant. Properties are computed from the band structure only:

  * band edges at Gamma / L / X and Delta_so (VBM reference)
  * Gamma effective masses along [100], [110], [111]:
    electron, heavy hole, light hole (Gamma8 subspace tracking),
    split-off hole (Gamma7 pair tracking)
  * L- and X-valley masses: longitudinal + transverse (state tracking)
  * Luttinger parameters gamma1, gamma2, gamma3 inverted from the
    Gamma hole masses, with the [111]-based gamma1 as consistency check
  * density of states on a uniform BZ grid (states / eV / unit cell)

Usage:
    python3 bulk_properties.py GaAs        # writes into params_pool/GaAs_300K/

To add a material: fit it with fit_engine.py, register it in the pool,
add an entry to MATERIALS below. No other code changes.
"""
import json
import sys
from pathlib import Path

import numpy as np

import fit_gaas_300k as F  # Hamiltonian machinery (scheme-fixed)

HBAR2_2M0 = F.HBAR2_OVER_2M0  # eV * A^2  (hbar^2 / 2 m0)
DELTAS = np.array([0.01, 0.02, 0.03, 0.04])

# ---------------------------------------------------------------- materials
# Each entry: pool directory name + parameter source. Parameters are the
# LOCKED sets; see params_pool/<dir>/params.json for provenance.
def _from_module(mod):
    return dict(a0=mod.A0, VS_RY=mod.VS_RY, VA_RY=mod.VA_RY,
                L_SO=mod.L_SO, BETA=mod.BETA)

import params_gaas_300k as _gaas
MATERIALS = {
    "GaAs": dict(pool="GaAs_300K", **_from_module(_gaas)),
}


def build_V_of(mat):
    p = [mat["VS_RY"][q] for q in (3, 4, 8, 11)] + \
        [mat["VA_RY"][q] for q in (3, 4, 8, 11)]
    return F.build_V(p)


class Bulk:
    def __init__(self, mat):
        self.mat = mat
        self.V = build_V_of(mat)
        self.L = mat["L_SO"]
        self.beta = mat["BETA"]
        self.U = 2 * np.pi / mat["a0"]  # 2pi/a in m^-1 (Hamiltonian units)
        self.K = {"G": np.zeros(3),
                  "L": 0.5 * self.U * np.array([1., 1., 1.]),
                  "X": self.U * np.array([1., 0., 0.])}
        eg = self.eigs(self.K["G"])
        self.vbm = eg[7]

    def eigs(self, k, vectors=False):
        return F.eigs(np.asarray(k, float), self.V, self.L, self.beta,
                      vectors=vectors)

    # ------------------------------------------------------------- edges
    def edges(self):
        eg = self.eigs(self.K["G"]); el = self.eigs(self.K["L"])
        ex = self.eigs(self.K["X"])
        return {"Eg_Gamma_eV": eg[8] - self.vbm,
                "E_L_eV": el[8] - self.vbm,
                "E_X_eV": ex[8] - self.vbm,
                "Delta_so_eV": eg[7] - eg[3]}

    # ------------------------------------------------- tracked band mass
    def _track(self, k0, state0, direction, deltas=DELTAS):
        """Follow one non-degenerate state from k0 along `direction`;
        return energies at |dk| = deltas * U (plus 0)."""
        d = np.asarray(direction, float); d /= np.linalg.norm(d)
        prev = state0
        es = []
        for dl in np.concatenate([[0.0], deltas]):
            w, v = self.eigs(k0 + dl * self.U * d, vectors=True)
            if dl == 0.0:
                j = int(np.argmax(np.abs(v.conj().T @ prev) ** 2))
                prev = v[:, j]; es.append(w[j]); continue
            j = int(np.argmax(np.abs(v.conj().T @ prev) ** 2))
            prev = v[:, j]; es.append(w[j])
        kk = np.concatenate([[0.0], deltas]) * self.U
        a = np.polyfit(kk ** 2, np.array(es), 1)[0]
        return abs(HBAR2_2M0 / a)

    def fold(self, k):
        """Reduce a STRICTLY exterior k to the first BZ (nearest
        reciprocal-lattice shift). Points inside or on the BZ are
        returned unchanged: the truncated basis is not periodic, and an
        unnecessary shift changes the spectrum it produces."""
        k = np.asarray(k, float)
        ku = k / self.U
        s = max(abs(ku[0] + ku[1] + ku[2]), abs(ku[0] + ku[1] - ku[2]),
                abs(ku[0] - ku[1] + ku[2]), abs(-ku[0] + ku[1] + ku[2]))
        if np.max(np.abs(ku)) <= 1.0 + 1e-9 and s <= 1.5 + 1e-9:
            return k
        best, bestn = k, np.inf
        for n1 in (-2, -1, 0, 1, 2):
            for n2 in (-2, -1, 0, 1, 2):
                for n3 in (-2, -1, 0, 1, 2):
                    if not (n1 % 2 == n2 % 2 == n3 % 2):
                        continue  # fcc reciprocal: n1,n2,n3 same parity
                    cand = k - self.U * np.array([n1, n2, n3], float)
                    nn = float(cand @ cand)
                    if nn < bestn:
                        best, bestn = cand, nn
        return best

    def _pair_mass(self, k0, pair_idx, direction, deltas=DELTAS,
                   signs=(+1, -1)):
        """Mass of a Kramers pair tracked as a 2-dim subspace (mean
        energy). Sampling is kept inside the BZ: at a zone-face point
        the exterior side is the time-reversed mirror of the interior
        side, so a one-sided interior fit (signs=(+1,) with `direction`
        pointing inward) is the correct symmetric parabola."""
        d = np.asarray(direction, float); d /= np.linalg.norm(d)
        w0, v0 = self.eigs(k0, vectors=True)
        sub0 = v0[:, list(pair_idx)]
        es = {0.0: float(np.mean(w0[list(pair_idx)]))}
        for sgn in signs:
            sub = sub0
            for dl in deltas:
                w, v = self.eigs(k0 + sgn * dl * self.U * d, vectors=True)
                weight = np.sum(np.abs(sub.conj().T @ v) ** 2, axis=0)
                idx = np.argsort(weight)[-2:]
                sub = v[:, idx]
                es[sgn * dl] = float(np.mean(w[idx]))
        ks = np.array(sorted(es)) * self.U
        a = np.polyfit(ks ** 2, np.array([es[s] for s in sorted(es)]), 1)[0]
        return abs(HBAR2_2M0 / a)

    def gamma_masses(self, direction):
        me, mlh, mhh = F.parabolic_masses(self.V, self.L, self.beta, direction)
        w0, v0 = self.eigs(self.K["G"], vectors=True)
        mso = self._pair_mass(self.K["G"], (2, 3), direction)
        return {"m_e": me, "m_lh": mlh, "m_hh": mhh, "m_so": mso}

    def valley_masses(self, valley):
        """Valley report with stationarity diagnostics. Transverse masses
        are pair-tracked at the symmetry point. The longitudinal profile
        is scanned: if the symmetry point is not the line minimum, the
        displaced minimum (position, energy, longitudinal mass there) is
        reported instead of a meaningless parabola at a non-stationary
        point."""
        k0 = self.K[valley]
        if valley == "L":
            axis = np.array([1., 1, 1]); trs = [[1, -1, 0], [1, 1, -2]]
            trange = (0.35, 0.5)
        else:  # X
            axis = np.array([1., 0, 0]); trs = [[0, 1, 0], [0, 0, 1]]
            trange = (0.8, 1.0)
        mt = float(np.mean([self._pair_mass(k0, (8, 9), t) for t in trs]))
        ts = np.linspace(*trange, 31)
        Es = np.array([self.eigs(t * self.U * axis)[8] for t in ts])
        i = int(np.argmin(Es))
        rep = {"m_trans": mt, "E_point_eV": float(Es[-1] - self.vbm)}
        if i < len(ts) - 1:  # minimum is NOT at the symmetry point
            tmin, emin = float(ts[i]), float(Es[i] - self.vbm)
            kmin = tmin * self.U * axis
            ml = self._pair_mass(kmin, (8, 9), axis)
            rep.update({"line_min_t": tmin, "line_min_E_eV": emin,
                        "m_long_at_line_min": ml,
                        "m_dos_valley": (ml * mt ** 2) ** (1 / 3),
                        "stationary_at_point": False})
        else:  # stationary at the point: one-sided interior parabola
            inward = -axis
            ml = self._pair_mass(k0, (8, 9), inward, signs=(+1,))
            d = axis / np.linalg.norm(axis)
            e0 = float(np.mean(self.eigs(k0)[8:10]))
            a_small = (float(np.mean(self.eigs(k0 - 0.01 * self.U * d)[8:10])) - e0) / 0.01 ** 2
            a_large = (float(np.mean(self.eigs(k0 - 0.04 * self.U * d)[8:10])) - e0) / 0.04 ** 2
            rep.update({"m_long": ml, "m_dos_valley": (ml * mt ** 2) ** (1 / 3),
                        "stationary_at_point": True,
                        "parabolicity_ratio": float(a_small / a_large)})
        return rep

    # ---------------------------------------------------------- Luttinger
    @staticmethod
    def luttinger(g100, g111):
        i_hh100, i_lh100 = 1 / g100["m_hh"], 1 / g100["m_lh"]
        i_hh111, i_lh111 = 1 / g111["m_hh"], 1 / g111["m_lh"]
        g1_a = 0.5 * (i_hh100 + i_lh100)
        g2 = 0.25 * (i_lh100 - i_hh100)
        g1_b = 0.5 * (i_hh111 + i_lh111)
        g3 = 0.25 * (i_lh111 - i_hh111)
        return {"gamma1_from_100": g1_a, "gamma1_from_111": g1_b,
                "gamma2": g2, "gamma3": g3}

    # ---------------------------------------------------------------- DOS
    def dos(self, n=44, nbands=30):
        U = self.U
        ks = np.linspace(-U, U, n)
        pts = []
        for kx in ks:
            for ky in ks:
                for kz in ks:
                    k = np.array([kx, ky, kz])
                    if np.max(np.abs(k)) > U + 1e-9:
                        continue
                    s = max(abs(kx + ky + kz), abs(kx + ky - kz),
                            abs(kx - ky + kz), abs(-kx + ky + kz))
                    if s <= 1.5 * U + 1e-9:
                        pts.append(k)
        evals = []
        for k in pts:
            evals.append(self.eigs(k)[:nbands])
        E = np.concatenate(evals) - self.vbm
        bins = np.arange(-9.0, 5.01, 0.005)
        hist, edges_b = np.histogram(E, bins=bins)
        # (1/N_k) sum_k sum_n delta(E - E_nk): integral over E = nbands/cell.
        dos = hist / (len(pts) * 0.005)
        centres = 0.5 * (bins[1:] + bins[:-1])
        kern = np.exp(-0.5 * (np.arange(-60, 61) * 0.005 / 0.025) ** 2)
        kern /= kern.sum()
        dos_s = np.convolve(dos, kern, mode="same")
        return centres, dos_s, len(pts)


def run(name):
    mat = MATERIALS[name]
    b = Bulk(mat)
    out = {"material": name, "a0_A": mat["a0"] * 1e10}
    out["edges"] = b.edges()
    g = {d: b.gamma_masses(dirv) for d, dirv in
         (("100", [1, 0, 0]), ("110", [1, 1, 0]), ("111", [1, 1, 1]))}
    out["gamma_masses_m0"] = g
    out["luttinger"] = Bulk.luttinger(g["100"], g["111"])
    out["valleys"] = {v: b.valley_masses(v) for v in ("L", "X")}
    centres, dos_s, nk = b.dos()
    out["dos_grid_points"] = nk

    pool = Path(__file__).parent / "params_pool" / mat["pool"]
    np.savez(pool / "dos.npz", energy_eV=centres, dos=dos_s)
    (pool / "properties.json").write_text(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=200)
    axes[0].plot(centres, dos_s, color="#0f766e", lw=1.2)
    axes[0].set_xlim(-8, 4); axes[0].set_xlabel("E − VBM (eV)")
    axes[0].set_ylabel("DOS (durum/eV/hücre)")
    axes[0].set_title(f"{name}: tam DOS", loc="left", fontsize=10.5)
    axes[1].plot(centres, dos_s, color="#0f766e", lw=1.2)
    axes[1].set_xlim(-1.2, 2.6); axes[1].set_ylim(0, 8)
    axes[1].axvline(0, color="#9ca3af", lw=0.7, ls="--")
    axes[1].set_xlabel("E − VBM (eV)")
    axes[1].set_title("Bant kenarı yakınları", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(pool / "dos.png")

    L = out["luttinger"]; V = out["valleys"]; E = out["edges"]
    md = [f"# {name} — elektronik parametreler (Katman 2, kilitli setten)",
          "",
          f"Kenarlar: Eg(Γ) {E['Eg_Gamma_eV']:.4f} eV · E_L {E['E_L_eV']:.4f} · "
          f"E_X {E['E_X_eV']:.4f} · Δso {E['Delta_so_eV']:.4f} eV",
          "",
          "| Yön | m_e | m_hh | m_lh | m_so |",
          "|---|---|---|---|---|"]
    for d in ("100", "110", "111"):
        r = g[d]
        md.append(f"| [{d}] | {r['m_e']:.4f} | {r['m_hh']:.4f} | "
                  f"{r['m_lh']:.4f} | {r['m_so']:.4f} |")
    md += ["",
           f"Luttinger: γ1[100] {L['gamma1_from_100']:.3f} · γ1[111] "
           f"{L['gamma1_from_111']:.3f} · γ2 {L['gamma2']:.3f} · γ3 {L['gamma3']:.3f}",
           ""]
    Lv, Xv = V["L"], V["X"]
    if Lv.get("stationary_at_point"):
        md.append(f"L vadisi: m_l {Lv['m_long']:.3f} · m_t {Lv['m_trans']:.4f} · "
                  f"m_dos {Lv['m_dos_valley']:.3f} (dejenerelik 4)")
    else:
        md.append(f"L vadisi: m_t(L) {Lv['m_trans']:.4f} · UYARI: L noktası Λ "
                  f"boyunca durağan değil — çizgi minimumu t={Lv['line_min_t']:.3f} "
                  f"konumunda, E {Lv['line_min_E_eV']:.4f} eV "
                  f"(E_L'den {1000*(Lv['E_point_eV']-Lv['line_min_E_eV']):.1f} meV aşağıda), "
                  f"orada m_l {Lv['m_long_at_line_min']:.3f}, m_dos {Lv['m_dos_valley']:.3f}")
    if Xv.get("stationary_at_point"):
        md.append(f"X vadisi: m_l {Xv['m_long']:.3f} (paraboliklik oranı "
                  f"{Xv['parabolicity_ratio']:.2f}) · m_t {Xv['m_trans']:.4f} · "
                  f"m_dos {Xv['m_dos_valley']:.3f} (dejenerelik 3)")
    md += ["",
           f"DOS: {nk} k noktası, dos.npz + dos.png. Birim: durum/eV/ilkel hücre.",
           "Genelleştirme: bulk_properties.py MATERIALS sözlüğü — yeni malzeme = fit + havuz kaydı."]
    (pool / "properties.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "GaAs")
