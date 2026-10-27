"""
lorentzian_regge_4d.py
======================
Linearised LORENTZIAN Regge calculus on a periodic Freudenthal (Kuhn) triangulated hypercubic
4D lattice (24 four-simplices per hypercube, 15 edge directions per vertex) around flat
Minkowski space with background metric G = diag(-alpha^2, 1, 1, 1) (time step dt = alpha * dx).

What is postulated (the lattice model):
  * spacetime is piecewise flat (flat Lorentzian 4-simplices glued along tetrahedra);
  * the dynamics is the Regge action S = sum_triangles |A_t| eps_t (Regge 1961; Sorkin 1975),
    whose only dimensionful input is the edge lengths.
What is NOT put in: Einstein's equations, the metric field equations, the graviton, its speed,
its polarisations, or the relative signs of timelike/spacelike hinge terms (these are fixed
intrinsically by the Lorentzian Schlafli identity of each flat 4-simplex).

Everything is computed from the 10 squared edge lengths of each 4-simplex: triangle areas
(Heron), rotation angles at timelike triangles, boost angles at spacelike triangles. First
derivatives are obtained by complex-step differentiation (exact to round-off), and the
second variation at flat space is H_ef = -sum_sigma sum_t c_t (d|A_t|/ds_e)(d theta_t/ds_f).

Scope (honest): this is the LINEARISED theory on a flat background. It derives the graviton
sector of GR (gauge invariance / Bianchi identity, 2 polarisations, propagation at c, no other
propagating mode) from the lattice action and evolves it in discrete time. It does not perform
nonlinear 3+1D Regge evolution, and the Freudenthal complex is not the 18-point stencil.

Lattice artefacts found and reported (not hidden):
  * The 4D hyperdiagonal edge (1,1,1,1) drops out of the quadratic Regge action identically (its
    Hessian column vanishes at every k), although it changes deficit angles. It is undetermined by
    the linearised equations and is fixed by hand together with the 4 lapse/shift edges.
  * Free (unconstrained) slice-by-slice evolution has no exponentially growing mode (all transfer
    eigenvalues |lambda| <= 1), but constraint-violating perturbations can grow polynomially
    (sub-linearly to linearly in n), the usual behaviour of free evolution without constraint damping.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

DIRS = [d for d in itertools.product((0, 1), repeat=4) if any(d)]   # 15 edge directions (t,x,y,z)
DID = {d: i for i, d in enumerate(DIRS)}
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]           # 10 edges of a 4-simplex
SYM = [(m, n) for m in range(4) for n in range(m, 4)]                 # 10 metric components
GAUGE_FIX_DIRS = [(1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1)]


def freudenthal_simplices():
    """24 four-simplices of the unit hypercube: monotone lattice paths 0 -> (1,1,1,1)."""
    out = []
    for perm in itertools.permutations(range(4)):
        v = [np.zeros(4, dtype=int)]
        for ax in perm:
            nv = v[-1].copy()
            nv[ax] += 1
            v.append(nv)
        out.append(v)
    return out


def simplex_geometry(s, lorentzian: bool):
    """
    s: length-10 array of squared edge lengths (ordered as PAIRS), real or complex (complex step).
    Returns areas[10], angles[10], kinds[10]; triangle index = index of the complementary vertex
    pair (i, j) whose facets meet at that triangle.
    Timelike triangle (spacelike normal plane): rotation angle arccos(-<n_i,n_j>/|n_i||n_j|).
    Spacelike triangle (Lorentzian normal plane): boost angle arccosh|f| (normals of same type)
    or arcsinh f (mixed type). Signs relative to each other are fixed later by Schlafli.
    """
    sq = np.zeros((5, 5), dtype=complex)
    for e, (i, j) in enumerate(PAIRS):
        sq[i, j] = sq[j, i] = s[e]
    Gm = np.zeros((4, 4), dtype=complex)
    for a in range(1, 5):
        for b in range(1, 5):
            Gm[a - 1, b - 1] = 0.5 * (sq[0, a] + sq[0, b] - sq[a, b])
    Gi = np.linalg.inv(Gm)
    B = np.vstack([-np.ones(4), np.eye(4)])
    N = B @ Gi @ B.T                                   # <n^i, n^j>, inward facet normals
    areas = np.zeros(10, dtype=complex)
    angles = np.zeros(10, dtype=complex)
    kinds = []
    for idx, (i, j) in enumerate(PAIRS):
        tri = [v for v in range(5) if v not in (i, j)]
        s1, s2, s3 = sq[tri[0], tri[1]], sq[tri[1], tri[2]], sq[tri[0], tri[2]]
        a16 = 2 * (s1 * s2 + s2 * s3 + s3 * s1) - (s1 ** 2 + s2 ** 2 + s3 ** 2)
        areas[idx] = np.sqrt(np.sign(a16.real) * a16) / 4.0          # |A_t|
        nn = N[i, i] * N[j, j]
        D = nn - N[i, j] ** 2
        if (not lorentzian) or D.real > 0:
            f = -N[i, j] / np.sqrt(nn)
            angles[idx] = np.arccos(f)
            kinds.append("T" if lorentzian else "E")
            if lorentzian:
                assert a16.real < 0, "timelike triangle must have negative Heron area^2"
        else:
            f = -N[i, j] / np.sqrt(np.sign(nn.real) * nn)
            if nn.real > 0:
                angles[idx] = np.arccosh(np.sign(f.real) * f)
                kinds.append("S_same")
            else:
                angles[idx] = np.arcsinh(f)
                kinds.append("S_mixed")
            assert a16.real > 0, "spacelike triangle must have positive Heron area^2"
    return areas, angles, kinds


def simplex_derivatives(verts, G, lorentzian: bool, h: float = 1e-30):
    """Complex-step derivatives d|A_t|/ds_e and d theta_t/ds_e at the flat simplex (exact to round-off)."""
    s0 = np.array([float((verts[j] - verts[i]) @ G @ (verts[j] - verts[i])) for (i, j) in PAIRS])
    A0, th0, kinds = simplex_geometry(s0.astype(complex), lorentzian)
    dA = np.zeros((10, 10))
    dth = np.zeros((10, 10))
    for e in range(10):
        sp = s0.astype(complex)
        sp[e] += 1j * h
        Ap, tp, kp = simplex_geometry(sp, lorentzian)
        assert kp == kinds
        dA[:, e] = Ap.imag / h
        dth[:, e] = tp.imag / h
    return s0, A0.real, th0.real, kinds, dA, dth


class LorentzianReggeLattice:
    """Linearised Regge calculus on the periodic Freudenthal lattice with G = diag(+-alpha^2,1,1,1)."""

    def __init__(self, alpha: float = 0.5, lorentzian: bool = True):
        self.alpha = alpha
        self.lorentzian = lorentzian
        self.G = np.diag([(-1.0 if lorentzian else 1.0) * alpha ** 2, 1.0, 1.0, 1.0])
        self.blocks = []          # (dir a, base a, dir b, base b, h_ab)
        self.deficit_terms = []   # (triangle key, owner, dir e, base e, d eps / d s_e)
        self.schlafli_rel_residuals = []
        self.hinge_weights = []
        self.hinge_kinds = []
        tri_keys = {}
        for v in freudenthal_simplices():
            s0, A0, th0, kinds, dA, dth = simplex_derivatives(v, self.G, lorentzian)
            M = A0[:, None] * dth                       # [triangle, edge]
            U, S, Vt = np.linalg.svd(M.T)
            c = Vt[-1]
            self.schlafli_rel_residuals.append(float(S[-1] / S[0]))
            if lorentzian:
                c = c / c[kinds.index("T")]            # rotation angles at timelike hinges: +1
            else:
                c = c / c[0]
            self.hinge_weights.append(c)
            self.hinge_kinds.append(kinds)
            hloc = -np.einsum("t,te,tf->ef", c, dA, dth)
            for a, (i, j) in enumerate(PAIRS):
                da = DID[tuple(v[j] - v[i])]
                for b, (k, l) in enumerate(PAIRS):
                    self.blocks.append((da, v[i], DID[tuple(v[l] - v[k])], v[k], hloc[a, b]))
            # linearised deficit angle of each triangle: d eps_t = -sum_sigma c_t d theta_t
            for t, (i, j) in enumerate(PAIRS):
                tri = sorted([tuple(v[q]) for q in range(5) if q not in (i, j)])
                P0 = np.array(tri[0])
                key = (tuple(np.array(tri[1]) - P0), tuple(np.array(tri[2]) - P0))
                tri_keys.setdefault(key, len(tri_keys))
                for e, (p, q) in enumerate(PAIRS):
                    self.deficit_terms.append(
                        (tri_keys[key], P0, DID[tuple(v[q] - v[p])], v[p], -c[t] * dth[t, e])
                    )
        self.n_triangle_types = len(tri_keys)
        self._pack()

    def _pack(self):
        self._da = np.array([b[0] for b in self.blocks])
        self._db = np.array([b[2] for b in self.blocks])
        self._dp = np.array([b[3] - b[1] for b in self.blocks], dtype=float)
        self._h = np.array([b[4] for b in self.blocks])
        self._tk = np.array([t[0] for t in self.deficit_terms])
        self._te = np.array([t[2] for t in self.deficit_terms])
        self._tp = np.array([t[3] - t[1] for t in self.deficit_terms], dtype=float)
        self._tv = np.array([t[4] for t in self.deficit_terms])

    def hessian(self, k):
        H = np.zeros((15, 15), dtype=complex)
        np.add.at(H, (self._da, self._db), self._h * np.exp(1j * (self._dp @ k)))
        return H

    def time_blocks(self, k_space):
        """A_tau(k_space), tau = -1, 0, +1, with H(k_t, k) = sum_tau A_tau exp(i k_t tau)."""
        out = {}
        ph = np.exp(1j * (self._dp[:, 1:] @ np.asarray(k_space, float)))
        for tau in (-1, 0, 1):
            m = self._dp[:, 0] == tau
            A = np.zeros((15, 15), dtype=complex)
            np.add.at(A, (self._da[m], self._db[m]), self._h[m] * ph[m])
            out[tau] = A
        return out

    def deficit_map(self, k):
        D = np.zeros((self.n_triangle_types, 15), dtype=complex)
        np.add.at(D, (self._tk, self._te), self._tv * np.exp(1j * (self._tp @ k)))
        return D

    def gauge_vectors(self, k):
        """Vertex translations xi(x) = xi exp(ik.x): delta s_d = 2 <d, xi> (exp(ik.d) - 1)."""
        g = np.zeros((15, 4), dtype=complex)
        for i, d in enumerate(DIRS):
            d = np.array(d, float)
            g[i, :] = 2.0 * (self.G @ d) * (np.exp(1j * (k @ d)) - 1.0)
        return g

    def reduced_hessian(self, k):
        g = self.gauge_vectors(k)
        Qg, _ = np.linalg.qr(g, mode="complete")
        Q = Qg[:, 4:]
        return Q.conj().T @ self.hessian(k) @ Q, Q


def metric_map(k):
    """15x10 map from a constant-amplitude metric perturbation h_mu_nu to delta s_d (edge midpoint)."""
    M = np.zeros((15, 10), dtype=complex)
    for i, d in enumerate(DIRS):
        d = np.array(d, float)
        ph = np.exp(0.5j * (k @ d))
        for a, (m, n) in enumerate(SYM):
            M[i, a] = d[m] * d[n] * (1 if m == n else 2) * ph
    return M


def continuum_gauge(G, k):
    C = np.zeros((10, 4), dtype=complex)
    for a, (m, n) in enumerate(SYM):
        for r in range(4):
            C[a, r] = 1j * (k[m] * G[n, r] + k[n] * G[m, r])
    return C


def count_negative(lat, k, rel_tol=1e-13):
    Hr, _ = lat.reduced_hessian(k)
    ev = np.linalg.eigvalsh(Hr)
    return int(np.sum(ev < -rel_tol * np.max(np.abs(ev))))


def propagating_modes(lat, k_space, n_grid: int = 1201, rel_tol=1e-13):
    """All real k_t in (0, pi] at which a non-gauge null vector appears, with multiplicities."""
    k_space = np.asarray(k_space, float)
    grid = np.linspace(1e-4, math.pi, n_grid)
    counts = [count_negative(lat, np.array([kt, *k_space]), rel_tol) for kt in grid]
    found = []
    for i in range(n_grid - 1):
        if counts[i] != counts[i + 1]:
            lo, hi, clo = grid[i], grid[i + 1], counts[i]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if count_negative(lat, np.array([mid, *k_space]), rel_tol) == clo:
                    lo = mid
                else:
                    hi = mid
            found.append((0.5 * (lo + hi), abs(counts[i + 1] - counts[i])))
    return found


def graviton_null_space(lat, k):
    """Null vectors of H(k) orthogonal to the 4 gauge vectors and the extra exact zero mode."""
    H = lat.hessian(k)
    w, V = np.linalg.eigh(0.5 * (H + H.conj().T))
    order = np.argsort(np.abs(w))
    return V[:, order], w[order]


def polarisation_content(lat, k, n_null: int):
    """
    For the n_null smallest-|eigenvalue| vectors at an on-shell k: least-squares metric content h,
    with continuum gauge directions projected out; returns the non-gauge metric parts.
    """
    V, w = graviton_null_space(lat, k)
    M = metric_map(k)
    C = continuum_gauge(lat.G, k)
    Qc, _ = np.linalg.qr(C)
    hs = []
    for i in range(n_null):
        hfit, *_ = np.linalg.lstsq(M, V[:, i], rcond=None)
        hs.append(hfit - Qc @ (Qc.conj().T @ hfit))
    return hs, w[:n_null]


# ---------------------------------------------------------------------------------------------
# Discrete-time (Lorentzian) Regge evolution of the linearised equations, per spatial Fourier mode
# ---------------------------------------------------------------------------------------------
I_HYPER = DID[(1, 1, 1, 1)]          # 4D hyperdiagonal: drops out of the quadratic action
I_BODY3 = DID[(0, 1, 1, 1)]          # spatial body diagonal: first enters one step later
Y_DIRS = [i for i, d in enumerate(DIRS) if d[0] == 1]
X_DIRS = [i for i, d in enumerate(DIRS) if d[0] == 0]
FIX_DIRS = [DID[d] for d in GAUGE_FIX_DIRS] + [I_HYPER]       # lapse/shift edges + hyperdiagonal
Y_FREE = [i for i in Y_DIRS if i not in FIX_DIRS]              # 3
X_FREE = [i for i in X_DIRS if i != I_BODY3]                   # 6


class ReggeTimeStepper:
    """
    Solves the linearised Lorentzian Regge equations E_n = A_-1 s_{n-1} + A_0 s_n + A_+1 s_{n+1} = 0
    (one equation per edge owned by time slice n) slice by slice for one spatial wavevector.
    Unknowns at step n: Y_n (slice-connecting edges, minus 4 gauge-fixed lapse/shift edges and the
    hyperdiagonal), X_{n+1} (spatial edges of the next slice, minus the body diagonal) and the lagged
    body diagonal X_n(0,1,1,1): 10 unknowns, 15 equations, of which 4 are Bianchi-redundant
    constraints and 1 (the hyperdiagonal equation) is identically zero.
    """

    def __init__(self, lat, k_space):
        self.A = lat.time_blocks(k_space)
        A = self.A
        self.K = np.hstack([A[0][:, Y_FREE], A[1][:, X_FREE], A[0][:, [I_BODY3]]])
        sv = np.linalg.svd(self.K, compute_uv=False)
        self.rank_ratio = float(sv[-1] / sv[0])
        self.K_pinv = np.linalg.pinv(self.K)   # full column rank -> unique least-squares solution

    def step(self, s_prev, s_cur_partial):
        """
        s_prev: complete s_{n-1}; s_cur_partial: s_n with X_n (except body diagonal) known and the
        gauge-fixed components zero. Returns complete s_n, partial s_{n+1}, relative residual of E_n.
        """
        A = self.A
        known = A[-1] @ s_prev + A[0] @ s_cur_partial
        x = -(self.K_pinv @ known)
        s_cur = s_cur_partial.copy()
        s_cur[Y_FREE] = x[:3]
        s_cur[I_BODY3] = x[9]
        s_next = np.zeros(15, dtype=complex)
        s_next[X_FREE] = x[3:9]
        res = np.linalg.norm(A[-1] @ s_prev + A[0] @ s_cur + A[1] @ s_next) / max(np.linalg.norm(known), 1e-300)
        return s_cur, s_next, float(res)


def gauge_adjusted_plane_waves(lat, k):
    """The two exact on-shell graviton null vectors of H(k), shifted by lattice gauge vectors and the
    hyperdiagonal so that the 4 lapse/shift edges and the hyperdiagonal vanish (stepper gauge)."""
    H = lat.hessian(k)
    g = lat.gauge_vectors(k)
    w, V = np.linalg.eigh(0.5 * (H + H.conj().T))
    N = V[:, np.argsort(np.abs(w))[:7]]
    e_h = np.zeros((15, 1), dtype=complex)
    e_h[I_HYPER, 0] = 1.0
    B = np.column_stack([g, e_h])
    Bq, _ = np.linalg.qr(B)
    R = N - Bq @ (Bq.conj().T @ N)
    U, S, _ = np.linalg.svd(R)
    out = []
    for j in range(2):
        v = U[:, j]
        coef = np.linalg.solve(B[FIX_DIRS, :], -v[FIX_DIRS])
        out.append(v + B @ coef)
    return out, float(S[2] / S[0])


# ---------------------------------------------------------------------------------------------
# Audits used by tests/module_15_lorentzian_regge_4d.py (Tests 56-58)
# ---------------------------------------------------------------------------------------------
def _extra_free_null_count(H, rel=1e-12):
    sv = np.linalg.svd(H, compute_uv=False)
    return int(np.sum(sv < rel * sv[0]))


def audit_lorentzian_regge_structure(alpha: float = 0.5, n_random: int = 20) -> dict:
    """Test 56: Lorentzian Schlafli hinge weights, gauge invariance, Bianchi identity, zero-mode content."""
    lat = LorentzianReggeLattice(alpha, lorentzian=True)
    rng = np.random.default_rng(56)
    herm = gauge = bianchi = hyper_col = deficit_gauge = 0.0
    null_counts = set()
    hyper_curv = []
    for _ in range(n_random):
        k = rng.uniform(-math.pi, math.pi, 4)
        H = lat.hessian(k)
        g = lat.gauge_vectors(k)
        D = lat.deficit_map(k)
        nH, ng = np.linalg.norm(H), np.linalg.norm(g)
        herm = max(herm, float(np.linalg.norm(H - H.conj().T) / nH))
        gauge = max(gauge, float(np.linalg.norm(H @ g) / (nH * ng)))
        bianchi = max(bianchi, float(np.linalg.norm(g.conj().T @ H) / (nH * ng)))
        hyper_col = max(hyper_col, float(np.linalg.norm(H[:, I_HYPER]) / nH))
        deficit_gauge = max(deficit_gauge, float(np.linalg.norm(D @ g) / (np.linalg.norm(D) * ng)))
        hyper_curv.append(float(np.linalg.norm(D[:, I_HYPER]) / np.linalg.norm(D)))
        null_counts.add(_extra_free_null_count(H))
    pattern = {}
    for c, kd in zip(lat.hinge_weights, lat.hinge_kinds):
        for ci, kname in zip(c, kd):
            key = f"{kname}:{int(round(ci)):+d}"
            pattern[key] = pattern.get(key, 0) + 1
    euc = LorentzianReggeLattice(alpha, lorentzian=False)
    return {
        "alpha": alpha,
        "n_simplices_per_hypercube": len(lat.hinge_weights),
        "n_edge_directions": 15,
        "n_triangle_types": lat.n_triangle_types,
        "max_lorentzian_schlafli_rel_residual": max(lat.schlafli_rel_residuals),
        "max_abs_hinge_weight_minus_1": float(max(np.max(np.abs(np.abs(c) - 1.0)) for c in lat.hinge_weights)),
        "all_timelike_hinge_weights_plus_1": bool(all(
            np.allclose(c[[i for i, kname in enumerate(kd) if kname == "T"]], 1.0, atol=1e-12)
            for c, kd in zip(lat.hinge_weights, lat.hinge_kinds))),
        "hinge_weight_pattern": pattern,
        "euclidean_control_max_abs_weight_minus_1": float(max(np.max(np.abs(c - 1.0)) for c in euc.hinge_weights)),
        "max_hermiticity_error": herm,
        "max_gauge_invariance_error_H_g": gauge,
        "max_bianchi_identity_error_gH_H": bianchi,
        "null_space_dimensions_seen": sorted(null_counts),
        "max_hyperdiagonal_hessian_column_rel": hyper_col,
        "min_hyperdiagonal_deficit_coupling_rel": float(min(hyper_curv)),
        "max_gauge_deficit_change_rel": deficit_gauge,
    }


def audit_graviton_dispersion(alphas=(0.5, 0.3), kmags=(0.025, 0.05, 0.1, 0.2)) -> dict:
    """Test 57: propagating modes of the linearised Lorentzian Regge equations over the full k_t zone."""
    dirs = {"z": [0, 0, 1], "xy": [1, 1, 0], "xyz": [1, 1, 1], "123": [1, 2, 3]}
    table = {}
    single_double_crossing = True
    worst_low_k = 0.0
    orders = []
    for alpha in alphas:
        lat = LorentzianReggeLattice(alpha)
        for name, dvec in dirs.items():
            kd = np.array(dvec, float) / np.linalg.norm(dvec)
            devs = []
            for km in kmags:
                modes = propagating_modes(lat, km * kd, n_grid=601)
                if len(modes) != 1 or modes[0][1] != 2:
                    single_double_crossing = False
                    devs.append(float("nan"))
                    continue
                devs.append(1.0 - modes[0][0] / (alpha * km))
            table[f"alpha={alpha},dir={name}"] = devs
            worst_low_k = max(worst_low_k, abs(devs[0]))
            orders.append(math.log(devs[-2] / devs[-1]) / math.log(kmags[-2] / kmags[-1]))
    # polarisations at |k| = 0.05 along z (alpha = 0.5)
    lat = LorentzianReggeLattice(0.5)
    km = 0.05
    kt = propagating_modes(lat, [0, 0, km], n_grid=601)[0][0]
    k = np.array([kt, 0, 0, km])
    vs, sep = gauge_adjusted_plane_waves(lat, k)
    M = metric_map(k)
    g = lat.gauge_vectors(k)
    e_h = np.zeros((15, 1), dtype=complex)
    e_h[I_HYPER, 0] = 1.0
    C = continuum_gauge(lat.G, k)
    Qc, _ = np.linalg.qr(C)
    ep = np.zeros(10, dtype=complex)
    ep[SYM.index((1, 1))], ep[SYM.index((2, 2))] = 1.0, -1.0
    ec = np.zeros(10, dtype=complex)
    ec[SYM.index((1, 2))] = 1.0
    TTq, _ = np.linalg.qr(np.column_stack([ep, ec]))
    tt_fracs, nonmetric = [], []
    for v in vs:
        X = np.column_stack([M, e_h, g])
        coef, *_ = np.linalg.lstsq(X, v, rcond=None)
        nonmetric.append(float(np.linalg.norm(v - X @ coef) / np.linalg.norm(v)))
        hp = coef[:10] - Qc @ (Qc.conj().T @ coef[:10])
        tt_fracs.append(float(np.linalg.norm(TTq.conj().T @ hp) / np.linalg.norm(hp)))
    # Euclidean control: no real propagating solutions
    euc = LorentzianReggeLattice(0.5, lorentzian=False)
    euc_modes = propagating_modes(euc, [0, 0, 0.1], n_grid=601)
    return {
        "kmags": list(kmags),
        "one_minus_v_over_c": table,
        "exactly_one_doubly_degenerate_mode_in_full_kt_zone": single_double_crossing,
        "max_abs_one_minus_v_at_smallest_k": worst_low_k,
        "convergence_orders_last_pair": orders,
        "graviton_polarisation_TT_fractions_k005_z": tt_fracs,
        "graviton_non_metric_residuals_k005_z": nonmetric,
        "graviton_vs_other_null_separation": sep,
        "euclidean_control_real_crossings": len(euc_modes),
    }


def audit_regge_time_evolution(alpha: float = 0.5) -> dict:
    """Test 58: slice-by-slice Lorentzian Regge evolution (plane waves, stability, wave packet)."""
    lat = LorentzianReggeLattice(alpha)
    # (a) exact plane wave, 2000 steps
    km = 0.2
    kt = propagating_modes(lat, [0, 0, km], n_grid=601)[0][0]
    vs, _ = gauge_adjusted_plane_waves(lat, np.array([kt, 0, 0, km]))
    st = ReggeTimeStepper(lat, [0, 0, km])
    pw_err = pw_res = 0.0
    for v in vs:
        s_prev = v.copy()
        s_cur = np.zeros(15, dtype=complex)
        s_cur[X_FREE] = (v * np.exp(1j * kt))[X_FREE]
        for n in range(1, 2001):
            s_c, s_next, res = st.step(s_prev, s_cur)
            pw_res = max(pw_res, res)
            pw_err = max(pw_err, float(np.linalg.norm(s_c - v * np.exp(1j * kt * n)) / np.linalg.norm(v)))
            s_prev, s_cur = s_c, s_next
    # (b) one-step transfer map: spectrum and boundedness
    max_lam = 0.0
    growth = 0.0
    unit_phases_ok = True
    for ks in ([0, 0, 0.2], [0.4, -0.3, 0.9], [math.pi, math.pi, math.pi], [0.0, 0.0, 0.01]):
        stp = ReggeTimeStepper(lat, ks)
        T = np.zeros((21, 21), dtype=complex)
        for j in range(21):
            zv = np.zeros(21, dtype=complex)
            zv[j] = 1.0
            sp = zv[:15].copy()
            sp[FIX_DIRS] = 0.0
            sc = np.zeros(15, dtype=complex)
            sc[X_FREE] = zv[15:]
            a, b, _ = stp.step(sp, sc)
            T[:15, j] = a
            T[15:, j] = b[X_FREE]
        lam = np.linalg.eigvals(T)
        max_lam = max(max_lam, float(np.max(np.abs(lam))))
        allowed = [0.0, -ks[0], -ks[1], -ks[2]] + [m[0] for m in propagating_modes(lat, ks, n_grid=601)]
        allowed += [-a for a in allowed]
        for l in lam[np.abs(np.abs(lam) - 1.0) < 1e-9]:
            ph = np.angle(l)
            if min(abs(np.angle(np.exp(1j * (ph - a)))) for a in allowed) > 1e-7:
                unit_phases_ok = False
        Tn = np.eye(21, dtype=complex)
        n1000 = None
        for n in range(1, 4001):
            Tn = T @ Tn
            if n == 1000:
                n1000 = np.linalg.norm(Tn, 2)
        growth = max(growth, float(np.linalg.norm(Tn, 2) / n1000))
    # (c) cross-polarised wave packet in real space (Nz = 1024, k0 = 0.2)
    Nz, k0, sk, nsteps = 1024, 0.2, 0.025, 1200
    ixx, iyy, ixy = SYM.index((1, 1)), SYM.index((2, 2)), SYM.index((1, 2))
    zz = np.arange(Nz)
    hxy = np.zeros((nsteps + 1, Nz), dtype=complex)
    pk_res = 0.0
    kts = []
    for m in range(1, Nz // 2):
        kz = 2 * math.pi * m / Nz
        if abs(kz - k0) > 4 * sk:
            continue
        ktm = propagating_modes(lat, [0, 0, kz], n_grid=241)[0][0]
        kts.append((kz, ktm))
        k4 = np.array([ktm, 0, 0, kz])
        vv, _ = gauge_adjusted_plane_waves(lat, k4)
        Mp = np.linalg.pinv(metric_map(k4))
        h1, h2 = Mp @ vv[0], Mp @ vv[1]
        cc = np.array([h2[ixx] - h2[iyy], -(h1[ixx] - h1[iyy])])
        v = cc[0] * vv[0] + cc[1] * vv[1]
        v = v / (Mp @ v)[ixy]
        amp = math.exp(-0.5 * ((kz - k0) / sk) ** 2)
        stp = ReggeTimeStepper(lat, [0, 0, kz])
        ph = np.exp(1j * kz * zz)
        s_prev = v.copy()
        s_cur = np.zeros(15, dtype=complex)
        s_cur[X_FREE] = (v * np.exp(1j * ktm))[X_FREE]
        hxy[0] += amp * (Mp @ v)[ixy] * ph
        for n in range(1, nsteps + 1):
            s_c, s_next, res = stp.step(s_prev, s_cur)
            pk_res = max(pk_res, res)
            hxy[n] += amp * (Mp @ s_c)[ixy] * ph
            s_prev, s_cur = s_c, s_next
    dens = np.abs(hxy) ** 2
    ang = 2 * math.pi * zz / Nz
    cent = np.array([math.atan2((d * np.sin(ang)).sum(), (d * np.cos(ang)).sum()) for d in dens])
    cent = np.unwrap(cent) * Nz / (2 * math.pi)
    tphys = alpha * np.arange(nsteps + 1)
    vg = abs(float(np.polyfit(tphys[100:], cent[100:], 1)[0]))
    kk, tt = np.array(kts).T
    vg_disp = float(np.interp(k0, kk, np.gradient(tt, kk)) / alpha)
    return {
        "plane_wave_2000_steps_max_rel_error": pw_err,
        "plane_wave_max_equation_residual": pw_res,
        "transfer_map_max_abs_eigenvalue": max_lam,
        "transfer_map_unit_modulus_only_graviton_or_residual_gauge": unit_phases_ok,
        "transfer_map_norm_growth_T4000_over_T1000_linear_bound_4": growth,
        "packet_k0": k0,
        "packet_steps": nsteps,
        "packet_max_equation_residual": pk_res,
        "packet_group_velocity_over_c": vg,
        "dispersion_group_velocity_over_c": vg_disp,
        "packet_norm_ratio_end_over_start": float(np.linalg.norm(hxy[-1]) / np.linalg.norm(hxy[0])),
    }
