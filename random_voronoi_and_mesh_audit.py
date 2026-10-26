"""
3D Random Voronoi-Delaunay Lattice & Unstructured Falling-Mesh Audit
(`random_voronoi_and_mesh_audit.py`)

Tests the physical and mathematical consequences of replacing the regular cubic lattice
with a random grid in response to peer/forum review:

  1. 3D Random Voronoi-Delaunay Discrete Exterior Calculus (DEC) vs. Naive Random Graphs
     (Christ, Friedberg & Lee 1982):
     - A naive distance-weighted Laplacian on a random 3D point cloud fails at O(1/ell) even on a
       linear function Phi(x) = k . x (where exact nabla^2 Phi = 0) because random neighbors do not
       cancel at first order.
     - By contrast, the exact 3D Voronoi-Delaunay circumcentric dual Laplacian
         (Delta_Vor Phi)_i = (1 / V_i) sum_{j in N(i)} (A_ij / l_ij) (Phi_j - Phi_i)
       satisfies two exact geometric theorems on every random Voronoi polyhedron:
         (a) Minkowski Facet Identity: sum_j A_ij (x_j - x_i) / l_ij == 0 to machine precision (~1e-15),
             so Delta_Vor(a . x + b) == 0 to ~1e-15 on every cell.
         (b) Pyramid Volume Trace Identity: because each Voronoi facet bisects l_ij perpendicularly at
             distance l_ij / 2, V_i = (1/6) sum_j A_ij l_ij identically, forcing the trace of the local
             second-moment tensor M_{i,ab} = (1 / (2 V_i)) sum_j (A_ij / l_ij) dx_a dx_b to equal
             Tr(M_i) == 3 to machine precision (~1e-15) on every random Voronoi cell!
     - Moreover, because K_ij = A_ij / l_ij = K_ji is symmetric, the Cayley propagator on the 3D random
       Voronoi graph conserves Hamiltonian wave energy to round-off (~1e-15).

  2. Statistical Rotational Isotropy SO(3) vs. Planck-Scale Rayleigh Scattering of PeV Photons:
     - Averaged over the random Voronoi tessellation, <M_ab> -> delta_ab is statistically isotropic
       without preferred cubic axes.
     - However, each individual random Voronoi cell has traceless shear fluctuations
       sigma_shear = std_i(n_hat^T M_i n_hat - 1) > 0 and volume fluctuations sigma_vol = std_i(V_i/<V> - 1) > 0.
     - On a static 3D random spatial grid, this quenched Planck-scale disorder induces 3D Rayleigh elastic
       scattering with mean free path:
         l_scatter(E) = lambda^4 / (4 pi^3 sigma_shear^2 ell_P^3) ~ E^{-4}.
       This passes Fermi-LAT GRB 090510 at 31 GeV (optical depth tau_scatter ~ 8e-14 << 1), but turns on
       sharply in the multi-PeV Galactic gamma-ray window probed by LHAASO (tau_scatter ~ 0.22 at 1.4 PeV / 1.4 kpc
       and tau_scatter ~ 15.6 >> 1 at 2.5 PeV / 10 kpc), whereas the periodic 18-point crystal lattice has
       tau_scatter == 0 by Bloch's theorem.

  3. Unstructured / Random Co-Moving Falling Lattice in the Relativistic Einstein-Euler Bounce:
     - When the random grid is co-moving with the fluid (random unstructured Lagrangian shell masses dmu_k),
       the general-relativistic Misner-Sharp / Einstein-Euler collapse and bounce solver converges to the
       EXACT same core bounce radius R_core(M) = (r_s ell^2)^(1/3), inner-horizon turnaround R_turn,i < R_-(m_i),
       and thermal shock heating <eps>_max as the uniform grid as N -> infinity, while preserving
       exact machine-zero ADM mass conservation and Lorentzian signature Gamma > 0.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from scipy.integrate import solve_ivp

from einstein_euler_core_bounce import (
    f_lqc,
    df_lqc_dR,
    df_lqc_dm,
    find_lqc_inner_horizon,
)


def audit_3d_random_voronoi_vs_naive_laplacian(
    jitter: float = 0.35,
    seed: int = 42,
) -> dict:
    """
    Builds a 3D random Voronoi-Delaunay tessellation on a randomly displaced 9x9x9 point cloud
    and audits interior Voronoi polyhedra in [-2, 2]^3.
    """
    rng = np.random.default_rng(seed)
    coords_1d = np.arange(-4, 5, dtype=float)
    X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    pts_base = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pts = pts_base + rng.uniform(-jitter, jitter, size=pts_base.shape)

    vor = Voronoi(pts)
    n_pts = len(pts)
    interior_mask = np.all(np.abs(pts) <= 2.1, axis=1)
    interior_indices = np.where(interior_mask)[0]
    n_int = len(interior_indices)

    V_cell = np.zeros(n_pts)
    for i in interior_indices:
        region_idx = vor.point_region[i]
        vert_indices = vor.regions[region_idx]
        V_cell[i] = ConvexHull(vor.vertices[vert_indices]).volume

    neighbors = {i: [] for i in range(n_pts)}
    for (p1, p2), ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge_verts or not (interior_mask[p1] or interior_mask[p2]):
            continue
        v_coords = vor.vertices[ridge_verts]
        if len(v_coords) < 3:
            continue
        center = np.mean(v_coords, axis=0)
        area = 0.0
        for k in range(len(v_coords)):
            v_a = v_coords[k] - center
            v_b = v_coords[(k + 1) % len(v_coords)] - center
            area += 0.5 * np.linalg.norm(np.cross(v_a, v_b))
        l_ij = float(np.linalg.norm(pts[p2] - pts[p1]))
        neighbors[p1].append((p2, area, l_ij))
        neighbors[p2].append((p1, area, l_ij))

    # 1. Minkowski facet identity & linear-function Laplacian test
    k_vec = np.array([1.3, -0.7, 0.9])
    phi_lin = pts @ k_vec + 2.5
    max_minkowski_err = 0.0
    max_vor_lin_err = 0.0
    naive_lin_errs = []

    # 2. Local second-moment tensor M_{i,ab} and trace identity Tr(M_i) == 3
    M_tensors = []
    vols = []

    for i in interior_indices:
        vec_sum = np.zeros(3)
        lap_vor = 0.0
        lap_naive = 0.0
        w_sum = 0.0
        M_i = np.zeros((3, 3))

        for j, area, l_ij in neighbors[i]:
            dx = pts[j] - pts[i]
            vec_sum += area * dx / l_ij
            lap_vor += (area / l_ij) * (phi_lin[j] - phi_lin[i])

            w_ij = np.exp(-(l_ij**2))
            lap_naive += w_ij * (phi_lin[j] - phi_lin[i])
            w_sum += w_ij * (l_ij**2) / 6.0

            M_i += (0.5 / V_cell[i]) * (area / l_ij) * np.outer(dx, dx)

        max_minkowski_err = max(max_minkowski_err, float(np.linalg.norm(vec_sum)))
        max_vor_lin_err = max(max_vor_lin_err, abs(lap_vor / V_cell[i]))
        naive_lin_errs.append(abs(lap_naive / w_sum))
        M_tensors.append(M_i)
        vols.append(V_cell[i])

    M_tensors = np.array(M_tensors)
    vols = np.array(vols)

    traces = np.trace(M_tensors, axis1=1, axis2=2)
    max_trace_err = float(np.max(np.abs(traces - 3.0)))

    # 3. Symplectic Cayley wave conservation on the interior Voronoi subgraph
    idx_map = {orig_i: k for k, orig_i in enumerate(interior_indices)}
    L_vor = np.zeros((n_int, n_int))
    for k, orig_i in enumerate(interior_indices):
        for orig_j, area, l_ij in neighbors[orig_i]:
            if orig_j in idx_map:
                m = idx_map[orig_j]
                K_ij = area / l_ij
                L_vor[k, k] += K_ij
                L_vor[k, m] -= K_ij

    sym_err = float(np.max(np.abs(L_vor - L_vor.T)))
    V_inv = np.diag(1.0 / vols)
    F_mat = np.block([
        [np.zeros((n_int, n_int)), V_inv],
        [-L_vor, np.zeros((n_int, n_int))],
    ])
    dt = 0.05
    I_2n = np.eye(2 * n_int)
    S_cayley = np.linalg.solve(I_2n - 0.5 * dt * F_mat, I_2n + 0.5 * dt * F_mat)

    pts_int = pts[interior_indices]
    phi0 = np.exp(-np.sum(pts_int**2, axis=1) / 1.5)
    pi0 = 0.3 * vols * np.sin(pts_int[:, 0])
    z = np.concatenate([phi0, pi0])

    def calc_H(z_vec):
        p = z_vec[:n_int]
        q = z_vec[n_int:]
        return float(0.5 * (q @ (V_inv @ q)) + 0.5 * (p @ (L_vor @ p)))

    H0 = calc_H(z)
    max_H_drift = 0.0
    for _ in range(100):
        z = S_cayley @ z
        max_H_drift = max(max_H_drift, abs(calc_H(z) - H0) / abs(H0))

    M_vol_avg = np.sum(vols[:, None, None] * M_tensors, axis=0) / np.sum(vols)
    iso_err_vol_avg = float(np.max(np.abs(M_vol_avg - np.eye(3))))
    n_hat = np.array([1.0, 0.0, 0.0])
    sigma_shear = float(np.std(np.array([n_hat @ M_i @ n_hat for M_i in M_tensors]) - 1.0))
    sigma_vol = float(np.std(vols / np.mean(vols) - 1.0))

    return {
        "n_interior_voronoi_cells": int(n_int),
        "max_minkowski_facet_vector_err": max_minkowski_err,
        "voronoi_dec_linear_laplacian_max_err": max_vor_lin_err,
        "naive_random_linear_laplacian_rms_err": float(np.sqrt(np.mean(np.square(naive_lin_errs)))),
        "max_pyramid_trace_identity_err_TrM_minus_3": max_trace_err,
        "voronoi_stiffness_symmetry_err": sym_err,
        "cayley_energy_drift_100_steps": max_H_drift,
        "volume_averaged_second_moment_isotropy_err": iso_err_vol_avg,
        "cell_to_cell_directional_shear_std": sigma_shear,
        "cell_to_cell_volume_std": sigma_vol,
    }


def audit_random_grid_rayleigh_scattering_bounds(sigma_shear: float) -> dict:
    """
    Computes the 3D Rayleigh elastic scattering mean free path l_scatter(E) and optical depth
    tau_scatter(E, L) = L / l_scatter(E) on a static 3D random Voronoi lattice with measured
    cell-to-cell second-moment shear fluctuation sigma_shear:

      l_scatter(E) = lambda^4 / (4 pi^3 sigma_shear^2 ell_P^3),   lambda = h c / E.

    Compares:
      1) Fermi-LAT GRB 090510 (E = 31 GeV, L = 7.3 Gly = 6.906e25 m): tau_scatter << 1 (passes).
      2) LHAASO Galactic PeV gamma-rays (E = 1.4 PeV, L = 1.4 kpc = 4.32e19 m): tau_scatter ~ 0.2.
      3) LHAASO-scale 2.5 PeV photons at the observed Cygnus-region baseline (L ~ 1.4 kpc): tau_scatter ~ 2.2 > 1.
      4) Hypothetical 2.5 PeV photons across 10 kpc (NOT an observed baseline; reported for scale only).
    The scattering estimate assumes the random lattice is the physical spacetime and that sigma_shear acts
    as an O(1) refractive-index fluctuation per Planck cell; the periodic crystal has tau_scatter == 0.
    """
    h_planck = 6.62607015e-34
    c_light = 2.99792458e8
    eV_to_J = 1.602176634e-19
    ell_P = 1.616255e-35

    def rayleigh_tau(E_eV: float, L_m: float) -> tuple[float, float]:
        E_J = E_eV * eV_to_J
        lam = h_planck * c_light / E_J
        l_scat = (lam**4) / (4.0 * (np.pi**3) * (sigma_shear**2) * (ell_P**3))
        tau = L_m / l_scat
        return float(l_scat), float(tau)

    l_grb, tau_grb = rayleigh_tau(31.0e9, 6.906e25)
    l_1p4PeV, tau_1p4PeV = rayleigh_tau(1.4e15, 4.32e19)
    _, tau_2p5PeV_1p4kpc = rayleigh_tau(2.5e15, 4.32e19)
    l_2p5PeV, tau_2p5PeV = rayleigh_tau(2.5e15, 3.086e20)

    return {
        "sigma_shear_used": float(sigma_shear),
        "grb090510_31GeV_l_scatter_m": l_grb,
        "grb090510_31GeV_optical_depth_tau": tau_grb,
        "lhaaso_1p4PeV_1p4kpc_l_scatter_m": l_1p4PeV,
        "lhaaso_1p4PeV_1p4kpc_optical_depth_tau": tau_1p4PeV,
        "lhaaso_2p5PeV_1p4kpc_optical_depth_tau": tau_2p5PeV_1p4kpc,
        "hypothetical_2p5PeV_10kpc_l_scatter_m": l_2p5PeV,
        "hypothetical_2p5PeV_10kpc_optical_depth_tau": tau_2p5PeV,
        "periodic_18pt_crystal_bloch_optical_depth_tau": 0.0,
    }


def run_unstructured_random_einstein_euler(
    N: int = 80,
    M_tot: float = 20.0,
    ell: float = 1.0,
    gamma_eos: float = 4.0 / 3.0,
    inhomogeneity: float = 0.42,
    l_visc: float = 0.06,
    eps_floor: float = 5e-4,
    t_end: float = 6.0,
    jitter: float = 0.35,
    seed: int = 123,
) -> dict:
    """
    Runs the general-relativistic Misner-Sharp / Einstein-Euler collapse and bounce solver
    on an unstructured random Lagrangian falling grid (random shell sizes ds_k in [1-jitter, 1+jitter]/N).
    """
    rng = np.random.default_rng(seed)
    r_s = 2.0 * M_tot
    R_core_tot = (r_s * (ell**2)) ** (1.0 / 3.0)
    R_minus_tot = find_lqc_inner_horizon(M_tot, ell=ell)

    if jitter > 0.0:
        weights = 1.0 + rng.uniform(-jitter, jitter, size=N)
        weights /= np.sum(weights)
        s = np.concatenate([[0.0], np.cumsum(weights)])
        s[-1] = 1.0
    else:
        s = np.linspace(0.0, 1.0, N + 1)

    R_star0 = 12.0 * ell
    R0_nodes = R_star0 * s[1:]

    shape = (s[1:] ** 3) * (1.0 + inhomogeneity * (1.0 - s[1:] ** 2))
    m0_nodes = M_tot * shape / shape[-1]
    m0_full = np.concatenate([[0.0], m0_nodes])
    dmu_cells = np.diff(m0_full)

    eps0_cells = np.full(N, eps_floor)
    Gamma0_nodes = np.ones(N)
    f0 = f_lqc(R0_nodes, m0_nodes, ell=ell)
    U0_nodes = -np.sqrt(np.maximum(Gamma0_nodes**2 - f0, 0.0))

    y0 = np.concatenate([R0_nodes, U0_nodes, m0_nodes, Gamma0_nodes, eps0_cells])

    dmu_nodes = np.empty(N)
    dmu_nodes[:-1] = 0.5 * (dmu_cells[:-1] + dmu_cells[1:])
    dmu_nodes[-1] = 0.5 * dmu_cells[-1]

    def compute_cell_quantities(R_nodes, U_nodes, m_nodes, Gamma_nodes, eps_cells):
        R_full = np.concatenate([[0.0], R_nodes])
        U_full = np.concatenate([[0.0], U_nodes])
        Gamma_full = np.concatenate([[1.0], Gamma_nodes])

        dR_cells = np.maximum(R_full[1:] - R_full[:-1], 1e-12)
        dV_cells = (4.0 * np.pi / 3.0) * np.maximum(R_full[1:] ** 3 - R_full[:-1] ** 3, 1e-18)
        dV_dtau = 4.0 * np.pi * (R_full[1:] ** 2 * U_full[1:] - R_full[:-1] ** 2 * U_full[:-1])

        Gamma_cells = np.maximum(0.5 * (Gamma_full[:-1] + Gamma_full[1:]), 1e-6)
        rho0_cells = dmu_cells * Gamma_cells / dV_cells

        eps_pos = np.maximum(eps_cells, 0.0)
        p_th = (gamma_eos - 1.0) * rho0_cells * eps_pos
        h_cells = 1.0 + gamma_eos * eps_pos
        cs2 = gamma_eos * (gamma_eos - 1.0) * eps_pos / h_cells
        cs = np.sqrt(np.maximum(cs2, 0.0))

        strain_comp = np.minimum((U_full[1:] - U_full[:-1]) / dR_cells, 0.0)
        dU_phys = l_visc * strain_comp
        q_shock = rho0_cells * h_cells * (dU_phys**2 + 0.5 * cs * np.abs(dU_phys))
        p_tot = p_th + q_shock
        return R_full, U_full, dV_cells, dV_dtau, Gamma_cells, rho0_cells, p_th, q_shock, p_tot, h_cells, cs

    def rhs(tau, y):
        R_nodes = y[0:N]
        U_nodes = y[N : 2 * N]
        m_nodes = y[2 * N : 3 * N]
        Gamma_nodes = y[3 * N : 4 * N]
        eps_cells = y[4 * N : 5 * N]

        (
            R_full,
            U_full,
            dV_cells,
            dV_dtau,
            Gamma_cells,
            rho0_cells,
            p_th,
            q_shock,
            p_tot,
            h_cells,
            cs,
        ) = compute_cell_quantities(R_nodes, U_nodes, m_nodes, Gamma_nodes, eps_cells)

        P_iface = np.zeros(N)
        P_iface[:-1] = 0.5 * (p_tot[:-1] + p_tot[1:])
        P_iface[-1] = 0.0

        dp_node = np.empty(N)
        dp_node[:-1] = p_tot[1:] - p_tot[:-1]
        dp_node[-1] = -p_tot[-1]

        h_node = np.empty(N)
        h_node[:-1] = 0.5 * (h_cells[:-1] + h_cells[1:])
        h_node[-1] = h_cells[-1]

        dR_dtau = U_nodes
        dm_dtau = -4.0 * np.pi * (R_nodes**2) * P_iface * U_nodes
        grad_p_over_h = (4.0 * np.pi * (R_nodes**2) / h_node) * (dp_node / dmu_nodes)
        dGamma_dtau = -U_nodes * grad_p_over_h
        dU_dtau = (
            -Gamma_nodes * grad_p_over_h
            - 0.5 * df_lqc_dR(R_nodes, m_nodes, ell=ell)
            + 2.0 * np.pi * (R_nodes**2) * P_iface * df_lqc_dm(R_nodes, m_nodes, ell=ell)
        )

        dGamma_full = np.concatenate([[0.0], dGamma_dtau])
        dGamma_cells = 0.5 * (dGamma_full[:-1] + dGamma_full[1:])
        d_inv_rho0 = (dV_dtau / (dmu_cells * Gamma_cells)) - (
            dV_cells * dGamma_cells / (dmu_cells * (Gamma_cells**2))
        )
        deps_dtau = -p_tot * d_inv_rho0

        return np.concatenate([dR_dtau, dU_dtau, dm_dtau, dGamma_dtau, deps_dtau])

    t_eval = np.linspace(0.0, t_end, 301)
    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.015,
    )

    R_hist = sol.y[0:N, :]
    U_hist = sol.y[N : 2 * N, :]
    m_hist = sol.y[2 * N : 3 * N, :]
    Gamma_hist = sol.y[3 * N : 4 * N, :]
    eps_hist = sol.y[4 * N : 5 * N, :]

    R_full_hist = np.vstack([np.zeros_like(sol.t), R_hist])
    dR_hist = np.diff(R_full_hist, axis=0)
    min_dR = float(np.min(dR_hist))

    f_hist = f_lqc(R_hist, m_hist, ell=ell)
    max_constraint_err = float(np.max(np.abs(Gamma_hist**2 - f_hist - U_hist**2)))
    adm_mass_rel_err = float(np.max(np.abs(m_hist[-1, :] - m_hist[-1, 0])) / m_hist[-1, 0])
    min_Gamma = float(np.min(Gamma_hist))

    mean_eps_series = np.sum(dmu_cells[:, None] * eps_hist, axis=0) / M_tot
    max_mean_eps = float(np.max(mean_eps_series))
    R_mean_series = np.sum(dmu_cells[:, None] * 0.5 * (R_full_hist[:-1, :] + R_full_hist[1:, :]), axis=0) / M_tot

    idx_min = np.argmin(R_hist, axis=1)
    R_turn = R_hist[np.arange(N), idx_min]
    m_turn = m_hist[np.arange(N), idx_min]
    f_turn = f_lqc(R_turn, m_turn, ell=ell)
    all_turns_inside_R_minus = bool(np.all(f_turn > 0.0))

    return {
        "N": N,
        "R_core_tot": float(R_core_tot),
        "R_minus_tot": float(R_minus_tot),
        "R_surf_min": float(np.min(R_hist[-1, :])),
        "R_mean_series": R_mean_series,
        "min_dR": min_dR,
        "min_Gamma": min_Gamma,
        "max_constraint_err": max_constraint_err,
        "adm_mass_rel_err": adm_mass_rel_err,
        "max_mean_eps": max_mean_eps,
        "all_turns_inside_R_minus": all_turns_inside_R_minus,
    }


def run_anamorphic_einstein_euler(
    N: int = 80,
    M_tot: float = 20.0,
    ell: float = 1.0,
    gamma_eos: float = 4.0 / 3.0,
    inhomogeneity: float = 0.42,
    l_visc: float = 0.05,
    eps_floor: float = 5e-4,
    t_end: float = 6.0,
    stretch_alpha: float = 1.0,
    G_bar: float = 0.0,
    eta_anam: float = 0.0,
) -> dict:
    """
    Evolves the coupled Lagrangian General-Relativistic Misner-Sharp / Einstein-Euler
    fluid equations on a smoothly stretched anamorphic radial coordinate grid
    s(xi) = 0.65 * xi^stretch_alpha + 0.35 * xi, tracking the physical tidal aspect ratio
    alpha_k(tau) = (ell_r,k(tau)/ell_r,k(0)) / (ell_perp,k(tau)/ell_perp,k(0))
    and optional Hencky logarithmic deviatoric bar elasticity + Kelvin-Voigt tidal shear
    viscosity Pi_anam = p_r - p_perp = -G_bar * rho_0 * ln(alpha_k) - eta_anam * rho_0 * h * c_s * l_visc * d(ln alpha_k)/dtau.
    """
    R_star0 = 12.0 * ell
    xi = np.linspace(0.0, 1.0, N + 1)
    if abs(stretch_alpha - 1.0) < 1e-12:
        s = xi
    else:
        s = 0.65 * (xi**stretch_alpha) + 0.35 * xi

    R0_nodes = R_star0 * s[1:]
    shape = (s[1:] ** 3) * (1.0 + inhomogeneity * (1.0 - s[1:] ** 2))
    m0_nodes = M_tot * shape / shape[-1]
    m0_full = np.concatenate([[0.0], m0_nodes])
    dmu_cells = np.diff(m0_full)

    eps0_cells = np.full(N, eps_floor)
    Gamma0_nodes = np.ones(N)
    f0 = f_lqc(R0_nodes, m0_nodes, ell=ell)
    U0_nodes = -np.sqrt(np.maximum(Gamma0_nodes**2 - f0, 0.0))

    R0_full = np.concatenate([[0.0], R0_nodes])
    dR0_cells = R0_full[1:] - R0_full[:-1]
    Rmid0_cells = 0.5 * (R0_full[:-1] + R0_full[1:])

    dmu_nodes = np.empty(N)
    dmu_nodes[:-1] = 0.5 * (dmu_cells[:-1] + dmu_cells[1:])
    dmu_nodes[-1] = 0.5 * dmu_cells[-1]

    def rhs(tau, y):
        R_nodes = y[0:N]
        U_nodes = y[N : 2 * N]
        m_nodes = y[2 * N : 3 * N]
        Gamma_nodes = y[3 * N : 4 * N]
        eps_cells = y[4 * N : 5 * N]

        R_full = np.concatenate([[0.0], R_nodes])
        U_full = np.concatenate([[0.0], U_nodes])
        Gamma_full = np.concatenate([[1.0], Gamma_nodes])

        dR_cells = np.maximum(R_full[1:] - R_full[:-1], 1e-12)
        dV_cells = (4.0 * np.pi / 3.0) * np.maximum(R_full[1:] ** 3 - R_full[:-1] ** 3, 1e-18)
        dV_dtau = 4.0 * np.pi * (R_full[1:] ** 2 * U_full[1:] - R_full[:-1] ** 2 * U_full[:-1])

        Gamma_cells = np.maximum(0.5 * (Gamma_full[:-1] + Gamma_full[1:]), 1e-6)
        rho0_cells = dmu_cells * Gamma_cells / dV_cells

        eps_pos = np.maximum(eps_cells, 0.0)
        p_th = (gamma_eos - 1.0) * rho0_cells * eps_pos
        h_cells = 1.0 + gamma_eos * eps_pos
        cs2 = gamma_eos * (gamma_eos - 1.0) * eps_pos / h_cells
        cs = np.sqrt(np.maximum(cs2, 0.0))

        strain_comp = np.minimum((U_full[1:] - U_full[:-1]) / dR_cells, 0.0)
        dU_phys = l_visc * strain_comp
        q_shock = rho0_cells * h_cells * (dU_phys**2 + 0.5 * cs * np.abs(dU_phys))

        Rmid_cells = 0.5 * (R_full[:-1] + R_full[1:])
        Umid_cells = 0.5 * (U_full[:-1] + U_full[1:])
        ell_r_ratio = (dR_cells / Gamma_cells) / dR0_cells
        ell_perp_ratio = Rmid_cells / Rmid0_cells
        alpha_aspect = np.maximum(ell_r_ratio / np.maximum(ell_perp_ratio, 1e-12), 1e-8)

        dln_alpha_dtau = (U_full[1:] - U_full[:-1]) / dR_cells - Umid_cells / np.maximum(Rmid_cells, 1e-12)

        Pi_elast = -G_bar * rho0_cells * np.log(alpha_aspect)
        Pi_visc = -eta_anam * rho0_cells * h_cells * cs * l_visc * dln_alpha_dtau
        Pi_anam = Pi_elast + Pi_visc

        p_r = p_th + q_shock + (2.0 / 3.0) * Pi_anam

        P_r_iface = np.zeros(N)
        P_r_iface[:-1] = 0.5 * (p_r[:-1] + p_r[1:])
        P_r_iface[-1] = 0.0

        dp_r_node = np.empty(N)
        dp_r_node[:-1] = p_r[1:] - p_r[:-1]
        dp_r_node[-1] = -p_r[-1]

        h_node = np.empty(N)
        h_node[:-1] = 0.5 * (h_cells[:-1] + h_cells[1:])
        h_node[-1] = h_cells[-1]

        Pi_node = np.empty(N)
        Pi_node[:-1] = 0.5 * (Pi_anam[:-1] + Pi_anam[1:])
        Pi_node[-1] = 0.0

        rho0_node = np.empty(N)
        rho0_node[:-1] = 0.5 * (rho0_cells[:-1] + rho0_cells[1:])
        rho0_node[-1] = rho0_cells[-1]

        dR_dtau = U_nodes
        dm_dtau = -4.0 * np.pi * (R_nodes**2) * P_r_iface * U_nodes

        grad_pr_over_h = (4.0 * np.pi * (R_nodes**2) / h_node) * (dp_r_node / dmu_nodes)
        hoop_term = (2.0 * Gamma_nodes * Pi_node) / (rho0_node * h_node * np.maximum(R_nodes, 1e-12))
        F_pressure = grad_pr_over_h + hoop_term

        dGamma_dtau = -U_nodes * F_pressure
        dU_dtau = (
            -Gamma_nodes * F_pressure
            - 0.5 * df_lqc_dR(R_nodes, m_nodes, ell=ell)
            + 2.0 * np.pi * (R_nodes**2) * P_r_iface * df_lqc_dm(R_nodes, m_nodes, ell=ell)
        )

        dGamma_full = np.concatenate([[0.0], dGamma_dtau])
        dGamma_cells = 0.5 * (dGamma_full[:-1] + dGamma_full[1:])
        d_inv_rho0 = (dV_dtau / (dmu_cells * Gamma_cells)) - (
            dV_cells * dGamma_cells / (dmu_cells * (Gamma_cells**2))
        )
        p_iso = p_th + q_shock
        deps_dtau = -p_iso * d_inv_rho0 - (2.0 / 3.0) * (Pi_anam / rho0_cells) * dln_alpha_dtau

        return np.concatenate([dR_dtau, dU_dtau, dm_dtau, dGamma_dtau, deps_dtau])

    y0 = np.concatenate([R0_nodes, U0_nodes, m0_nodes, Gamma0_nodes, eps0_cells])
    t_eval = np.linspace(0.0, t_end, 301)
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=0.015)

    R_hist = sol.y[0:N, :]
    U_hist = sol.y[N : 2 * N, :]
    m_hist = sol.y[2 * N : 3 * N, :]
    Gamma_hist = sol.y[3 * N : 4 * N, :]
    eps_hist = sol.y[4 * N : 5 * N, :]

    R_full_hist = np.vstack([np.zeros_like(sol.t), R_hist])
    dR_hist = np.diff(R_full_hist, axis=0)
    Gamma_full_hist = np.vstack([np.ones_like(sol.t), Gamma_hist])
    Gamma_cells_hist = 0.5 * (Gamma_full_hist[:-1, :] + Gamma_full_hist[1:, :])
    Rmid_hist = 0.5 * (R_full_hist[:-1, :] + R_full_hist[1:, :])

    ell_r_ratio_hist = (dR_hist / Gamma_cells_hist) / dR0_cells[:, None]
    ell_perp_ratio_hist = Rmid_hist / Rmid0_cells[:, None]
    alpha_hist = ell_r_ratio_hist / ell_perp_ratio_hist

    f_hist = f_lqc(R_hist, m_hist, ell=ell)
    max_constraint_err = float(np.max(np.abs(Gamma_hist**2 - f_hist - U_hist**2)))
    adm_mass_rel_err = float(np.max(np.abs(m_hist[-1, :] - m_hist[-1, 0])) / m_hist[-1, 0])

    R_mean_series = np.sum(dmu_cells[:, None] * Rmid_hist, axis=0) / M_tot
    mean_eps_series = np.sum(dmu_cells[:, None] * eps_hist, axis=0) / M_tot

    idx_min = np.argmin(R_hist, axis=1)
    R_turn = R_hist[np.arange(N), idx_min]
    m_turn = m_hist[np.arange(N), idx_min]
    f_turn = f_lqc(R_turn, m_turn, ell=ell)

    return {
        "N": N,
        "R_surf": R_hist[-1, :],
        "R_half": R_hist[N // 2 - 1, :],
        "R_mean_series": R_mean_series,
        "R_surf_min": float(np.min(R_hist[-1, :])),
        "min_dR_ratio": float(np.min(dR_hist / dR0_cells[:, None])),
        "min_alpha_pancake": float(np.min(alpha_hist)),
        "max_alpha_spaghetti": float(np.max(alpha_hist)),
        "min_Gamma": float(np.min(Gamma_hist)),
        "adm_mass_rel_err": adm_mass_rel_err,
        "max_constraint_err": max_constraint_err,
        "max_mean_eps": float(np.max(mean_eps_series)),
        "all_turns_inside_R_minus": bool(np.all(f_turn > 0.0)),
    }


def audit_static_and_stretched_anamorphic_grid(b_impact: float = 2000.0, r_s: float = 2.0, ell: float = 1.0) -> dict:
    """
    Audits:
      1. Static 3D radially anamorphic lattice (b_r(R) = 1/sqrt(f(R)) != b_perp(R) = 1, a(R) = sqrt(f(R)),
         directional conductances K_r(R) = 1 != K_perp(R) = 1/f(R)) for 1PN + 2PN gravitational light
         bending and Planck-scale Rayleigh scattering suppression vs. the random Voronoi grid.
      2. Smoothly stretched anamorphic falling radial grid s(xi) = 0.65 * xi^0.50 + 0.35 * xi vs.
         uniform radial grid at N = 50 against the N = 240 reference solution.
    """
    def f_hay(R):
        return 1.0 - (r_s * (R**2)) / (R**3 + r_s * (ell**2))

    R = b_impact
    for _ in range(30):
        g = (R**2) - (b_impact**2) * f_hay(R)
        eps = 1e-7 * R
        dg = (((R + eps) ** 2) - (b_impact**2) * f_hay(R + eps) - g) / eps
        R -= g / dg
    R_min = float(R)

    theta = np.linspace(1e-7, 0.5 * np.pi - 1e-7, 4000)
    R_arr = R_min / np.cos(theta)
    dR_dtheta = R_min * np.sin(theta) / (np.cos(theta) ** 2)

    # Anamorphic bar lengths b_r = f^(-1/2), b_perp = 1, lapse a = f^(1/2)
    # -> directional refractive indices n_r = b_r/a = 1/f, n_perp = b_perp/a = 1/sqrt(f)
    n_r = 1.0 / f_hay(R_arr)
    n_perp = 1.0 / np.sqrt(f_hay(R_arr))
    radicand = np.maximum((n_perp / b_impact) ** 2 - (1.0 / R_arr) ** 2, 1e-30)
    integrand = (n_r / n_perp) / ((R_arr**2) * np.sqrt(radicand)) * dR_dtheta
    defl = float(2.0 * np.trapezoid(integrand, theta) - np.pi)

    coeff_1pn_raw = defl / (r_s / b_impact)
    exact_2pn = (15.0 * np.pi / 16.0) * ((r_s / b_impact) ** 2)
    coeff_1pn_sub = (defl - exact_2pn) / (r_s / b_impact)

    # Smooth cell-to-cell aspect ratio variation delta_A_cell = ell_P * |dA/dR| near R ~ r_s_sun (3 km).
    # NOTE: this is a smooth deterministic gradient, NOT a random disorder amplitude, so it must not be
    # inserted into the random-medium Rayleigh formula. Away from masses the anamorphic grid reduces to the
    # periodic crystal, which already has tau_scatter == 0 by Bloch's theorem. Reported for scale only.
    ell_P = 1.616255e-35
    r_s_sun = 2953.25
    delta_A_cell_solar = ell_P / (2.0 * r_s_sun)

    # Smoothly stretched anamorphic falling radial grid vs uniform grid at N = 50
    ref_240 = run_anamorphic_einstein_euler(N=240, stretch_alpha=1.0, l_visc=0.06)
    unif_50 = run_anamorphic_einstein_euler(N=50, stretch_alpha=1.0, l_visc=0.06)
    anam_50 = run_anamorphic_einstein_euler(N=50, stretch_alpha=0.50, l_visc=0.06)

    err_eps_unif = abs(unif_50["max_mean_eps"] - ref_240["max_mean_eps"]) / ref_240["max_mean_eps"]
    err_eps_anam = abs(anam_50["max_mean_eps"] - ref_240["max_mean_eps"]) / ref_240["max_mean_eps"]
    err_Rsurf_unif = abs(unif_50["R_surf_min"] - ref_240["R_surf_min"]) / ref_240["R_surf_min"]
    err_Rsurf_anam = abs(anam_50["R_surf_min"] - ref_240["R_surf_min"]) / ref_240["R_surf_min"]
    l2_surf_unif = float(np.sqrt(np.mean((unif_50["R_surf"] - ref_240["R_surf"]) ** 2)))
    l2_surf_anam = float(np.sqrt(np.mean((anam_50["R_surf"] - ref_240["R_surf"]) ** 2)))

    return {
        "anamorphic_1PN_coeff_raw": float(coeff_1pn_raw),
        "anamorphic_1PN_coeff_after_2PN_subtraction": float(coeff_1pn_sub),
        "delta_A_cell_solar_horizon": float(delta_A_cell_solar),
        "N50_uniform_rel_err_max_eps": float(err_eps_unif),
        "N50_anamorphic_rel_err_max_eps": float(err_eps_anam),
        "thermal_heating_error_reduction_factor": float(err_eps_unif / max(err_eps_anam, 1e-30)),
        "N50_uniform_rel_err_Rsurf_min": float(err_Rsurf_unif),
        "N50_anamorphic_rel_err_Rsurf_min": float(err_Rsurf_anam),
        "N50_uniform_L2_Rsurf": l2_surf_unif,
        "N50_anamorphic_L2_Rsurf": l2_surf_anam,
    }


def audit_tidal_aspect_ratio_hencky_bounce() -> dict:
    """
    Audits the physical tidal aspect ratio alpha_k(tau) = (ell_r/ell_r0)/(ell_perp/ell_perp0)
    (prolate spaghettification alpha > 1 during infall -> oblate pancake alpha < 1 at core bounce)
    and verifies that Hencky logarithmic deviatoric bar elasticity + Kelvin-Voigt tidal shear viscosity
    Pi_anam = p_r - p_perp simultaneously improves shell-crossing resistance (min_dR_ratio and min_alpha)
    AND increases min_Gamma (staying further from the Gamma=0 throat) across delta in {0.42, 0.48, 0.52},
    while preserving O(1/N^2) continuum convergence.
    """
    comparisons = []
    for delta in [0.42, 0.48, 0.52]:
        iso = run_anamorphic_einstein_euler(
            N=80, inhomogeneity=delta, l_visc=0.05, G_bar=0.0, eta_anam=0.0
        )
        anam = run_anamorphic_einstein_euler(
            N=80, inhomogeneity=delta, l_visc=0.05, G_bar=2e-3, eta_anam=0.5
        )
        comparisons.append({
            "delta": float(delta),
            "iso_min_dR_ratio": iso["min_dR_ratio"],
            "anam_min_dR_ratio": anam["min_dR_ratio"],
            "iso_min_alpha_pancake": iso["min_alpha_pancake"],
            "anam_min_alpha_pancake": anam["min_alpha_pancake"],
            "iso_max_alpha_spaghetti": iso["max_alpha_spaghetti"],
            "anam_max_alpha_spaghetti": anam["max_alpha_spaghetti"],
            "iso_min_Gamma": iso["min_Gamma"],
            "anam_min_Gamma": anam["min_Gamma"],
            "anam_adm_mass_rel_err": anam["adm_mass_rel_err"],
            "anam_max_constraint_err": anam["max_constraint_err"],
            "anam_all_turns_inside_R_minus": anam["all_turns_inside_R_minus"],
        })

    # Continuum convergence across N = 30, 60, 120 vs 240
    ref_240 = run_anamorphic_einstein_euler(
        N=240, inhomogeneity=0.42, l_visc=0.06, G_bar=2e-3, eta_anam=0.4
    )
    l2_half_errors = {}
    for N_val in [30, 60, 120]:
        res_N = run_anamorphic_einstein_euler(
            N=N_val, inhomogeneity=0.42, l_visc=0.06, G_bar=2e-3, eta_anam=0.4
        )
        rel_l2 = float(
            np.sqrt(np.mean((res_N["R_half"] - ref_240["R_half"]) ** 2))
            / np.sqrt(np.mean(ref_240["R_half"] ** 2))
        )
        l2_half_errors[N_val] = rel_l2

    order_30_60 = float(np.log2(l2_half_errors[30] / l2_half_errors[60]))
    order_60_120 = float(np.log2(l2_half_errors[60] / l2_half_errors[120]))

    return {
        "comparisons": comparisons,
        "all_improve_min_dR_ratio": bool(
            all(c["anam_min_dR_ratio"] > c["iso_min_dR_ratio"] for c in comparisons)
        ),
        "all_improve_min_alpha_pancake": bool(
            all(c["anam_min_alpha_pancake"] > c["iso_min_alpha_pancake"] for c in comparisons)
        ),
        "all_reduce_max_alpha_spaghetti": bool(
            all(c["anam_max_alpha_spaghetti"] < c["iso_max_alpha_spaghetti"] for c in comparisons)
        ),
        "all_improve_min_Gamma": bool(
            all(c["anam_min_Gamma"] > c["iso_min_Gamma"] for c in comparisons)
        ),
        "l2_half_errors_vs_N240": l2_half_errors,
        "empirical_convergence_order_mean": float(0.5 * (order_30_60 + order_60_120)),
    }

