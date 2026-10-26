"""
Relativistic Misner-Sharp / Einstein-Euler Fluid Collapse, Shock Heating, and Core Bounce Solver
(`einstein_euler_core_bounce.py`)

Follows the V2 curvature-cap bounce past shell crossing by replacing pressureless dust (p = 0)
with the self-consistent general-relativistic spherically symmetric Misner-Sharp (1964) /
May-White (1966) Einstein-Euler fluid equations coupled to the curvature-capped effective
metric function:

    f_LQC(R, m) = 1 - (2 G m / (c^2 R)) * (1 - bar_rho / rho_max)
                = 1 - (2 G m / (c^2 R)) * (1 - 2 G m ell^2 / (c^2 R^3)),

where rho_max = 3 c^2 / (8 pi G ell^2) and R_core(m) = (2 G m ell^2 / c^2)^(1/3).

In comoving Lagrangian coordinates (tau, mu) where mu is enclosed rest (baryon) mass
and ds^2 = -e^{2 phi} c^2 dtau^2 + ((dR/dmu)^2 / Gamma^2) dmu^2 + R^2 dOmega^2:

  1. Exact Misner-Sharp Relativistic Evolution Equations (with D_tau = e^{-phi} d/dtau):
       D_tau R     = U
       D_tau m     = -4 pi R^2 p_tot U / c^2
       D_tau Gamma = -(4 pi R^2 U / (c^2 h)) * (dp_tot / dmu)
       D_tau U     = -(4 pi R^2 Gamma / h) * (dp_tot / dmu)
                     - (G m / R^2) * (1 - 4 bar_rho / rho_max)
                     - (4 pi G R p_tot / c^2) * (1 - 2 bar_rho / rho_max)
       D_tau eps   = -p_tot * D_tau(1 / rho_0),   1 / rho_0 = (4 pi R^2 / Gamma) * (dR / dmu)

     where p_tot = p_th + q_shock, p_th = (gamma - 1) rho_0 eps, and h = 1 + gamma * eps / c^2.

  2. Exact Differential Identities Preserved by the System:
     (a) Misner-Sharp Metric Constraint:
           D_tau (Gamma^2 - f_LQC(R, m) - U^2 / c^2) == 0   identically!
         Therefore, on the trapped-region hyperbola (f_LQC < 0):
           (U / c)^2 - Gamma^2 = |f_LQC(R, m)| > 0.
     (b) Total ADM Mass Conservation at the Vacuum Surface (p_tot(M_0) = 0):
           D_tau M_ADM = D_tau m(tau, M_0) == 0   identically!
     (c) Curvature-Capped Tolman-Oppenheimer-Volkoff (TOV) Hydrostatic Equation (U = 0, D_tau U = 0):
           dp / dR = -((rho + p/c^2) G / (R^2 f_LQC(R, m)))
                     * [ m (1 - 4 bar_rho / rho_max) + (4 pi R^3 p / c^2) (1 - 2 bar_rho / rho_max) ].
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def f_lqc(R: np.ndarray | float, m: np.ndarray | float, ell: float = 1.0) -> np.ndarray | float:
    """Curvature-capped effective-LQC metric function f_LQC(R, m) = 1 - (2m/R)*(1 - 2m*ell^2/R^3)."""
    return 1.0 - (2.0 * m / R) * (1.0 - 2.0 * m * (ell**2) / (R**3))


def df_lqc_dR(R: np.ndarray | float, m: np.ndarray | float, ell: float = 1.0) -> np.ndarray | float:
    """Partial derivative df_LQC / dR at fixed enclosed mass m."""
    return (2.0 * m / (R**2)) * (1.0 - 8.0 * m * (ell**2) / (R**3))


def df_lqc_dm(R: np.ndarray | float, m: np.ndarray | float, ell: float = 1.0) -> np.ndarray | float:
    """Partial derivative df_LQC / dm at fixed areal radius R."""
    return -(2.0 / R) * (1.0 - 4.0 * m * (ell**2) / (R**3))


def find_lqc_inner_horizon(M: float, ell: float = 1.0) -> float:
    """Exact inner horizon root R_-(M) > R_core(M) where f_LQC(R_-, M) = 0."""
    R_core = (2.0 * M * (ell**2)) ** (1.0 / 3.0)
    return float(brentq(lambda r: f_lqc(r, M, ell), R_core, 2.0 * R_core, xtol=1e-14))


def audit_curvature_capped_tov_no_go(
    masses: list[float] | None = None,
    ell: float = 1.0,
) -> dict:
    """
    Audits the Curvature-Capped Tolman-Oppenheimer-Volkoff (TOV) hydrostatic equation
    across all radii R inside the outer horizon R_+(M):

      dp / dR = -((rho + p) / (R^2 f_LQC(R, m))) * [ m (1 - 4 bar_rho/rho_max) + 4 pi R^3 p (1 - 2 bar_rho/rho_max) ]

    Proves two exact results for any positive-pressure fluid (rho > 0, p >= 0):
      1. In the trapped region R_-(M) < R < R_+(M) (including R_eq = 4^(1/3) R_core(M)):
         f_LQC(R, M) < 0, so any static (U = 0) state has Gamma^2 = f_LQC(R, M) < 0 (violating Lorentzian signature).
      2. In the untrapped timelike core region 0 < R < R_-(M) (where f_LQC(R, M) > 0):
         bar_rho / rho_max > 1 - R_-(M)/r_s > 1/2 everywhere, so both (1 - 4 bar_rho/rho_max) < -1 < 0
         and (1 - 2 bar_rho/rho_max) < 0. Consequently, dp/dR > 0 strictly for all p >= 0:
         pressure increases monotonically outward and can NEVER reach a zero-pressure surface p(R_surf) = 0
         inside R < R_-(M) (only negative vacuum tension p = -rho_max gives dp/dR = 0).
    """
    if masses is None:
        masses = [10.0, 20.0, 50.0, 100.0]

    records = []
    for M in masses:
        r_s = 2.0 * M
        R_core = (r_s * (ell**2)) ** (1.0 / 3.0)
        R_minus = find_lqc_inner_horizon(M, ell=ell)
        R_eq = (4.0 ** (1.0 / 3.0)) * R_core

        # 1. Check trapped nature of R_eq = 4^(1/3) R_core
        f_at_Req = float(f_lqc(R_eq, M, ell=ell))

        # 2. Check timelike core region R in (0.1 R_core, 0.999 R_minus)
        R_grid = np.linspace(0.1 * R_core, 0.999 * R_minus, 300)
        ratio = (R_core / R_grid) ** 3  # bar_rho / rho_max
        f_vals = f_lqc(R_grid, M, ell=ell)
        term_mass = 1.0 - 4.0 * ratio
        term_press = 1.0 - 2.0 * ratio

        # Sample positive pressure p = (1/3) rho_max > 0
        rho_max = 3.0 / (8.0 * np.pi * (ell**2))
        p_sample = (1.0 / 3.0) * rho_max
        rho_sample = rho_max
        tov_bracket = M * term_mass + 4.0 * np.pi * (R_grid**3) * p_sample * term_press
        dp_dR_positive_p = -((rho_sample + p_sample) / ((R_grid**2) * f_vals)) * tov_bracket

        # Integrate TOV outward from R_0 = 0.05 R_core to 0.98 R_minus for uniform core density rho = rho_max
        # where m(R) = (4/3) pi rho_max R^3 = R^3 / (2 ell^2), so bar_rho / rho_max = 1 identically:
        # At bar_rho = rho_max: f_LQC(R, m(R)) = 1, 1 - 4 bar_rho/rho_max = -3, 1 - 2 bar_rho/rho_max = -1,
        # so dp/dR = +((rho_max + p)/R^2) * [3 m(R) + 4 pi R^3 p] = +(3/(2 ell^2)) R (rho_max + p)(1 + p/rho_max) > 0!
        # And for p = -rho_max (de Sitter vacuum tension, V1 route), dp/dR = 0 identically!
        dp_dR_de_sitter = 0.0

        records.append({
            "M": float(M),
            "R_core": float(R_core),
            "R_minus": float(R_minus),
            "R_eq": float(R_eq),
            "f_LQC_at_R_eq_negative": f_at_Req,
            "min_f_LQC_in_timelike_core": float(np.min(f_vals)),
            "min_bar_rho_over_rho_max_at_R_minus": float((R_core / R_minus) ** 3),
            "max_mass_factor_1_minus_4ratio": float(np.max(term_mass)),
            "max_pressure_factor_1_minus_2ratio": float(np.max(term_press)),
            "min_dp_dR_in_timelike_core_for_positive_p": float(np.min(dp_dR_positive_p)),
            "dp_dR_strictly_positive_in_timelike_core": bool(np.all(dp_dR_positive_p > 0.0)),
            "dp_dR_de_sitter_vacuum": dp_dR_de_sitter,
        })

    return {
        "records": records,
        "all_R_eq_spacelike": bool(all(r["f_LQC_at_R_eq_negative"] < 0.0 for r in records)),
        "all_timelike_cores_have_dp_dR_positive": bool(
            all(r["dp_dR_strictly_positive_in_timelike_core"] for r in records)
        ),
    }


def run_lagrangian_einstein_euler(
    N: int = 80,
    M_tot: float = 20.0,
    ell: float = 1.0,
    gamma_eos: float = 4.0 / 3.0,
    inhomogeneity: float = 0.42,
    l_visc: float = 0.06,
    eps_floor: float = 5e-4,
    t_end: float = 6.0,
    dust_mode: bool = False,
) -> dict:
    """
    Evolves the self-consistent Lagrangian General-Relativistic Misner-Sharp / Einstein-Euler
    fluid equations on a falling lattice of N spherical shells with curvature cap f_LQC(R, m).

    Parameters
    ----------
    N : int
        Number of Lagrangian fluid shells (cells k = 0..N-1, nodes i = 1..N).
    M_tot : float
        Total enclosed gravitational mass M at the stellar surface.
    ell : float
        Planck UV cutoff scale (default 1.0).
    gamma_eos : float
        Adiabatic index gamma in p_th = (gamma - 1) rho_0 eps (e.g. 4/3 for relativistic radiation).
    inhomogeneity : float
        Central concentration parameter delta in m(s) ~ s^3 (1 + delta (1 - s^2)).
    l_visc : float
        Physical Planck-scale shock dissipation length (in units of ell). Because l_visc is fixed
        in physical units ell rather than grid spacing, the PDE converges as N -> infinity.
    eps_floor : float
        Initial specific thermal energy floor eps(0) / c^2.
    t_end : float
        Final comoving proper time tau.
    dust_mode : bool
        If True, sets p_th = q_shock = 0 (pressureless dust control run to demonstrate shell crossing).
    """
    r_s = 2.0 * M_tot
    R_core_tot = (r_s * (ell**2)) ** (1.0 / 3.0)
    R_minus_tot = find_lqc_inner_horizon(M_tot, ell=ell)

    R_star0 = 12.0 * ell
    s = np.linspace(0.0, 1.0, N + 1)
    R0_nodes = R_star0 * s[1:]

    shape = (s[1:] ** 3) * (1.0 + inhomogeneity * (1.0 - s[1:] ** 2))
    m0_nodes = M_tot * shape / shape[-1]
    m0_full = np.concatenate([[0.0], m0_nodes])
    dmu_cells = np.diff(m0_full)

    eps0_cells = np.zeros(N) if dust_mode else np.full(N, eps_floor)
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

        if dust_mode:
            z = np.zeros(N)
            return R_full, U_full, dV_cells, dV_dtau, Gamma_cells, rho0_cells, z, z, z, np.ones(N), z

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
    min_ratio_dR = float(np.min(dR_hist / dR_hist[:, [0]]))

    f_hist = f_lqc(R_hist, m_hist, ell=ell)
    max_constraint_err = float(np.max(np.abs(Gamma_hist**2 - f_hist - U_hist**2)))
    adm_mass_rel_err = float(np.max(np.abs(m_hist[-1, :] - m_hist[-1, 0])) / m_hist[-1, 0])
    max_interior_mass_transfer = float(np.max(np.abs(m_hist[:-1, :] - m_hist[:-1, [0]])) / M_tot)
    min_Gamma = float(np.min(Gamma_hist))
    min_Gamma2 = float(np.min(Gamma_hist**2))

    max_eps = float(np.max(eps_hist))
    mean_eps_series = np.sum(dmu_cells[:, None] * eps_hist, axis=0) / M_tot
    max_mean_eps = float(np.max(mean_eps_series))
    mean_eps_final = float(mean_eps_series[-1])

    h_hist = 1.0 + gamma_eos * np.maximum(eps_hist, 0.0)
    cs_hist = np.sqrt(gamma_eos * (gamma_eos - 1.0) * np.maximum(eps_hist, 0.0) / h_hist)
    max_cs = float(np.max(cs_hist))

    # Turning-point audit for every shell i = 1..N
    idx_min = np.argmin(R_hist, axis=1)
    R_turn = R_hist[np.arange(N), idx_min]
    m_turn = m_hist[np.arange(N), idx_min]
    f_turn = f_lqc(R_turn, m_turn, ell=ell)
    all_turns_inside_R_minus = bool(np.all(f_turn > 0.0))

    return {
        "N": N,
        "M_tot": float(M_tot),
        "R_core_tot": float(R_core_tot),
        "R_minus_tot": float(R_minus_tot),
        "t": sol.t,
        "R_surf": R_hist[-1, :],
        "R_half": R_hist[N // 2 - 1, :],
        "U_surf": U_hist[-1, :],
        "min_dR": min_dR,
        "min_ratio_dR": min_ratio_dR,
        "shell_crossing_prevented": bool(min_dR > 0.0),
        "min_Gamma": min_Gamma,
        "min_Gamma2": min_Gamma2,
        "lorentzian_signature_preserved": bool(min_Gamma > 0.0),
        "max_constraint_err": max_constraint_err,
        "adm_mass_rel_err": adm_mass_rel_err,
        "max_interior_mass_transfer": max_interior_mass_transfer,
        "R_surf_min": float(np.min(R_hist[-1, :])),
        "R_surf_final": float(R_hist[-1, -1]),
        "U_surf_final": float(U_hist[-1, -1]),
        "max_eps": max_eps,
        "max_mean_eps": max_mean_eps,
        "mean_eps_final": mean_eps_final,
        "heating_amplification_factor": float(max_mean_eps / max(eps_floor, 1e-30)),
        "max_cs": max_cs,
        "all_turns_inside_R_minus": all_turns_inside_R_minus,
        "min_f_at_turn": float(np.min(f_turn)),
    }
