"""
collapse_falling_lattice.py
===========================
Spherically symmetric gravitational collapse on a FALLING LATTICE: the grid is the space.

Setting (units G = c = ell = 1, so rho_max = 3 / (8 pi) and R_core(m) = (2 m)^(1/3)):
  * The radial lattice consists of nodes (spherical shells) labelled by their initial radius R0.
    Cell k lies between nodes k and k+1 and carries a fixed Misner-Sharp mass increment dm_k
    (dm_k = 0 for empty "vacuum" cells).
  * All nodes start falling from rest at infinity (marginally bound). In Einstein's equations this is the
    Lemaitre-Tolman-Bondi (LTB) dust solution written in Painleve-Gullstrand form:
        ds^2 = -dtau^2 + (dR - U dtau)^2 + R^2 dOmega^2,   U = dR/dtau = -sqrt(2 m / R),
    so the spatial slices are exactly flat, every node moves with the river velocity J = U (now
    DETERMINED by the enclosed mass, not prescribed), and the proper volume of cell k is exactly
    (4 pi / 3)(R_{k+1}^3 - R_k^3). The lattice cells are physical free-falling volumes whose proper size
    changes with time (Version A in the discussion: the grid carries the geometry).
  * Each node obeys its own ODE (no pressure, no shell crossing): d^2R/dtau^2 = -m / R^2  (GR).

Saturation rules tested inside the dynamics:
  'gr'          : plain GR (no rule).
  'local_cap'   : the rule acts only on cells with matter (cell energy density <= rho_max). Whatever the
                  matter does, an EMPTY node keeps its enclosed Misner-Sharp mass (dm/dtau = -4 pi R^2 p U = 0
                  where p = 0) and keeps following Schwarzschild free fall, so it reaches R -> 0 where the
                  Kretschmann scalar 48 m^2 / R^6 diverges, while its energy density is zero and the rule never
                  fires. (Matter cannot overtake it from inside unless the matter bounces.)
  'curvature_cap' (LQC-type, = V2): the rule acts on the mean enclosed density rho_bar = 3 m / (4 pi R^3),
                  which in vacuum is a curvature invariant (48 m^2 / R^6 = (8 pi rho_bar / 3)^2 * 12). Each node
                  obeys the effective equation
                      (dR/dtau)^2 = (2 m / R) (1 - rho_bar / rho_max),
                  i.e.  d^2R/dtau^2 = -(m / R^2) (1 - 4 R_b^3 / R^3),  R_b = (3 m / (4 pi rho_max))^(1/3),
                  with exact solution R^3 = R_b^3 + (9/2) m (tau - tau_b)^2. This is the effective-LQC form used
                  for homogeneous (Oppenheimer-Snyder) collapse by Kelly, Santacruz & Wilson-Ewing; applying it
                  shell by shell is the decoupled-shell approximation (it breaks down at shell crossing, where
                  their full treatment produces a shock wave).

Pressure alternative (homogeneous region, see homogeneous_collapse()): enforcing the cap with a causal
(c_s <= c), NEC-respecting pressure cannot stop collapse once it is underway: rho is non-decreasing while
H < 0 and H^2 = 8 pi rho / 3 never vanishes for a flat region. Only rho + p -> 0 at the cap (conversion into
vacuum energy, p = -rho: V1-type) holds rho fixed.
"""

import numpy as np
from scipy.integrate import solve_ivp

RHO_MAX = 3.0 / (8.0 * np.pi)  # with ell = 1


def r_core(m):
    return (3.0 * m / (4.0 * np.pi * RHO_MAX)) ** (1.0 / 3.0)  # = (2 m)^(1/3)


def build_lattice(M=100.0, R_ball_over_rs=2.0, R_out_over_rs=3.0, n_matter=240, n_vac=60, profile="parabolic"):
    """Nodes at initial radii R0 (node 0 is the centre). Returns R0 (n+1,), enclosed mass m at each node,
    dm per cell, and a mask of matter cells."""
    r_s = 2.0 * M
    R_ball = R_ball_over_rs * r_s
    R_out = R_out_over_rs * r_s
    R0_matter = np.linspace(0.0, R_ball, n_matter + 1)
    R0_vac = np.linspace(R_ball, R_out, n_vac + 1)[1:]
    R0 = np.concatenate([R0_matter, R0_vac])
    if profile == "homogeneous":
        w = np.ones(n_matter)
    else:  # smooth, centrally concentrated: rho0 ~ 1 - (r/R_ball)^2
        rc = 0.5 * (R0_matter[1:] + R0_matter[:-1])
        w = 1.0 - (rc / R_ball) ** 2
    vol = (4.0 / 3.0) * np.pi * (R0_matter[1:] ** 3 - R0_matter[:-1] ** 3)
    dm_matter = w * vol
    dm_matter *= M / dm_matter.sum()
    dm = np.concatenate([dm_matter, np.zeros(n_vac)])
    m_node = np.concatenate([[0.0], np.cumsum(dm)])
    matter_cell = np.concatenate([np.ones(n_matter, bool), np.zeros(n_vac, bool)])
    return R0, m_node, dm, matter_cell


def evolve_nodes(R0, m_node, mode, t_eval, R_floor=1e-3):
    """Integrate every node's radial free-fall ODE numerically (RK45, tight tolerances).
    Returns R[t, node] (np.nan after a node reaches R_floor in GR) and the numerical-vs-exact error."""
    n = R0.size
    R = np.full((t_eval.size, n), np.nan)
    R[:, 0] = 0.0
    max_err = 0.0
    bounce_R = np.full(n, np.nan)
    floor_time = np.full(n, np.nan)
    for i in range(1, n):
        m = m_node[i]
        Rb = r_core(m) if mode == "curvature_cap" else 0.0

        def rhs(t, y):
            r, v = y
            return [v, -(m / r**2) * (1.0 - 4.0 * Rb**3 / r**3)]

        v0 = -np.sqrt((2.0 * m / R0[i]) * (1.0 - Rb**3 / R0[i] ** 3))

        def hit_floor(t, y):
            return y[0] - R_floor
        hit_floor.terminal = True

        def turn(t, y):
            return y[1]
        turn.direction = 1.0

        sol = solve_ivp(rhs, (0.0, t_eval[-1]), [R0[i], v0], t_eval=t_eval, rtol=1e-10, atol=1e-10,
                        events=[hit_floor, turn], method="DOP853")
        if sol.t_events[0].size:
            floor_time[i] = sol.t_events[0][0]
        if sol.t_events[1].size:
            bounce_R[i] = sol.y_events[1][0][0]
        k = sol.y.shape[1]
        R[:k, i] = sol.y[0]
        exact = exact_radius(R0[i], m, t_eval[:k], mode)
        ok = np.isfinite(exact) & (exact > 10.0 * R_floor)
        if np.any(ok):
            max_err = max(max_err, float(np.max(np.abs(sol.y[0][ok] - exact[ok]) / exact[ok])))
    return R, max_err, {"bounce_R": bounce_R, "floor_time": floor_time}


def exact_radius(R0, m, t, mode):
    if mode == "curvature_cap":
        Rb3 = r_core(m) ** 3
        tb = np.sqrt((R0**3 - Rb3) / (4.5 * m))
        return (Rb3 + 4.5 * m * (t - tb) ** 2) ** (1.0 / 3.0)
    base = R0**1.5 - 1.5 * np.sqrt(2.0 * m) * t
    return np.where(base > 0, np.abs(base) ** (2.0 / 3.0), np.nan)


def cell_diagnostics(R, m_node, dm, matter_cell):
    """Local cell density, mean enclosed density at nodes, vacuum Kretschmann, shell-crossing flag."""
    vol = (4.0 / 3.0) * np.pi * (R[:, 1:] ** 3 - R[:, :-1] ** 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_cell = np.where(vol > 0, dm[None, :] / vol, np.inf)
        rho_bar = 3.0 * m_node[None, 1:] / (4.0 * np.pi * R[:, 1:] ** 3)
        K_vac = 48.0 * m_node[None, 1:] ** 2 / R[:, 1:] ** 6
    crossed = np.any(vol <= 0, axis=1)
    return rho_cell, rho_bar, K_vac, crossed


def homogeneous_collapse(rule, rho0=0.01 * RHO_MAX, rho_t=0.5 * RHO_MAX, t_max=50.0, k_curv=0.0):
    """Homogeneous collapsing region (FRW interior, H < 0), with the density cap enforced by:
       'stiff_pressure' : p = 0 below rho_t, p = rho - rho_t above (c_s = c, NEC holds)
       'vacuum'         : at rho = rho_max matter converts to vacuum energy, p = -rho (V1-type)
       'lqc'            : effective H^2 = (8 pi / 3) rho (1 - rho / rho_max) (V2-type)
    Returns max density reached / rho_max and whether rho stayed finite."""
    def pressure(rho, converted):
        if rule == "stiff_pressure":
            return max(rho - rho_t, 0.0)
        if rule == "vacuum" and converted:
            return -rho
        return 0.0

    a = 1.0
    rho = rho0
    converted = False
    t = 0.0
    sign = -1.0
    max_rho = rho
    dt = 1e-3
    a_min = a
    while t < t_max:
        corr = (1.0 - rho / RHO_MAX) if rule == "lqc" else 1.0
        H2 = (8.0 * np.pi / 3.0) * rho * corr - k_curv / a**2
        if H2 <= 0.0:
            sign = 1.0  # turning point (bounce)
            H2 = 0.0
        H = sign * np.sqrt(H2)
        p = pressure(rho, converted)
        drho = -3.0 * H * (rho + p)
        dt_eff = min(dt, 1e-3 / (abs(H) + 1e-12), 0.05 * rho / (abs(drho) + 1e-300))
        rho_new = rho + drho * dt_eff
        if rule == "vacuum" and rho_new >= RHO_MAX:
            rho_new = RHO_MAX
            converted = True
        if rule == "lqc" and rho_new >= RHO_MAX:
            rho_new = RHO_MAX
        a *= np.exp(H * dt_eff)
        rho = rho_new
        t += dt_eff
        max_rho = max(max_rho, rho)
        a_min = min(a_min, a)
        if not np.isfinite(rho) or rho > 1e6 * RHO_MAX:
            return {"max_rho_over_rho_max": float(max_rho / RHO_MAX), "finite": False, "time": t,
                    "bounced": sign > 0, "a_min": a_min}
    return {"max_rho_over_rho_max": float(max_rho / RHO_MAX), "finite": True, "time": t,
            "bounced": sign > 0, "a_min": a_min}


def run_collapse_study(M=100.0, profile="parabolic", n_matter=240, n_vac=60, n_t=1500):
    R0, m_node, dm, matter_cell = build_lattice(M=M, profile=profile, n_matter=n_matter, n_vac=n_vac)
    r_s = 2.0 * M
    i_surface = n_matter
    i_vac_out = R0.size - 1
    t_sing_center = (2.0 / 3.0) * R0[1] ** 1.5 / np.sqrt(2.0 * m_node[1])
    t_sing_edge = (2.0 / 3.0) * R0[-1] ** 1.5 / np.sqrt(2.0 * m_node[-1])
    out = {"M": M, "r_s": r_s, "R_core_of_M": r_core(M), "profile": profile, "n_cells": int(dm.size)}

    # ---------- GR ----------
    t_gr = np.linspace(0.0, 0.999 * t_sing_edge, n_t)
    R_gr, err_gr, info_gr = evolve_nodes(R0, m_node, "gr", np.append(t_gr, 1.01 * t_sing_edge))
    R_gr = R_gr[:-1]
    rho_cell, rho_bar, K_vac, crossed = cell_diagnostics(R_gr, m_node, dm, matter_cell)
    rho_matter = np.where(matter_cell[None, :], rho_cell, np.nan)
    first_cap = np.argmax(np.nanmax(rho_matter, axis=1) >= RHO_MAX)
    trapped = (2.0 * m_node[None, 1:] / R_gr[:, 1:]) > 1.0
    surface_trapped_idx = np.argmax(trapped[:, i_surface - 1])
    first_trapped_idx = int(np.argmax(np.any(trapped, axis=1)))
    # empty node just outside the matter
    vac_node = i_surface + 1
    K_vac_node = K_vac[:, vac_node - 1]
    finiteK = np.isfinite(K_vac_node)
    out["gr"] = {
        "node_ode_vs_exact_LTB_max_rel_error": err_gr,
        "max_matter_cell_density_over_rho_max": float(np.nanmax(rho_matter[np.isfinite(rho_matter)]) / RHO_MAX),
        "time_first_cell_reaches_rho_max": float(t_gr[first_cap]),
        "time_matter_surface_becomes_trapped": float(t_gr[surface_trapped_idx]),
        "time_first_trapped_surface": float(t_gr[first_trapped_idx]),
        "compactness_2m_over_R_of_any_region_at_rho_max_is_(R/ell)^2": True,
        "central_singularity_time_exact": float(t_sing_center),
        "empty_node_singularity_time_exact": float((2.0 / 3.0) * R0[vac_node] ** 1.5 / np.sqrt(2.0 * m_node[vac_node])),
        "empty_node_reaches_R_1e-3_ell_at_tau": float(info_gr["floor_time"][vac_node]),
        "empty_node_kretschmann_over_deSitter_bound_at_R_1e-3_ell": float(48.0 * m_node[vac_node] ** 2 / 1e-18 / 24.0),
        "shell_crossing": bool(np.any(crossed)),
    }
    # ---------- local cap (rule acts on matter cells only) ----------
    # Empty node: enclosed mass fixed (p = 0 there), so its motion is GR free fall regardless of the matter.
    out["local_cap"] = {
        "empty_node_energy_density": 0.0,
        "empty_node_reaches_R_over_ell": 1e-3,
        "empty_node_reaches_it_at_tau": float(info_gr["floor_time"][vac_node]),
        "empty_node_kretschmann_exceeds_deSitter_bound_below_R_over_Rcore": float(2.0 ** (-1.0 / 6.0)),
        "rule_ever_fires_on_empty_node": False,
        "singularity_avoided": bool(np.isnan(info_gr["floor_time"][vac_node])),
    }
    # ---------- curvature cap (LQC-type, V2) ----------
    tb_all = np.sqrt((R0[1:] ** 3 - r_core(m_node[1:]) ** 3) / (4.5 * m_node[1:]))
    t_cc = np.linspace(0.0, 1.6 * tb_all.max(), n_t)
    R_cc, err_cc, info_cc = evolve_nodes(R0, m_node, "curvature_cap", t_cc)
    rho_cell_c, rho_bar_c, K_vac_c, crossed_c = cell_diagnostics(R_cc, m_node, dm, matter_cell)
    i_cross = int(np.argmax(crossed_c)) if np.any(crossed_c) else None
    pre = slice(0, i_cross if i_cross else n_t)
    min_R = info_cc["bounce_R"][1:]
    bounce_err = float(np.nanmax(np.abs(min_R - r_core(m_node[1:])) / r_core(m_node[1:])))
    rho_bar_max_pre = float(np.nanmax(rho_bar_c[pre]) / RHO_MAX)
    rho_cell_pre = np.where(matter_cell[None, :], rho_cell_c[pre], np.nan)
    eff_compact = (2.0 * m_node[None, 1:] / R_cc[:, 1:]) * (1.0 - r_core(m_node[None, 1:]) ** 3 / R_cc[:, 1:] ** 3)
    trapped_c = np.any(eff_compact > 1.0)
    cross_info = {}
    if i_cross is not None:
        vol = R_cc[i_cross, 1:] ** 3 - R_cc[i_cross, :-1] ** 3
        k = int(np.argmin(vol))
        cross_info = {
            "first_shell_crossing_time": float(t_cc[i_cross]),
            "first_bounce_time": float(tb_all.min()),
            "crossing_radius_over_ell": float(R_cc[i_cross, k + 1]),
            "crossing_radius_over_rs": float(R_cc[i_cross, k + 1] / r_s),
            "crossing_inside_trapped_region": bool(eff_compact[i_cross, k] > 1.0),
        }
    out["curvature_cap_V2"] = {
        "node_ode_vs_exact_max_rel_error": err_cc,
        "bounce_radius_equals_R_core_of_enclosed_mass_rel_error": bounce_err,
        "empty_node_bounce_radius_over_ell": float(min_R[vac_node - 1]),
        "R_core_of_M_over_ell": r_core(M),
        "max_mean_density_over_rho_max_before_crossing": rho_bar_max_pre,
        "max_matter_cell_density_over_rho_max_before_crossing": float(np.nanmax(rho_cell_pre[np.isfinite(rho_cell_pre)]) / RHO_MAX),
        "trapped_region_forms": bool(trapped_c),
        **cross_info,
    }
    return out


if __name__ == "__main__":
    import json
    res = {
        "homogeneous_stiff_pressure": homogeneous_collapse("stiff_pressure"),
        "homogeneous_vacuum_conversion_flat": homogeneous_collapse("vacuum"),
        "homogeneous_vacuum_conversion_closed": homogeneous_collapse("vacuum", k_curv=0.005),
        "homogeneous_lqc": homogeneous_collapse("lqc"),
        "lattice_parabolic": run_collapse_study(profile="parabolic"),
        "lattice_homogeneous": run_collapse_study(profile="homogeneous"),
    }
    print(json.dumps(res, indent=2))
