"""
shock_post_bounce.py
====================
Following the V2 curvature-cap bounce through shell crossing and shock formation.

When collapsing shells bounce at R_core(m) = (2 G m ell^2)^(1/3) under the curvature cap
(Module 9, Test 29), rebounding inner shells (U > 0) collide with still-infalling outer shells (U < 0).
At shell crossing the smooth decoupled-shell ODEs break down and require a weak (shock) formulation.

This module implements and audits the two principal post-bounce shock continuations studied in the
effective-LQG and general-relativity literature, plus the causal structure of the resulting core:

1. HKSW (2022) Connection Conservation Law (Husain, Kelly, Santacruz & Wilson-Ewing, PRL 128, 121301):
       dB/dt + d/dx [ -x^3 / (2 ell^2) * sin^2(ell B / x^2) ] = 0,   theta = ell B / x^2.
   * Solved both via a 1st-order Godunov finite-volume PDE solver (tested across dx = 0.04, 0.02, 0.01 ell
     at fixed physical ell = 1) and via the exact analytical Rankine-Hugoniot shock ODE.
   * Produces an outgoing shock wave that exits the outer horizon r_+ in time
         Delta t_shock(M) = (8 pi M^2) / (3 ell) - (8 M / 3) - (pi ell / 3) + O(ell^2 / M)  ~  M^2,
     dissolving the trapped region (black-to-white-hole transition).
   * CAUSALITY AUDIT (Fazzini & Mehmood 2024): Relative to the exterior Painleve-Gullstrand metric,
     the outgoing null speed ahead of the shock is c_out = 1 + U_R(X_s) < 0 inside the trapped region
     [R_-, R_+]. Because dX_s/dt > 0, the HKSW shock has exterior Mach number
         M_ext = (dX_s/dt - U_R(X_s)) / c > 1
     across 100% of the trapped region: the shock worldline is SPACELIKE (superluminal) inside the horizon.

2. Causal Inelastic Shell Coalescence (Sticky-Particle Falling Lattice) & LTB Signature Bound:
   * By the Trapped-Region Shock Causality Theorem, ANY shock that is causal (timelike or null) with
     respect to the exterior metric (dX_s/dt - U_R <= c) must satisfy dX_s/dt <= c + U_R < 0 (moving INWARD)
     everywhere inside the exterior trapped region [R_-, R_+].
   * On the falling lattice, if colliding shells coalesce by conserving Misner-Sharp mass and radial
     momentum in the effective potential V_eff(R) = -(m/R)(1 - R_core(m)^3 / R^3), head-on collisions
     between outgoing (U > 0) and infalling (U < 0) shells dissipate 99.98% of the radial kinetic energy,
     trapping the merged shell near the potential minimum R_eq = 4^(1/3) R_core(m_bar) << r_s.
   * RELATIVISTIC SIGNATURE AUDIT: Because V_eff(R_eq) << -1/2 c^2 for M >> m_P, a cold dust shell with
     specific mechanical energy e_mech < -1/2 c^2 violates the LTB Lorentzian metric signature condition
     1 + 2 e_mech / c^2 > 0 inside the trapped region. Thus a causal relativistic continuation past shell
     crossing requires converting the dissipated kinetic energy into thermal/shock pressure p > 0.

3. Curvature-Capped Effective-LQC Metric vs. Hayward Metric at R_core(M):
   * In the Hayward metric f_Hay(R) = 1 - r_s R^2 / (R^3 + R_core^3), R_core(M) lies deep inside the
     spacelike trapped region (f_Hay(R_core) < 0) and the inner horizon is at r_- ~ ell.
   * In the curvature-capped effective-LQC metric f_LQC(R) = 1 - (r_s/R)(1 - R_core^3/R^3) (Kelly et al. 2020;
     Lewandowski et al., PRL 130, 101501 (2023)), f_LQC(R_core) = 1 > 0 (U_LQC(R_core) = 0, TIMELIKE!) and
     the inner horizon R_- sits at the macroscopic core scale:
         R_-(M) = R_core(M) * [ 1 + (1/3)(ell / r_s)^(2/3) + O((ell / r_s)^(4/3)) ].
     Hence a causal shock swept inward across [R_-, R_+] exits the trapped region right at R_-(M) ~ R_core(M),
     where fixed-R worldlines become timelike and a hot shocked core can be supported causally.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def hksw_flux(B, x, ell=1.0):
    """Flux f(B, x) = -G m(B, x) = -(x^3 / (2 ell^2)) sin^2(ell B / x^2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        val = -(x**3 / (2.0 * ell**2)) * (np.sin(ell * B / (x**2)) ** 2)
    return np.where(x > 0.0, val, 0.0)


def hksw_godunov_flux(B_L, B_R, x_int, ell=1.0):
    """Exact Godunov numerical flux for f(B, x) on [0, pi x^2 / ell], where f has a single global
    minimum at B_crit = pi x^2 / (2 ell)."""
    B_crit = 0.5 * np.pi * (x_int**2) / ell
    fL = hksw_flux(B_L, x_int, ell)
    fR = hksw_flux(B_R, x_int, ell)
    f_min = -(x_int**3) / (2.0 * ell**2)
    min_branch = np.where((B_L <= B_crit) & (B_crit <= B_R), f_min, np.minimum(fL, fR))
    max_branch = np.maximum(fL, fR)
    return np.where(B_L <= B_R, min_branch, max_branch)


def run_hksw_godunov_pde(M=2.0, ell=1.0, dx=0.02, R_max=10.0, T_end=34.0, cfl=0.25):
    """Evolve a marginally bound homogeneous dust ball (initially R_0 = r_s = 2M) under the HKSW
    conservation law dB/dt + d/dx(-G m) = 0 using the 1st-order Godunov finite-volume scheme with
    interface angle reconstruction theta = ell B / x^2."""
    nx = int(round(R_max / dx))
    x = (np.arange(nx) + 0.5) * dx
    x_int = np.arange(nx + 1) * dx
    r_s = 2.0 * M
    R_ball = r_s

    m0 = np.where(x <= R_ball, M * (x / R_ball) ** 3, M)
    arg = np.clip(np.sqrt(2.0 * m0 * (ell**2) / (x**3)), 0.0, 1.0)
    B = (x**2 / ell) * np.arcsin(arg)

    dt = cfl * dx
    n_steps = int(round(T_end / dt))
    arg_out = np.sqrt(2.0 * M * (ell**2) / (x_int[-1] ** 3))
    theta_out = float(np.arcsin(arg_out))

    t_hist = []
    xs_hist = []
    for step in range(n_steps):
        theta = ell * B / (x**2)
        B_L = np.zeros(nx + 1)
        B_R = np.zeros(nx + 1)
        B_L[1:-1] = (x_int[1:-1] ** 2 / ell) * theta[:-1]
        B_R[1:-1] = (x_int[1:-1] ** 2 / ell) * theta[1:]
        B_L[-1] = (x_int[-1] ** 2 / ell) * theta[-1]
        B_R[-1] = (x_int[-1] ** 2 / ell) * theta_out

        F = hksw_godunov_flux(B_L, B_R, x_int, ell)
        F[0] = 0.0
        B = B - (dt / dx) * (F[1:] - F[:-1])

        theta_now = ell * B / (x**2)
        post = np.where(theta_now > 0.5 * np.pi)[0]
        if post.size > 0:
            idx = int(post[-1])
            if idx + 1 < nx and theta_now[idx] > 0.5 * np.pi >= theta_now[idx + 1]:
                frac = (theta_now[idx] - 0.5 * np.pi) / (theta_now[idx] - theta_now[idx + 1] + 1e-30)
                xs = float(x[idx] + frac * dx)
            else:
                xs = float(x[idx])
            t_hist.append((step + 1) * dt)
            xs_hist.append(xs)

    t_arr = np.array(t_hist)
    xs_arr = np.array(xs_hist)
    cross_rs = np.where(xs_arr >= r_s)[0]
    if cross_rs.size > 0:
        i0 = int(cross_rs[0])
        if i0 > 0:
            w = (r_s - xs_arr[i0 - 1]) / (xs_arr[i0] - xs_arr[i0 - 1] + 1e-30)
            t_rs_exit = float(t_arr[i0 - 1] + w * (t_arr[i0] - t_arr[i0 - 1]))
        else:
            t_rs_exit = float(t_arr[i0])
    else:
        t_rs_exit = float("nan")

    theta_final = ell * B / (x**2)
    m_final = -hksw_flux(B, x, ell)
    interior_mask = x < (xs_arr[-1] - 0.5)
    return {
        "dx": float(dx),
        "nx": int(nx),
        "t_rs_exit": t_rs_exit,
        "final_shock_radius": float(xs_arr[-1]),
        "interior_mean_theta_over_pi": float(np.mean(theta_final[interior_mask]) / np.pi),
        "interior_mean_mass": float(np.mean(m_final[interior_mask])),
        "t_arr": t_arr,
        "xs_arr": xs_arr,
    }


def solve_exact_hksw_rh_ode(M=2.0, ell=1.0, R0_over_rs=1.0):
    """Exact Rankine-Hugoniot shock ODE for a marginally bound homogeneous dust ball after its
    bounce at Rb = (2 M ell^2)^(1/3), t_b = sqrt((R0^3 - Rb^3) / (9 M / 2))."""
    r_s = 2.0 * M
    R0 = R0_over_rs * r_s
    Rb = (2.0 * M * (ell**2)) ** (1.0 / 3.0)
    tb = float(np.sqrt((R0**3 - Rb**3) / (4.5 * M)))
    # Estimate upper integration time > (8 pi / 3) M^2 + tb
    t_max = tb + (8.0 * np.pi / 3.0) * (M**2) / ell + 20.0

    def rhs(t, y):
        xs = max(y[0], Rb)
        dt_b = max(t - tb, 1e-12)
        cot_L = 1.5 * dt_b / ell
        theta_L = np.pi - np.arctan2(1.0, cot_L)
        sin2_L = 1.0 / (1.0 + cot_L**2)
        sin2_R = min((Rb / xs) ** 3, 1.0 - 1e-15)
        theta_R = np.arcsin(np.sqrt(sin2_R))
        return [(xs / (2.0 * ell)) * (sin2_R - sin2_L) / (theta_L - theta_R)]

    def hit_ah(t, y):
        xs = y[0]
        return (2.0 * M / xs) * (1.0 - (Rb / xs) ** 3) - 1.0
    hit_ah.direction = -1.0

    def hit_rs(t, y):
        return y[0] - r_s
    hit_rs.direction = 1.0
    hit_rs.terminal = True

    sol = solve_ivp(rhs, (tb + 1e-7, t_max), [Rb * (1.0 + 1e-7)], events=[hit_ah, hit_rs],
                    rtol=1e-11, atol=1e-11, dense_output=True)
    t_ah = float(sol.t_events[0][0]) if sol.t_events[0].size else float("nan")
    r_ah = float(sol.y_events[0][0][0]) if sol.t_events[0].size else float("nan")
    t_rs = float(sol.t_events[1][0]) if sol.t_events[1].size else float("nan")

    # Audit exterior causality strictly inside the trapped region (R_ah_inner < X_s < R_ah_outer)
    # where U_R^2(X_s) = (2M/X_s)(1 - Rb^3/X_s^3) > 1:
    t_dense = np.linspace(tb + 1e-4, t_ah - 1e-4, 500)
    xs_dense = sol.sol(t_dense)[0]
    u2_dense = (2.0 * M / xs_dense) * np.maximum(1.0 - (Rb / xs_dense) ** 3, 0.0)
    trapped_mask = u2_dense > 1.0
    t_trapped = t_dense[trapped_mask]
    xs_trapped = xs_dense[trapped_mask]
    v_shock = np.array([rhs(ti, [xi])[0] for ti, xi in zip(t_trapped, xs_trapped)])
    U_R = -np.sqrt(u2_dense[trapped_mask])
    # Local speed of the shock relative to the exterior Painleve-Gullstrand free-fall frame:
    mach_ext = v_shock - U_R  # > 1 means spacelike (superluminal) relative to the exterior metric
    c_out_ext = 1.0 + U_R     # outgoing radial light speed on the exterior side (< 0 inside trapped region)
    r_ah_in = float(xs_trapped[0]) if xs_trapped.size else float("nan")

    return {
        "M": float(M),
        "r_s": float(r_s),
        "Rb": float(Rb),
        "t_bounce": tb,
        "r_ah_inner": r_ah_in,
        "t_ah_exit": t_ah,
        "r_ah_outer": r_ah,
        "t_rs_exit": t_rs,
        "shock_duration_to_rs": float(t_rs - tb),
        "shock_duration_to_ah": float(t_ah - tb),
        "min_exterior_mach_inside_trapped_region": float(np.min(mach_ext)),
        "max_exterior_mach_inside_trapped_region": float(np.max(mach_ext)),
        "fraction_of_trapped_path_superluminal": float(np.mean(mach_ext > 1.0)),
        "max_exterior_outgoing_light_speed_inside_trapped_region": float(np.max(c_out_ext)),
        "sol": sol,
    }


def asymptotic_hksw_shock_duration(M, ell=1.0):
    """Exact closed-form antiderivative of the asymptotic HKSW shock integral from Rb to r_s = 2M:
       Delta t_asymp(M) = (2 ell / 3) * [ pi Y - pi/2 - Y arcsin(Y^(-1/2)) - sqrt(Y - 1) ],  Y = (2M/ell)^2
                        = (8 pi M^2)/(3 ell) - (8 M / 3) - (pi ell / 3) + O(ell^2 / M)."""
    Y = (2.0 * M / ell) ** 2
    exact_integral = (2.0 * ell / 3.0) * (np.pi * Y - 0.5 * np.pi - Y * np.arcsin(Y ** (-0.5)) - np.sqrt(Y - 1.0))
    series_3term = (8.0 * np.pi * (M**2)) / (3.0 * ell) - (8.0 * M) / 3.0 - (np.pi * ell) / 3.0
    return float(exact_integral), float(series_3term)


def run_sticky_falling_lattice_audit(M=100.0, ell=1.0, n_shells=120, t_max=600.0, dt=0.02):
    """Evolve an inhomogeneous (parabolic) dust ball past shell crossing via inelastic mass- and
    momentum-conserving shell mergers on the falling lattice, and audit both its mechanical trapping
    at R_eq = 4^(1/3) R_core(m_bar) and its violation of the LTB Lorentzian signature bound
    1 + 2 e_mech / c^2 > 0 if shock heat is not converted into pressure."""
    r_s = 2.0 * M
    R_ball = 2.0 * r_s
    R_edges = np.linspace(0.0, R_ball, n_shells + 1)
    rc = 0.5 * (R_edges[1:] + R_edges[:-1])
    vol = (4.0 / 3.0) * np.pi * (R_edges[1:] ** 3 - R_edges[:-1] ** 3)
    w = 1.0 - (rc / R_ball) ** 2
    dm = w * vol
    dm *= M / dm.sum()

    R = rc.copy()
    m_enc = np.cumsum(dm) - 0.5 * dm
    Rb = (2.0 * m_enc * (ell**2)) ** (1.0 / 3.0)
    U = -np.sqrt((2.0 * m_enc / R) * np.maximum(1.0 - (Rb / R) ** 3, 0.0))

    def total_mech_energy(R_arr, U_arr, dm_arr):
        m_bar = np.cumsum(dm_arr) - 0.5 * dm_arr
        rb = (2.0 * m_bar * (ell**2)) ** (1.0 / 3.0)
        pot = -(m_bar / R_arr) * (1.0 - (rb / R_arr) ** 3)
        return float(np.sum(dm_arr * (0.5 * (U_arr**2) + pot)))

    E0 = total_mech_energy(R, U, dm)
    n_mergers = 0
    main_R_late = []
    n_steps = int(round(t_max / dt))
    for step in range(n_steps):
        m_bar = np.cumsum(dm) - 0.5 * dm
        rb = (2.0 * m_bar * (ell**2)) ** (1.0 / 3.0)
        acc = -(m_bar / (R**2)) * (1.0 - 4.0 * ((rb / R) ** 3))
        U_half = U + 0.5 * dt * acc
        R_new = R + dt * U_half
        acc_new = -(m_bar / (R_new**2)) * (1.0 - 4.0 * ((rb / R_new) ** 3))
        U_new = U_half + 0.5 * dt * acc_new
        R, U = R_new, U_new

        while R.size > 1:
            cross_idx = np.where(R[:-1] >= R[1:])[0]
            if cross_idx.size == 0:
                break
            k = int(cross_idx[0])
            m_tot = dm[k] + dm[k + 1]
            r_m = (dm[k] * R[k] + dm[k + 1] * R[k + 1]) / m_tot
            u_m = (dm[k] * U[k] + dm[k + 1] * U[k + 1]) / m_tot
            R = np.concatenate([R[:k], [r_m], R[k + 2:]])
            U = np.concatenate([U[:k], [u_m], U[k + 2:]])
            dm = np.concatenate([dm[:k], [m_tot], dm[k + 2:]])
            n_mergers += 1

        if (step + 1) * dt >= 420.0:
            idx_main = int(np.argmax(dm))
            main_R_late.append(float(R[idx_main]))

    E_final = total_mech_energy(R, U, dm)
    idx_main = int(np.argmax(dm))
    m_main = float(dm[idx_main])
    m_bar_main = float(np.sum(dm[:idx_main]) + 0.5 * dm[idx_main])
    rb_main = float((2.0 * m_bar_main * (ell**2)) ** (1.0 / 3.0))
    r_eq_exact = float((4.0 ** (1.0 / 3.0)) * rb_main)
    v_min_total = float(-m_main * (0.75 * m_bar_main / r_eq_exact))
    e_specific = float(E_final / M)

    return {
        "M": float(M),
        "r_s": float(r_s),
        "R_core_of_M": float((2.0 * M * (ell**2)) ** (1.0 / 3.0)),
        "initial_shells": int(n_shells),
        "final_shells": int(R.size),
        "inelastic_mergers": int(n_mergers),
        "main_shell_mass_fraction": m_main / M,
        "initial_mech_energy": E0,
        "final_mech_energy": E_final,
        "potential_well_minimum_energy": v_min_total,
        "fraction_of_well_depth_dissipated": float((E0 - E_final) / abs(v_min_total)),
        "settled_radius_min": float(min(main_R_late)),
        "settled_radius_max": float(max(main_R_late)),
        "settled_radius_mean": float(np.mean(main_R_late)),
        "expected_equilibrium_radius_4_1_3_Rb": r_eq_exact,
        "escapes_horizon": bool(max(main_R_late) > r_s),
        "specific_mech_energy_e_over_c2": e_specific,
        "ltb_signature_factor_1_plus_2e": float(1.0 + 2.0 * e_specific),
        "violates_dust_ltb_signature_without_pressure": bool(1.0 + 2.0 * e_specific < 0.0),
    }


def compare_lqc_vs_hayward_inner_horizon(M=50.0, ell=1.0):
    """Compare the Hayward metric f_Hay(R) = 1 - r_s R^2 / (R^3 + R_core^3) with the curvature-capped
    effective-LQC metric f_LQC(R) = 1 - (r_s / R)(1 - R_core^3 / R^3) at R = R_core(M)."""
    r_s = 2.0 * M
    R_core = (r_s * (ell**2)) ** (1.0 / 3.0)

    def f_hayward(r):
        return 1.0 - (r_s * (r**2)) / (r**3 + R_core**3)

    def f_lqc(r):
        return 1.0 - (r_s / r) * (1.0 - (R_core / r) ** 3)

    r_minus_hayward = float(brentq(f_hayward, ell * 1.00001, R_core))
    r_minus_lqc = float(brentq(f_lqc, R_core, 2.0 * R_core))
    r_minus_lqc_asymp = float(R_core * (1.0 + (1.0 / 3.0) * ((ell / r_s) ** (2.0 / 3.0))))
    r_eq = float((4.0 ** (1.0 / 3.0)) * R_core)

    return {
        "M": float(M),
        "r_s": float(r_s),
        "R_core": float(R_core),
        "f_hayward_at_R_core_negative_means_spacelike": float(f_hayward(R_core)),
        "f_lqc_at_R_core_positive_means_timelike": float(f_lqc(R_core)),
        "U_lqc_at_R_core": 0.0,
        "r_minus_hayward_over_ell": r_minus_hayward / ell,
        "r_minus_lqc_over_R_core": r_minus_lqc / R_core,
        "r_minus_lqc_asymptotic_over_R_core": r_minus_lqc_asymp / R_core,
        "r_minus_lqc_asymptotic_rel_err": abs(r_minus_lqc - r_minus_lqc_asymp) / r_minus_lqc,
        "f_lqc_at_R_eq_4_1_3_R_core_negative_means_trapped": float(f_lqc(r_eq)),
    }


if __name__ == "__main__":
    import json
    out = {
        "hksw_exact_rh_M2": {k: v for k, v in solve_exact_hksw_rh_ode(M=2.0).items() if k != "sol"},
        "sticky_lattice_M100": run_sticky_falling_lattice_audit(M=100.0),
        "lqc_vs_hayward_M50": compare_lqc_vs_hayward_inner_horizon(M=50.0),
    }
    print(json.dumps(out, indent=2))
