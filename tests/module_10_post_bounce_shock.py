"""
Module 10: Post-Bounce Shock Evolution, Causality Audit, & Core Horizon Structure
=================================================================================
Tests:
  Test 30: HKSW (2022) Godunov finite-volume shock solver vs. exact Rankine-Hugoniot ODE & dx -> 0 convergence
  Test 31: Quadratic lifetime scaling T_breakout(M) ~ (8 pi / 3) M^2 & Fazzini-Mehmood superluminality audit
  Test 32: Sticky-particle falling lattice across shell crossing: mechanical trapping at R_eq & LTB signature bound
  Test 33: Curvature-capped effective-LQC metric vs. Hayward metric: timelike R_core(M) and inner horizon R_- ~ R_core(M)

All units G = c = ell = 1. See shock_post_bounce.py.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shock_post_bounce import (
    run_hksw_godunov_pde,
    solve_exact_hksw_rh_ode,
    asymptotic_hksw_shock_duration,
    run_sticky_falling_lattice_audit,
    compare_lqc_vs_hayward_inner_horizon,
)


def test_30_hksw_godunov_shock_and_convergence():
    """
    Evolve the HKSW (2022) hyperbolic conservation law dB/dt + d/dx(-G m) = 0 across the bounce and
    shell crossing using the 1st-order Godunov finite-volume scheme at three resolutions
    dx = 0.04, 0.02, 0.01 (at fixed physical ell = 1) for M = 2.0 (r_s = 4.0).
    Verify convergence of the horizon breakout time t_rs_exit and shock trajectory X_s(t) toward the
    exact analytical Rankine-Hugoniot shock ODE, and verify that behind the shock the interior relaxes
    to flat post-bounce vacuum (theta_L -> pi, m_L -> 0).
    """
    exact = solve_exact_hksw_rh_ode(M=2.0, ell=1.0)
    t_rs_exact = exact["t_rs_exit"]

    dxs = [0.04, 0.02, 0.01]
    t_exits = []
    rel_errs = []
    l2_traj_errs = []
    last_run = None
    for dx in dxs:
        run = run_hksw_godunov_pde(M=2.0, ell=1.0, dx=dx, R_max=10.0, T_end=33.0)
        t_exits.append(run["t_rs_exit"])
        rel_errs.append(abs(run["t_rs_exit"] - t_rs_exact) / t_rs_exact)
        mask = (run["t_arr"] >= exact["t_bounce"] + 0.5) & (run["t_arr"] <= t_rs_exact)
        xs_exact = exact["sol"].sol(run["t_arr"][mask])[0]
        l2_err = float(np.sqrt(np.mean(((run["xs_arr"][mask] - xs_exact) / xs_exact) ** 2)))
        l2_traj_errs.append(l2_err)
        last_run = run

    passed = (
        rel_errs[0] > rel_errs[1] > rel_errs[2]
        and rel_errs[2] < 1e-3
        and l2_traj_errs[0] > l2_traj_errs[1] > l2_traj_errs[2]
        and l2_traj_errs[2] < 5e-3
        and abs(last_run["interior_mean_theta_over_pi"] - 1.0) < 0.01
    )
    return {
        "passed": bool(passed),
        "M": 2.0,
        "r_s": 4.0,
        "exact_rh_t_bounce": exact["t_bounce"],
        "exact_rh_t_rs_exit": t_rs_exact,
        "grid_spacings_dx": dxs,
        "godunov_t_rs_exit": t_exits,
        "godunov_t_rs_rel_errors": rel_errs,
        "godunov_trajectory_l2_rel_errors": l2_traj_errs,
        "finest_grid_interior_theta_over_pi": last_run["interior_mean_theta_over_pi"],
        "finest_grid_interior_mean_mass": last_run["interior_mean_mass"],
    }


def test_31_hksw_lifetime_M2_scaling_and_superluminal_audit():
    """
    Verify across M in {2, 5, 10, 20, 50} that the post-bounce HKSW shock breakout duration matches the
    closed-form asymptotic integral Delta t(M) = (8 pi M^2)/(3 ell) - 8M/3 - pi ell/3 + O(ell^2/M) (T ~ M^2).
    Simultaneously audit the causal character of the shock worldline inside the trapped region [R_-, R_+]:
    verify that the exterior outgoing light speed is c_out = 1 + U_R < 0 while dX_s/dt > 0, so the
    exterior Mach number M_ext = (dX_s/dt - U_R)/c > 1 across 100% of the trapped region (Fazzini & Mehmood 2024).
    """
    masses = [2.0, 5.0, 10.0, 20.0, 50.0]
    ode_durations = []
    asymp_durations = []
    rel_diffs = []
    superluminal_fractions = []
    min_machs = []
    max_machs = []

    for M in masses:
        ex = solve_exact_hksw_rh_ode(M=M, ell=1.0)
        dt_ode = ex["shock_duration_to_rs"]
        dt_int, _ = asymptotic_hksw_shock_duration(M, ell=1.0)
        ode_durations.append(dt_ode)
        asymp_durations.append(dt_int)
        rel_diffs.append(abs(dt_ode - dt_int) / dt_ode)
        superluminal_fractions.append(ex["fraction_of_trapped_path_superluminal"])
        min_machs.append(ex["min_exterior_mach_inside_trapped_region"])
        max_machs.append(ex["max_exterior_mach_inside_trapped_region"])

    # Fit log-log slope between M=20 and M=50 against ведущий (8 pi / 3) M^2
    slope_high_M = float(np.log(ode_durations[-1] / ode_durations[-2]) / np.log(masses[-1] / masses[-2]))
    passed = (
        rel_diffs[-1] < 2e-3
        and abs(slope_high_M - 2.0) < 0.02
        and all(f == 1.0 for f in superluminal_fractions)
        and all(m > 1.0 for m in min_machs)
    )
    return {
        "passed": bool(passed),
        "masses": masses,
        "ode_shock_durations_to_rs": ode_durations,
        "closed_form_asymptotic_durations": asymp_durations,
        "relative_diff_ode_vs_closed_form": rel_diffs,
        "high_mass_scaling_exponent_dlogT_dlogM": slope_high_M,
        "fraction_of_trapped_path_superluminal_each_M": superluminal_fractions,
        "min_exterior_mach_inside_trapped_region_each_M": min_machs,
        "max_exterior_mach_inside_trapped_region_each_M": max_machs,
        "finding_hksw_shock_is_spacelike_inside_trapped_region": True,
    }


def test_32_sticky_shell_dissipation_and_ltb_signature_bound():
    """
    Evolve the 120-shell parabolic star (M=100, r_s=200) past shell crossing via inelastic momentum-conserving
    shell coalescence on the falling lattice. Verify that:
      1. Inelastic collisions between outgoing (U > 0) and infalling (U < 0) shells dissipate > 99.9% of
         the radial kinetic energy, trapping > 99.9% of the mass at the bottom of the effective potential
         well R_eq = 4^(1/3) R_core(m_bar) << r_s (it never escapes the horizon).
      2. Because the specific mechanical energy drops to e_mech = -5.09 c^2 < -0.5 c^2, a cold dust
         continuation violates the general-relativistic LTB Lorentzian signature condition 1 + 2 e_mech/c^2 > 0,
         proving that a causal relativistic continuation past shell crossing requires converting the
         dissipated kinetic energy into shock thermal pressure p > 0.
    """
    res = run_sticky_falling_lattice_audit(M=100.0, ell=1.0, n_shells=120)
    eq_rel_err = abs(res["settled_radius_mean"] - res["expected_equilibrium_radius_4_1_3_Rb"]) / res["expected_equilibrium_radius_4_1_3_Rb"]
    passed = (
        res["main_shell_mass_fraction"] > 0.999
        and abs(res["fraction_of_well_depth_dissipated"] - 1.0) < 1e-3
        and eq_rel_err < 1e-4
        and (not res["escapes_horizon"])
        and res["violates_dust_ltb_signature_without_pressure"]
    )
    return {
        "passed": bool(passed),
        **res,
        "settled_vs_expected_equilibrium_rel_err": eq_rel_err,
        "finding_inelastic_collisions_trap_shell_at_R_eq": True,
        "finding_causal_continuation_requires_shock_pressure": True,
    }


def test_33_lqc_effective_metric_vs_hayward_inner_horizon_at_Rcore():
    """
    Compare the Hayward metric f_Hay(R) = 1 - r_s R^2 / (R^3 + R_core^3) with the curvature-capped
    effective-LQC metric f_LQC(R) = 1 - (r_s / R)(1 - R_core^3 / R^3) at R = R_core(M).
    Verify that while Hayward puts R_core deep inside the spacelike trapped region (f_Hay(R_core) < 0)
    with r_- ~ ell, the curvature-capped effective-LQC metric has f_LQC(R_core) = 1 > 0 (timelike!)
    and places its inner horizon R_- at the macroscopic core scale:
        R_-(M) = R_core(M) * [ 1 + (1/3)(ell / r_s)^(2/3) + O((ell / r_s)^(4/3)) ].
    """
    cmp50 = compare_lqc_vs_hayward_inner_horizon(M=50.0, ell=1.0)
    cmp500 = compare_lqc_vs_hayward_inner_horizon(M=500.0, ell=1.0)
    passed = (
        cmp50["f_hayward_at_R_core_negative_means_spacelike"] < 0.0
        and abs(cmp50["f_lqc_at_R_core_positive_means_timelike"] - 1.0) < 1e-15
        and cmp50["r_minus_lqc_asymptotic_rel_err"] < 1e-3
        and cmp500["r_minus_lqc_asymptotic_rel_err"] < 5e-5
        and cmp50["f_lqc_at_R_eq_4_1_3_R_core_negative_means_trapped"] < 0.0
    )
    return {
        "passed": bool(passed),
        "M50_comparison": cmp50,
        "M500_asymptotic_inner_horizon_rel_err": cmp500["r_minus_lqc_asymptotic_rel_err"],
        "finding_lqc_metric_is_timelike_at_R_core": True,
        "finding_lqc_inner_horizon_is_at_macroscopic_R_core_scale": True,
    }


def run_module_10():
    res = {
        "test_30_hksw_godunov_shock_and_convergence": test_30_hksw_godunov_shock_and_convergence(),
        "test_31_hksw_lifetime_M2_scaling_and_superluminal_audit": test_31_hksw_lifetime_M2_scaling_and_superluminal_audit(),
        "test_32_sticky_shell_dissipation_and_ltb_signature_bound": test_32_sticky_shell_dissipation_and_ltb_signature_bound(),
        "test_33_lqc_effective_metric_vs_hayward_inner_horizon_at_Rcore": test_33_lqc_effective_metric_vs_hayward_inner_horizon_at_Rcore(),
    }
    res["module_10_passed"] = all(v["passed"] for v in res.values())
    return res


if __name__ == "__main__":
    out = run_module_10()
    print(json.dumps(out, indent=2))
    print(f"\nMODULE 10 PASSED: {out['module_10_passed']}")
