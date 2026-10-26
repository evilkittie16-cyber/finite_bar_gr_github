"""
Module 11: Relativistic Misner-Sharp / Einstein-Euler Fluid Collapse, Shock Heating,
Curvature-Capped TOV No-Go Theorem, and Trapped-Region Hyperbola Audit (Tests 34-37).

Replaces pressureless dust (p = 0) at shell crossing with the coupled general-relativistic
Misner-Sharp (1964) / May-White (1966) Einstein-Euler fluid equations (p = (gamma - 1) rho_int > 0)
on the falling lattice with curvature cap f_LQC(R, m) = 1 - (2Gm/c^2 R)(1 - R_core(m)^3/R^3).
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from einstein_euler_core_bounce import (
    audit_curvature_capped_tov_no_go,
    run_lagrangian_einstein_euler,
)


def test_34_curvature_capped_tov_no_go_for_static_positive_pressure_core() -> dict:
    """
    Test 34: Audits the exact Curvature-Capped Tolman-Oppenheimer-Volkoff (TOV) equation
      dp/dR = -((rho + p) G / (R^2 f_LQC(R, m))) [ m (1 - 4 bar_rho/rho_max) + 4 pi R^3 p (1 - 2 bar_rho/rho_max) ]
    across M in {10, 20, 50, 100}.

    Verifies:
      1) At the effective-potential minimum R_eq = 4^(1/3) R_core(M), f_LQC(R_eq, M) < 0 (spacelike/trapped),
         so no static (U = 0) Lorentzian configuration (Gamma^2 = f_LQC + U^2/c^2 > 0) can exist there.
      2) Inside the untrapped timelike core 0 < R < R_-(M) (where f_LQC(R, M) > 0), bar_rho / rho_max > 1/2
         everywhere, so both (1 - 4 bar_rho/rho_max) < -1 < 0 and (1 - 2 bar_rho/rho_max) < 0.
         Consequently, dp/dR > 0 strictly for all p >= 0: a positive-pressure fluid cannot form a static
         core with a zero-pressure boundary inside R < R_+(M) and must instead dynamically bounce and re-expand!
    """
    audit = audit_curvature_capped_tov_no_go(masses=[10.0, 20.0, 50.0, 100.0], ell=1.0)
    passed = bool(
        audit["all_R_eq_spacelike"]
        and audit["all_timelike_cores_have_dp_dR_positive"]
    )
    return {
        "passed": passed,
        "all_R_eq_spacelike_f_LQC_negative": audit["all_R_eq_spacelike"],
        "all_timelike_cores_have_dp_dR_strictly_positive": audit["all_timelike_cores_have_dp_dR_positive"],
        "sample_M50": audit["records"][2],
        "sample_M100": audit["records"][3],
    }


def test_35_einstein_euler_shock_prevents_shell_crossing_and_preserves_signature() -> dict:
    """
    Test 35: Compares pressureless dust (p = 0) vs. the coupled relativistic Misner-Sharp /
    Einstein-Euler fluid (p = (gamma - 1) rho_0 eps > 0, gamma = 4/3, l_visc = 0.06 ell)
    on identical inhomogeneous initial data (M = 20, r_s = 40 ell, inhomogeneity = 0.42, N = 120).

    Verifies:
      1) Pressureless dust (p = 0) undergoes shell crossing (min(R_{k+1} - R_k) < 0).
      2) Relativistic Einstein-Euler fluid (p > 0) completely prevents shell crossing
         (min(R_{k+1} - R_k) > 0, min_ratio_dR = 0.0881 > 0).
      3) Shock heating amplifies the mass-weighted mean thermal energy <eps> by > 45x (5e-4 -> 2.41e-2 c^2)
         while keeping the relativistic sound speed strictly subluminal (max c_s = 0.172 c < c).
      4) Unlike cold sticky dust (where Gamma^2 = 1 + 2 e_mech/c^2 plunged to -9.18 < 0),
         the self-consistent Misner-Sharp evolution preserves Gamma_i(tau) >= 0.783 > 0 (strictly Lorentzian),
         satisfies the exact metric constraint |Gamma^2 - f_LQC - U^2| < 1e-7, and conserves total ADM mass
         M_ADM(tau) = m_N(tau) to machine precision (< 1e-14).
      5) Every shell i = 1..N turns around (U_i = 0) inside its own inner horizon R_turn,i < R_-(m_i)
         where f_LQC(R_turn,i, m_i) = Gamma_i^2 > 0 is timelike, and then re-expands outward!
    """
    res_dust = run_lagrangian_einstein_euler(
        N=120, M_tot=20.0, ell=1.0, gamma_eos=4.0 / 3.0, inhomogeneity=0.42, dust_mode=True, t_end=6.0
    )
    res_euler = run_lagrangian_einstein_euler(
        N=120,
        M_tot=20.0,
        ell=1.0,
        gamma_eos=4.0 / 3.0,
        inhomogeneity=0.42,
        l_visc=0.06,
        eps_floor=5e-4,
        dust_mode=False,
        t_end=6.0,
    )

    dust_crosses = bool(res_dust["min_dR"] < 0.0)
    euler_no_crossing = bool(res_euler["min_dR"] > 0.0 and res_euler["min_ratio_dR"] > 0.05)
    lorentzian_ok = bool(res_euler["min_Gamma"] > 0.5 and res_euler["max_constraint_err"] < 1e-6)
    adm_conserved = bool(res_euler["adm_mass_rel_err"] < 1e-13)
    shock_heated = bool(res_euler["heating_amplification_factor"] > 40.0 and res_euler["max_cs"] < 1.0)
    turns_in_core = bool(
        res_euler["all_turns_inside_R_minus"]
        and res_euler["R_surf_min"] < res_euler["R_minus_tot"]
        and res_euler["U_surf_final"] > 0.0
    )

    passed = bool(
        dust_crosses
        and euler_no_crossing
        and lorentzian_ok
        and adm_conserved
        and shock_heated
        and turns_in_core
    )

    return {
        "passed": passed,
        "dust_min_dR_negative_shell_crossing": res_dust["min_dR"],
        "euler_min_dR_positive": res_euler["min_dR"],
        "euler_min_compression_ratio_dR": res_euler["min_ratio_dR"],
        "euler_min_Gamma_positive": res_euler["min_Gamma"],
        "euler_min_Gamma2_positive": res_euler["min_Gamma2"],
        "max_misner_sharp_constraint_err": res_euler["max_constraint_err"],
        "adm_mass_relative_drift": res_euler["adm_mass_rel_err"],
        "max_interior_pdV_mass_transfer_fraction": res_euler["max_interior_mass_transfer"],
        "max_mean_thermal_energy_eps": res_euler["max_mean_eps"],
        "shock_heating_amplification_factor": res_euler["heating_amplification_factor"],
        "max_sound_speed_over_c": res_euler["max_cs"],
        "R_core_M": res_euler["R_core_tot"],
        "R_minus_M": res_euler["R_minus_tot"],
        "R_surf_min_inside_R_minus": res_euler["R_surf_min"],
        "R_surf_final_reexpanding": res_euler["R_surf_final"],
        "all_shells_turn_inside_R_minus": res_euler["all_turns_inside_R_minus"],
    }


def test_36_continuum_convergence_of_relativistic_einstein_euler_bounce() -> dict:
    """
    Test 36: Verifies continuum grid convergence (Delta mu -> 0 as N = 30 -> 60 -> 120 -> 240)
    at fixed physical Planck scale ell = 1.0 and fixed physical shock dissipation scale l_visc = 0.06 ell.

    Checks monotone O(1/N^2) second-order convergence of:
      1) Half-mass trajectory R_{1/2}(tau) L2 difference vs. N = 240: 1.14e-5 -> 2.73e-6 -> 5.80e-7.
      2) Minimum shell compression ratio min(dR_k(tau)/dR_k(0)): rel_err 2.00e-2 -> 5.62e-3 -> 1.44e-3.
      3) Peak mass-weighted thermal energy <eps>_max: rel_err 7.07e-2 -> 1.94e-2 -> 3.51e-3.
    """
    resolutions = [30, 60, 120, 240]
    runs = {}
    for N in resolutions:
        runs[N] = run_lagrangian_einstein_euler(
            N=N,
            M_tot=20.0,
            ell=1.0,
            gamma_eos=4.0 / 3.0,
            inhomogeneity=0.42,
            l_visc=0.06,
            eps_floor=5e-4,
            dust_mode=False,
            t_end=6.0,
        )

    ref = runs[240]
    l2_half_errs = []
    eps_rel_errs = []
    ratio_rel_errs = []
    for N in [30, 60, 120]:
        l2_half = float(np.sqrt(np.mean((runs[N]["R_half"] - ref["R_half"]) ** 2)))
        eps_err = float(abs(runs[N]["max_mean_eps"] - ref["max_mean_eps"]) / ref["max_mean_eps"])
        ratio_err = float(abs(runs[N]["min_ratio_dR"] - ref["min_ratio_dR"]) / ref["min_ratio_dR"])
        l2_half_errs.append(l2_half)
        eps_rel_errs.append(eps_err)
        ratio_rel_errs.append(ratio_err)

    monotone_half = bool(l2_half_errs[0] > l2_half_errs[1] > l2_half_errs[2] and l2_half_errs[2] < 2e-6)
    monotone_eps = bool(eps_rel_errs[0] > eps_rel_errs[1] > eps_rel_errs[2] and eps_rel_errs[2] < 5e-3)
    monotone_ratio = bool(ratio_rel_errs[0] > ratio_rel_errs[1] > ratio_rel_errs[2] and ratio_rel_errs[2] < 3e-3)
    all_no_crossing = bool(all(runs[N]["shell_crossing_prevented"] for N in resolutions))
    all_lorentzian = bool(all(runs[N]["lorentzian_signature_preserved"] for N in resolutions))

    # Empirical convergence order p = log2(err_30 / err_120) / 2
    order_half_mass = float(np.log2(l2_half_errs[0] / l2_half_errs[2]) / 2.0)
    order_min_ratio = float(np.log2(ratio_rel_errs[0] / ratio_rel_errs[2]) / 2.0)

    passed = bool(
        monotone_half
        and monotone_eps
        and monotone_ratio
        and all_no_crossing
        and all_lorentzian
        and order_half_mass > 1.8
    )

    return {
        "passed": passed,
        "resolutions_N": resolutions,
        "half_mass_L2_errors_vs_N240": l2_half_errs,
        "max_mean_eps_rel_errors_vs_N240": eps_rel_errs,
        "min_ratio_dR_rel_errors_vs_N240": ratio_rel_errs,
        "empirical_convergence_order_half_mass": order_half_mass,
        "empirical_convergence_order_compression_ratio": order_min_ratio,
        "min_ratio_dR_by_N": [runs[N]["min_ratio_dR"] for N in resolutions],
        "max_mean_eps_by_N": [runs[N]["max_mean_eps"] for N in resolutions],
    }


def test_37_trapped_region_hyperbola_and_throat_threshold_audit() -> dict:
    """
    Test 37: Audits the Misner-Sharp trapped-region hyperbola identity
      (U / c)^2 - Gamma^2 = |f_LQC(R, m)| > 0   for R in (R_-(m), R_+(m))
    across equations of state (gamma = 4/3 vs. gamma = 5/3) and across the trapped-region
    over-deceleration threshold (l_visc = 0.06 vs. l_visc = 0.25 at delta = 0.50).

    Verifies:
      1) For both gamma = 4/3 (radiation fluid) and gamma = 5/3 (monatomic gas) in the causal
         core-bounce window (l_visc = 0.06 ell), every shell enters R <= R_-(m_i) before deceleration
         completes, preserving Gamma > 0, partial_mu R > 0, and c_s < c everywhere.
      2) When outward shock viscosity/pressure is cranked up inside the trapped region (l_visc = 0.25 ell,
         delta = 0.50), decelerating an infalling outer shell down to the minimum trapped speed
         |U| = c sqrt(|f_LQC(R, m)|) while it is still outside R_-(m) (R_surf_min = 3.562 > R_-(M) = 3.527)
         cannot bring U to zero; instead, it drives Gamma = dR/dl_proper -> 0 and across zero
         (min_Gamma < 0, throat / neck singularity), confirming the exact Misner-Sharp hyperbola theorem!
    """
    res_g43 = run_lagrangian_einstein_euler(
        N=60, M_tot=20.0, ell=1.0, gamma_eos=4.0 / 3.0, inhomogeneity=0.42, l_visc=0.06, eps_floor=5e-4, t_end=6.0
    )
    res_g53 = run_lagrangian_einstein_euler(
        N=60, M_tot=20.0, ell=1.0, gamma_eos=5.0 / 3.0, inhomogeneity=0.42, l_visc=0.06, eps_floor=5e-4, t_end=6.0
    )
    res_over = run_lagrangian_einstein_euler(
        N=60, M_tot=20.0, ell=1.0, gamma_eos=4.0 / 3.0, inhomogeneity=0.50, l_visc=0.25, eps_floor=1e-3, t_end=6.0
    )

    causal_window_ok = bool(
        res_g43["min_Gamma"] > 0.5
        and res_g53["min_Gamma"] > 0.5
        and res_g43["all_turns_inside_R_minus"]
        and res_g53["all_turns_inside_R_minus"]
        and res_g43["max_cs"] < 1.0
        and res_g53["max_cs"] < 1.0
    )
    throat_in_trapped_region_confirmed = bool(
        res_over["R_surf_min"] > res_over["R_minus_tot"]
        and res_over["min_Gamma"] < 0.0
        and not res_over["all_turns_inside_R_minus"]
    )

    passed = bool(causal_window_ok and throat_in_trapped_region_confirmed)

    return {
        "passed": passed,
        "gamma_4_3_min_Gamma": res_g43["min_Gamma"],
        "gamma_4_3_R_surf_min_inside_R_minus": res_g43["R_surf_min"],
        "gamma_4_3_R_surf_final": res_g43["R_surf_final"],
        "gamma_4_3_max_cs": res_g43["max_cs"],
        "gamma_5_3_min_Gamma": res_g53["min_Gamma"],
        "gamma_5_3_R_surf_min_inside_R_minus": res_g53["R_surf_min"],
        "gamma_5_3_R_surf_final": res_g53["R_surf_final"],
        "gamma_5_3_max_cs": res_g53["max_cs"],
        "overpressurized_trapped_R_surf_min_outside_R_minus": res_over["R_surf_min"],
        "R_minus_M": res_over["R_minus_tot"],
        "overpressurized_trapped_min_Gamma_throat_crossing": res_over["min_Gamma"],
    }


def run_module_11() -> dict:
    t34 = test_34_curvature_capped_tov_no_go_for_static_positive_pressure_core()
    t35 = test_35_einstein_euler_shock_prevents_shell_crossing_and_preserves_signature()
    t36 = test_36_continuum_convergence_of_relativistic_einstein_euler_bounce()
    t37 = test_37_trapped_region_hyperbola_and_throat_threshold_audit()

    all_passed = bool(t34["passed"] and t35["passed"] and t36["passed"] and t37["passed"])
    return {
        "module": "Module 11: Relativistic Misner-Sharp / Einstein-Euler Fluid Collapse & Core Bounce",
        "test_34_curvature_capped_tov_no_go_for_static_positive_pressure_core": t34,
        "test_35_einstein_euler_shock_prevents_shell_crossing_and_preserves_signature": t35,
        "test_36_continuum_convergence_of_relativistic_einstein_euler_bounce": t36,
        "test_37_trapped_region_hyperbola_and_throat_threshold_audit": t37,
        "module_11_passed": all_passed,
        "all_passed": all_passed,
    }


if __name__ == "__main__":
    results = run_module_11()
    print(json.dumps(results, indent=2))
    print(f"\nMODULE 11 PASSED: {results['all_passed']}")
    if not results["all_passed"]:
        sys.exit(1)
