"""
Module 14: Non-GR Grid Mechanisms Audit & Microscopic Regge Bar-Grid Foundation (Tests 48-55).

Tests five distinct physical mechanisms on the discrete lattice (Tests 48-52) plus the three
foundational pieces of Outcome 5 (Simplicial Regge Calculus + Compact Holonomy, Tests 53-55):
  - Test 48: Pure 3D 18-Point Hooke's-Law Vector Spring Relaxation (Eshelby 1/r^3 Inclusion vs. 1/r Gravity)
  - Test 49: Crystalline Topological Disclination Theory (Kroner Incompatibility eta_{ij} == G^{(3)}_{ij})
  - Test 50: 3D 18-Point Lattice Poisson + Bernoulli Superfluid Condensate Flow & Bopp-Podolsky UV Audit
  - Test 51: Sakharov One-Loop Quantum Zero-Point Energy (Induced Gravity) on the 18-Point Brillouin Zone
  - Test 52: Exact 3D Simplicial Regge Calculus (Cayley-Menger, Schlafli Identity, Exterior Variational Stationarity & Jacobson Area Deficit)
  - Test 53: Bounded Trigonometric Holonomy / Regge Action Deriving the Planck Density Cap (rho_max) and Core Bounce (R_core) via Extrinsic-Curvature Bragg Reflection
  - Test 54: Variational Regge + Matter Derivation of Karl Schwarzschild's (1916) Interior Star and the Regular de Sitter Core
  - Test 55: Full 3D Periodic Kuhn-Regge Complex — Nonlinear Diffeomorphism Gauge Invariance (dS_Regge/du_i == 0) vs. Spin-2 Curvature Stiffness
"""

from __future__ import annotations
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_gr_grid_mechanisms_audit import (
    audit_3d_hooke_eshelby_vs_gravity,
    audit_kroner_disclination_incompatibility_3d,
    audit_18pt_poisson_bernoulli_and_biharmonic_uv,
    audit_sakharov_one_loop_induced_gravity_18pt,
    audit_simplicial_regge_variational_and_jacobson_deficit,
    audit_bounded_holonomy_regge_core_bounce,
    audit_variational_regge_interior_star_and_desitter_core,
    audit_3d_kuhn_regge_diffeomorphism_invariance_vs_spin2,
)


def test_48_3d_18pt_hooke_eshelby_inclusion_vs_gravity() -> dict:
    """
    Test 48: Pure 3D 18-Point Hooke's-Law Vector Spring Relaxation (Eshelby 1/r^3 vs. 1/r Gravity).
    """
    res = audit_3d_hooke_eshelby_vs_gravity(N=64)
    passed = bool(
        res["aniso_elastostatic_k1_eq_k2_18pt"] < 1e-3
        and res["aniso_elastostatic_axial_only_6pt"] > 1.0
        and abs(res["cL_over_cT_isotropic_truss"] - res["cL_over_cT_exact_sqrt3"]) < 1e-3
        and -2.2 < res["eshelby_u_r_power_law_slope"] < -1.8
        and res["eshelby_eps_rr_power_law_slope"] < -2.8
        and res["max_linearised_einstein_curvature_from_elastic_displacement"] < 1e-14
    )
    return {
        "passed": passed,
        **res,
    }


def test_49_kroner_crystalline_disclination_incompatibility_3d() -> dict:
    """
    Test 49: Crystalline Topological Disclination Theory (Kroner 1981; Kleinert 1987).
    """
    res = audit_kroner_disclination_incompatibility_3d(N=48)
    passed = bool(
        res["max_eta_norm"] > 1e-2
        and res["max_diff_kroner_eta_vs_linearised_3d_einstein_G3"] < 1e-14
        and res["max_disclination_continuity_violation_div_eta"] < 1e-14
        and abs(res["disclination_spatial_strain_power_law_slope"] - (-1.0)) < 0.02
    )
    return {
        "passed": passed,
        **res,
    }


def test_50_18pt_poisson_bernoulli_condensate_and_biharmonic_uv() -> dict:
    """
    Test 50: 3D 18-Point Lattice Poisson + Bernoulli Superfluid Condensate Flow & Bopp-Podolsky UV Audit.
    """
    res = audit_18pt_poisson_bernoulli_and_biharmonic_uv(N=48)
    passed = bool(
        res["isotropy_improvement_18pt_over_6pt"] > 20.0
        and abs(res["bernoulli_pg_1pn_deflection_over_newtonian"] - 2.0) < 0.01
        and res["unregularised_to_biharmonic_origin_ratio"] > 4.0
        and abs(res["bopp_podolsky_small_r_kretschmann_slope"] - (-2.0)) < 0.02
    )
    return {
        "passed": passed,
        **res,
    }


def test_51_sakharov_one_loop_zero_point_induced_gravity_18pt() -> dict:
    """
    Test 51: Sakharov (1967) One-Loop Quantum Zero-Point Energy (Induced Gravity) on the 18-Point Lattice.
    """
    res = audit_sakharov_one_loop_induced_gravity_18pt(N_bz=48, N_3d=24)
    passed = bool(
        0.9 < res["dimensionless_vacuum_energy_18pt"] < 1.1
        and res["sakharov_c1_coefficient_18pt"] > 0.005
        and res["induced_tt_curvature_stiffness"] > 0.1
        and abs(res["tt_wave_quadratic_scaling_ratio"] - 4.0) < 1e-3
    )
    return {
        "passed": passed,
        **res,
    }


def test_52_simplicial_regge_schlafli_and_variational_stationarity() -> dict:
    """
    Test 52: Exact 3D Simplicial Regge Calculus (Cayley-Menger, Schlafli Identity & Exterior Variational Stationarity).
    """
    res = audit_simplicial_regge_variational_and_jacobson_deficit(
        N_list=(50, 100, 200)
    )
    passed = bool(
        abs(res["flat_space_6_tetrahedra_deficit_angle"]) < 1e-12
        and res["max_schlafli_identity_residual_3d_tetrahedron"] < 1e-7
        and res["errors_A"][-1] < 1e-5
        and res["errors_a"][-1] < 1e-5
        and 1.95 < res["convergence_order_A"] < 2.05
        and 1.95 < res["convergence_order_a"] < 2.05
        and res["max_jacobson_area_deficit_error"] < 1e-5
    )
    return {
        "passed": passed,
        **res,
    }


def test_53_bounded_holonomy_regge_core_bounce_bragg_reflection() -> dict:
    """
    Test 53: Bounded Trigonometric Holonomy / Regge Action Deriving the Planck Density Cap (rho_max)
    and Core Bounce (R_core) via Extrinsic-Curvature Bragg Reflection at theta_plaq = -pi / 2.
    """
    res = audit_bounded_holonomy_regge_core_bounce(m=5.0, ell=1.0)
    passed = bool(
        res["rel_err_R_core"] < 1e-6
        and res["rel_err_rho_max"] < 1e-6
        and res["abs_err_theta_plaq_bounce"] < 1e-3
        and res["max_hamiltonian_constraint_drift"] < 1e-6
        and res["unbounded_continuum_H_collapsed_to_singularity"]
    )
    return {
        "passed": passed,
        **res,
    }


def test_54_variational_regge_interior_star_and_desitter_core() -> dict:
    """
    Test 54: Pure Numerical Variational Stationarity of the Discrete Regge + Matter Action
    for (A) the Regular de Sitter Planck Core (p_r = -rho_max c^2) and (B) Karl Schwarzschild's
    (1916) Constant-Density Interior Star (via the Schutz-Brown Relativistic Fluid Action).
    """
    res = audit_variational_regge_interior_star_and_desitter_core(
        N_list=(50, 100, 200)
    )
    passed = bool(
        res["desitter_core_errors_A"][-1] < 2e-5
        and res["desitter_core_errors_a"][-1] < 2e-5
        and 1.95 < res["desitter_core_order_A"] < 2.05
        and 1.95 < res["desitter_core_order_a"] < 2.05
        and res["schwarzschild_star_errors_A"][-1] < 1e-5
        and res["schwarzschild_star_errors_a"][-1] < 1e-5
        and res["schwarzschild_star_errors_p"][-1] < 1e-4
        and 1.95 < res["schwarzschild_star_order_A"] < 2.05
        and 1.95 < res["schwarzschild_star_order_p"] < 2.05
    )
    return {
        "passed": passed,
        **res,
    }


def test_55_3d_kuhn_regge_diffeomorphism_invariance_vs_spin2() -> dict:
    """
    Test 55: Full 3D Periodic Kuhn-Triangulated Regge Complex (384 Tetrahedra, 448 Edges) —
    Exact Nonlinear Diffeomorphism Invariance on Vertex Displacements u_i (deficit = 0,
    dS_Regge/du_i = 0 vs. large Hooke force) and Quadratic Curvature Response to Incompatible
    3D Transverse-Traceless Spin-2 Metric Waves.
    """
    res = audit_3d_kuhn_regge_diffeomorphism_invariance_vs_spin2(L=4)
    passed = bool(
        res["flat_space_max_deficit"] < 1e-13
        and res["diffeo_displaced_max_deficit"] < 1e-13
        and res["diffeo_regge_action_shift"] < 1e-11
        and res["diffeo_hooke_energy_shift"] > 0.1
        and res["diffeo_regge_vertex_force_norm"] < 1e-7
        and res["diffeo_hooke_vertex_force_norm"] > 0.1
        and res["tt_spin2_wave_max_deficit"] > 1e-2
        and abs(res["tt_spin2_regge_quadratic_ratio"] - 4.0) < 1e-2
    )
    return {
        "passed": passed,
        **res,
    }


def run_module_14() -> dict:
    t48 = test_48_3d_18pt_hooke_eshelby_inclusion_vs_gravity()
    t49 = test_49_kroner_crystalline_disclination_incompatibility_3d()
    t50 = test_50_18pt_poisson_bernoulli_condensate_and_biharmonic_uv()
    t51 = test_51_sakharov_one_loop_zero_point_induced_gravity_18pt()
    t52 = test_52_simplicial_regge_schlafli_and_variational_stationarity()
    t53 = test_53_bounded_holonomy_regge_core_bounce_bragg_reflection()
    t54 = test_54_variational_regge_interior_star_and_desitter_core()
    t55 = test_55_3d_kuhn_regge_diffeomorphism_invariance_vs_spin2()

    all_passed = bool(
        t48["passed"]
        and t49["passed"]
        and t50["passed"]
        and t51["passed"]
        and t52["passed"]
        and t53["passed"]
        and t54["passed"]
        and t55["passed"]
    )
    return {
        "module": "Module 14: Non-GR Grid Mechanisms Audit & Microscopic Regge Bar-Grid Foundation (Tests 48-55)",
        "test_48_3d_18pt_hooke_eshelby_inclusion_vs_gravity": t48,
        "test_49_kroner_crystalline_disclination_incompatibility_3d": t49,
        "test_50_18pt_poisson_bernoulli_condensate_and_biharmonic_uv": t50,
        "test_51_sakharov_one_loop_zero_point_induced_gravity_18pt": t51,
        "test_52_simplicial_regge_schlafli_and_variational_stationarity": t52,
        "test_53_bounded_holonomy_regge_core_bounce_bragg_reflection": t53,
        "test_54_variational_regge_interior_star_and_desitter_core": t54,
        "test_55_3d_kuhn_regge_diffeomorphism_invariance_vs_spin2": t55,
        "module_14_passed": all_passed,
    }


if __name__ == "__main__":
    out = run_module_14()
    print(json.dumps(out, indent=2))
