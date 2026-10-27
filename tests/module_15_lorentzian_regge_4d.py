"""
Module 15: Linearised LORENTZIAN Regge calculus on a 4D Freudenthal lattice (Tests 56-59).

Postulates: piecewise-flat Lorentzian spacetime (flat 4-simplices) + the Regge action
S = sum_t |A_t| eps_t. Nothing about Einstein's equations, the graviton, its speed or its
polarisations is put in; the relative signs of timelike/spacelike hinge terms are fixed by the
Lorentzian Schlafli identity of each simplex.

  - Test 56: Lorentzian Schlafli hinge weights, exact gauge invariance (vertex translations),
             contracted Bianchi identity, zero-mode content (incl. the hyperdiagonal artefact).
  - Test 57: Propagating modes over the whole temporal Brillouin zone: exactly one doubly
             degenerate mode (2 polarisations), speed -> c at O(k^2), independent of dt/dx and of
             direction, transverse-traceless; no real modes in Euclidean signature (control).
  - Test 58: Slice-by-slice Lorentzian Regge evolution: exact plane waves for 2000 steps, no
             exponentially growing mode, real-space graviton wave packet moving at the lattice
             group velocity (-> c).
  - Test 59: Anamorphic BCC / Sommerville disphenoid comparison (gamma = 0 vs gamma = -1/3):
             compressing along (1,1,1) removes the 5th hyperdiagonal zero mode and cuts
             directional dispersion anisotropy by >5.4x, but turns the 5th mode into a UV cutoff
             optical mode and pushes unweighted stepper |lambda|_max above 1, confirming why
             gamma = 0 is kept as the time-evolution baseline.
Scope: linearised theory on flat space only; no nonlinear 3+1D evolution.
"""

from __future__ import annotations
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lorentzian_regge_4d import (
    audit_lorentzian_regge_structure,
    audit_graviton_dispersion,
    audit_regge_time_evolution,
    audit_anamorphic_bcc_lattice_comparison,
)


def test_56_lorentzian_regge_schlafli_gauge_bianchi() -> dict:
    res = audit_lorentzian_regge_structure(alpha=0.5)
    passed = bool(
        res["max_lorentzian_schlafli_rel_residual"] < 1e-14
        and res["max_abs_hinge_weight_minus_1"] < 1e-12
        and res["all_timelike_hinge_weights_plus_1"]
        and res["euclidean_control_max_abs_weight_minus_1"] < 1e-12
        and res["max_hermiticity_error"] < 1e-13
        and res["max_gauge_invariance_error_H_g"] < 1e-13
        and res["max_bianchi_identity_error_gH_H"] < 1e-13
        and res["null_space_dimensions_seen"] == [5]
        and res["max_hyperdiagonal_hessian_column_rel"] < 1e-13
        and res["min_hyperdiagonal_deficit_coupling_rel"] > 1e-2
        and res["max_gauge_deficit_change_rel"] < 1e-13
    )
    return {"passed": passed, **res}


def test_57_lorentzian_regge_graviton_dispersion_and_polarisations() -> dict:
    res = audit_graviton_dispersion()
    passed = bool(
        res["exactly_one_doubly_degenerate_mode_in_full_kt_zone"]
        and res["max_abs_one_minus_v_at_smallest_k"] < 1e-4
        and all(1.95 < p < 2.05 for p in res["convergence_orders_last_pair"])
        and min(res["graviton_polarisation_TT_fractions_k005_z"]) > 1 - 1e-6
        and max(res["graviton_non_metric_residuals_k005_z"]) < 1e-3
        and res["euclidean_control_real_crossings"] == 0
    )
    return {"passed": passed, **res}


def test_58_lorentzian_regge_time_evolution_and_wave_packet() -> dict:
    res = audit_regge_time_evolution(alpha=0.5)
    passed = bool(
        res["plane_wave_2000_steps_max_rel_error"] < 1e-6
        and res["plane_wave_max_equation_residual"] < 1e-11
        and res["transfer_map_max_abs_eigenvalue"] < 1 + 1e-8
        and res["transfer_map_unit_modulus_only_graviton_or_residual_gauge"]
        and res["transfer_map_norm_growth_T4000_over_T1000_linear_bound_4"] < 4.5
        and res["packet_max_equation_residual"] < 1e-11
        and abs(res["packet_group_velocity_over_c"] - res["dispersion_group_velocity_over_c"]) < 1e-3
        and abs(res["packet_group_velocity_over_c"] - 1.0) < 1e-2
        and abs(res["packet_norm_ratio_end_over_start"] - 1.0) < 1e-6
    )
    return {"passed": passed, **res}


def test_59_anamorphic_bcc_lattice_comparison() -> dict:
    res = audit_anamorphic_bcc_lattice_comparison(alpha=0.5, kmag=0.2)
    g0 = res["unstretched_gamma_0"]
    gbcc = res["bcc_disphenoid_gamma_minus_1_3"]
    passed = bool(
        g0["spatial_edge_length_sq_set"] == [1.0, 2.0, 3.0]
        and gbcc["spatial_edge_length_sq_set"] == [1.0, 1.333333]
        and g0["max_schlafli_rel_residual"] < 1e-14
        and gbcc["max_schlafli_rel_residual"] < 1e-14
        and g0["gauge_invariance_error"] < 1e-13
        and gbcc["gauge_invariance_error"] < 1e-13
        and g0["bianchi_identity_error"] < 1e-13
        and gbcc["bianchi_identity_error"] < 1e-13
        and res["hyperdiagonal_artefact_removed_by_bcc_stretch"]
        and res["anisotropy_reduction_factor"] > 5.0
        and res["axis_and_body_diagonal_speeds_equalised_at_bcc"]
        and res["unstretched_has_zero_spurious_uv_modes"]
        and res["bcc_stretch_introduces_uv_cutoff_mode"]
        and g0["n_neg_eigenvalues_static_limit"] == 4
        and g0["n_neg_eigenvalues_zone_edge"] == 2
        and gbcc["n_neg_eigenvalues_static_limit"] == 4
        and gbcc["n_neg_eigenvalues_zone_edge"] == 1
        and res["unstretched_stepper_spectrally_bounded"]
        and res["bcc_unweighted_stepper_exceeds_unit_circle"]
    )
    return {"passed": passed, **res}


def run_module_15() -> dict:
    t56 = test_56_lorentzian_regge_schlafli_gauge_bianchi()
    t57 = test_57_lorentzian_regge_graviton_dispersion_and_polarisations()
    t58 = test_58_lorentzian_regge_time_evolution_and_wave_packet()
    t59 = test_59_anamorphic_bcc_lattice_comparison()
    return {
        "test_56_lorentzian_regge_schlafli_gauge_bianchi": t56,
        "test_57_lorentzian_regge_graviton_dispersion_and_polarisations": t57,
        "test_58_lorentzian_regge_time_evolution_and_wave_packet": t58,
        "test_59_anamorphic_bcc_lattice_comparison": t59,
        "module_15_passed": bool(t56["passed"] and t57["passed"] and t58["passed"] and t59["passed"]),
    }


if __name__ == "__main__":
    print(json.dumps(run_module_15(), indent=2, default=str))
