"""
Module 13: Discretised Spherical Einstein/TOV Constraints in Anamorphic Variables & 3D 18-Point
Linearised Spin-2 Lattice Audit (Tests 43-47).

Scope: Einstein's spherically symmetric constraint equations (G^t_t, G^r_r, hydrostatic balance) are
INPUT; they are discretised at O(1/N^2) in anamorphic variables without a closed-form metric ansatz.
  - Test 43: Discretised exterior Schwarzschild (A_i, a_i), vacuum reciprocity a_i * A_i -> 1, and 1PN
             light bending (2.000 vs. 1.000 for 1911 time-only), O(1/N^2) across N = 200 -> 1600.
  - Test 44: Discretised Schwarzschild (1916) constant-density interior (A_i, a_i, p_i), O(1/N^2) across
             N = 100 -> 800, and the constant-density central-pressure divergence as 2GM/(c^2 R) -> 8/9
             (the case that saturates Buchdahl's bound; Buchdahl's general theorem is not tested).
  - Test 45: Painleve-Gullstrand coordinate map J_i = -c sqrt(Q_i / R_i) applied to the discretely
             integrated curvature charge (the identity itself holds by definition; only the quadrature
             error of Q_i is a genuine check).
  - Test 46: 18-bar rank-6 strain reconstruction (genuine kinematic result); TT/energy/Bianchi checks
             hold by construction (see test docstring).
  - Test 47: Continuum Regge-Wheeler formula (spin factor 1 - s^2 input) on the discretised background.
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergent_anamorphic_gr import (
    solve_emergent_exterior_and_light_bending,
    solve_emergent_interior_schwarzschild_star,
    solve_emergent_anamorphic_to_river_duality,
    audit_3d_anamorphic_spin2_gravitational_waves,
    audit_emergent_spin2_regge_wheeler_vs_scalar,
)


def test_43_discretised_exterior_schwarzschild_and_light_bending_from_rho() -> dict:
    """
    Test 43: Starts from a compact stellar density profile rho(R) on [0, R_star] and integrates the
    discretised G^t_t equation for the bar aspect ratio A_i = b_{r,i}/b_{perp,i} and the discretised
    G^r_r equation for the clock rate a_i across N = 200, 400, 800, 1600 (Einstein's equations are input;
    no closed-form metric is input).
    """
    resolutions = [200, 400, 800, 1600]
    reciprocity_errs = []
    a_errs = []
    last_res = None

    for N in resolutions:
        res = solve_emergent_exterior_and_light_bending(
            N=N, M_tot=1.0, R_star=6.0, R_max=80000.0, b_impact=1500.0
        )
        reciprocity_errs.append(res["max_vacuum_reciprocity_err_aA_minus_1"])
        a_errs.append(res["max_rel_err_a_exterior"])
        last_res = res

    orders_a = [
        float(np.log2(a_errs[k] / a_errs[k + 1])) for k in range(len(a_errs) - 1)
    ]
    mean_order = float(np.mean(orders_a))

    passed = bool(
        last_res is not None
        and last_res["r_s_rel_err"] < 1e-6
        and last_res["max_rel_err_A_exterior"] < 1e-6
        and last_res["max_rel_err_a_exterior"] < 1e-5
        and last_res["max_vacuum_reciprocity_err_aA_minus_1"] < 1e-5
        and mean_order > 1.9
        and abs(last_res["emergent_1PN_light_bending_after_2PN"] - 2.0) < 2e-3
        and abs(last_res["control_1911_time_only_coeff"] - 1.0) < 5e-3
    )

    return {
        "passed": passed,
        "resolutions_N": resolutions,
        "exterior_clock_rel_errors": a_errs,
        "vacuum_reciprocity_aA_minus_1_errors": reciprocity_errs,
        "empirical_convergence_order_mean": mean_order,
        "N1600_emergent_1PN_light_bending_raw": last_res["emergent_1PN_light_bending_raw"],
        "N1600_emergent_1PN_light_bending_after_2PN": last_res["emergent_1PN_light_bending_after_2PN"],
        "N1600_control_1911_time_only_coeff": last_res["control_1911_time_only_coeff"],
    }


def test_44_discretised_interior_schwarzschild_star_and_constant_density_8_9_divergence() -> dict:
    """
    Test 44: Solves the discretised TOV system (G^t_t + G^r_r + hydrostatic balance) inside a
    constant-density star (2GM/(c^2 R_star) = 0.5) across N = 100, 200, 400, 800, and scans the
    constant-density central-pressure divergence as compaction -> 8/9 (Buchdahl-saturating case only).
    """
    resolutions = [100, 200, 400, 800]
    p_errs = []
    a_errs = []
    last_res = None

    for N in resolutions:
        res = solve_emergent_interior_schwarzschild_star(
            N=N, compaction_rs_over_Rstar=0.5, R_star=10.0
        )
        p_errs.append(res["max_rel_err_p"])
        a_errs.append(res["max_err_a"])
        last_res = res

    orders_p = [
        float(np.log2(p_errs[k] / p_errs[k + 1])) for k in range(len(p_errs) - 1)
    ]
    mean_order_p = float(np.mean(orders_p))

    # Buchdahl limit scan as compaction 2GM/(c^2 R_star) -> 8/9 (0.888888...)
    buchdahl_scan = []
    for comp in [0.50, 0.75, 0.85, 0.885]:
        b_res = solve_emergent_interior_schwarzschild_star(
            N=600, compaction_rs_over_Rstar=comp, R_star=10.0
        )
        buchdahl_scan.append({
            "compaction_rs_over_Rstar": comp,
            "central_clock_a0_emergent": b_res["central_clock_a0_emergent"],
            "central_clock_a0_exact": b_res["central_clock_a0_exact"],
            "central_pressure_over_rho0_emergent": b_res["central_pressure_over_rho0_emergent"],
            "central_pressure_over_rho0_exact": b_res["central_pressure_over_rho0_exact"],
        })

    passed = bool(
        last_res is not None
        and last_res["max_err_A"] < 1e-14
        and last_res["max_err_a"] < 1e-6
        and last_res["max_rel_err_p"] < 1e-5
        and mean_order_p > 1.95
        and last_res["interior_aA_product_center"] < 1.0
        and buchdahl_scan[-1]["central_clock_a0_emergent"] < 0.02
        and buchdahl_scan[-1]["central_pressure_over_rho0_emergent"] > 10.0
    )

    return {
        "passed": passed,
        "resolutions_N": resolutions,
        "interior_pressure_rel_errors": p_errs,
        "interior_clock_abs_errors": a_errs,
        "empirical_convergence_order_mean": mean_order_p,
        "N800_interior_aA_product_center": last_res["interior_aA_product_center"],
        "buchdahl_limit_scan": buchdahl_scan,
    }


def test_45_painleve_gullstrand_coordinate_map_of_discrete_curvature_charge() -> dict:
    """
    Test 45: Applies the Painleve-Gullstrand coordinate definition J_i = -c sqrt(Q_i / R_i) to the
    discretely integrated curvature charge across all three regular-BH zones. The static/river identity
    1 - Q/R == 1 - J^2 holds BY DEFINITION (reported as pg_definition_identity_err_by_construction);
    the only genuine check is the quadrature accuracy of Q_i against the exact mass function.
    """
    res = solve_emergent_anamorphic_to_river_duality(N=2000, M_tot=50.0, ell=1.0, R_max=200.0)
    passed = bool(
        res["three_zones_emerged"]
        and res["pg_definition_identity_err_by_construction"] < 1e-14
        and res["max_emergent_vs_exact_metric_err"] < 1e-5
        and res["R_minus_emergent"] < res["R_core"] < res["R_plus_emergent"]
    )
    return {
        "passed": passed,
        **res,
    }


def test_46_18pt_rank6_strain_tt_evolution_and_by_construction_bianchi() -> dict:
    """
    Test 46: 3D 18-point lattice audit for symmetric 3x3 metric perturbations h_ij.
      GENUINE (non-trivial) results:
      1) 18-Bar Strain Reconstruction: the 6 axial + 12 face-diagonal bars have full rank 6 on symmetric
         3x3 tensors (reconstruction < 1e-15), whereas a 6-bar cubic truss has rank 3 (blind to x-shear).
         This is kinematic only: the evolution below does not use the bars.
      2) Scalar 18-pt vs 6-pt dispersion anisotropy at a single wavenumber |k| ell = pi/4 (ratio is
         k-dependent and is a property of the scalar stencil, not specific to gravity).
      BY CONSTRUCTION (reported, not evidence):
      3) (h_+, h_x) cos/sin(2 theta) law (property of the polarisation tensors).
      4) TT trace/divergence, delta G^0_0 == 0, delta R_ij[h^TT] == -(1/2) Delta_18 h^TT, Cayley energy
         (Fourier-space projection and mode-by-mode evolution with Delta_18).
      5) Real-space D_j G_ij[h] == 0 and gauge-mode Ricci == 0 with Delta = sum_k D_k D_k: identities for ANY
         commuting difference operators. The "mixed" anomaly only measures Delta_18 != sum_k D_k D_k; note
         that sum_k D_k D_k has spurious zone-boundary zero modes, and the evolution itself uses Delta_18.
    """
    res = audit_3d_anamorphic_spin2_gravitational_waves(L=16, n_steps=80, dt=0.15)
    passed = bool(
        res["rank_6bar_axial_only"] == 3
        and res["rank_18bar_with_face_diagonals"] == 6
        and res["max_18bar_strain_reconstruction_err"] < 1e-14
        and res["max_spin2_helicity_rotation_err"] < 1e-14
        and res["quadrupole_flip_at_pi_over_2_err"] < 1e-14
        and res["quadrupole_invariant_at_pi_err"] < 1e-14
        and res["polarization_orthogonality_err"] < 1e-15
        and res["ricci_tt_wave_identity_err"] < 1e-13
        and res["linearized_hamiltonian_constraint_tt_max"] < 1e-13
        and res["real_vs_spectral_lap18_err"] < 1e-13
        and res["cayley_80_steps_energy_rel_drift"] < 1e-13
        and res["final_tt_trace_max"] < 1e-13
        and res["final_tt_div_max"] < 1e-13
        and res["final_tensor_symmetry_err"] < 1e-14
        and res["real_space_consistent_bianchi_non_tt_max"] < 1e-14
        and res["real_space_mixed_lap18_bianchi_anomaly_non_tt"] > 1e-3
        and res["real_space_gauge_mode_ricci_max"] < 1e-14
        and res["gw_isotropy_improvement_factor_18pt_over_6pt"] > 15.0
    )
    return {
        "passed": passed,
        **res,
    }


def test_47_continuum_regge_wheeler_formula_on_discretised_background() -> dict:
    """
    Test 47: Evaluates the CONTINUUM quadrupole (l = 2) Regge-Wheeler formula, with the spin factor
    1 - s^2 (= -3 for s = 2, +1 for s = 0) taken from continuum GR perturbation theory, on the
    discretised background (R_i, A_i, a_i, Q_i). The < 1e-4 agreement measures the accuracy of the
    discretised background f = a/A; the lattice does not derive the spin-2 coupling.
    """
    res = audit_emergent_spin2_regge_wheeler_vs_scalar(N=1600, M_tot=1.0)
    passed = bool(
        res["max_rel_err_V_spin2_emergent_vs_exact"] < 1e-4
        and abs(res["Vmax_spin2_M2_emergent"] - res["Vmax_spin2_M2_exact"]) < 1e-4
        and abs(res["R_peak_spin2_over_M_emergent"] - res["R_peak_spin2_over_M_exact"]) < 0.1
        and res["spin2_barrier_lower_and_further_out_than_scalar"]
    )
    return {
        "passed": passed,
        **res,
    }


def run_module_13() -> dict:
    t43 = test_43_discretised_exterior_schwarzschild_and_light_bending_from_rho()
    t44 = test_44_discretised_interior_schwarzschild_star_and_constant_density_8_9_divergence()
    t45 = test_45_painleve_gullstrand_coordinate_map_of_discrete_curvature_charge()
    t46 = test_46_18pt_rank6_strain_tt_evolution_and_by_construction_bianchi()
    t47 = test_47_continuum_regge_wheeler_formula_on_discretised_background()

    all_passed = bool(
        t43["passed"]
        and t44["passed"]
        and t45["passed"]
        and t46["passed"]
        and t47["passed"]
    )
    return {
        "module": "Module 13: Discretised Spherical Einstein/TOV Constraints in Anamorphic Variables & 3D 18-Point Linearised Spin-2 Lattice Audit",
        "test_43_discretised_exterior_schwarzschild_and_light_bending_from_rho": t43,
        "test_44_discretised_interior_schwarzschild_star_and_constant_density_8_9_divergence": t44,
        "test_45_painleve_gullstrand_coordinate_map_of_discrete_curvature_charge": t45,
        "test_46_18pt_rank6_strain_tt_evolution_and_by_construction_bianchi": t46,
        "test_47_continuum_regge_wheeler_formula_on_discretised_background": t47,
        "module_13_passed": all_passed,
        "all_passed": all_passed,
    }


if __name__ == "__main__":
    results = run_module_13()
    print(json.dumps(results, indent=2))
    print(f"\nMODULE 13 PASSED: {results['all_passed']}")
    if not results["all_passed"]:
        sys.exit(1)
