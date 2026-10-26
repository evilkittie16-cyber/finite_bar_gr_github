"""
Module 13: Emergent General Relativity from the Discrete Anamorphic Grid Constraints (Tests 43-45).

Verifies Step 1 of producing General Relativity directly from the discrete anamorphic grid
equations (without putting Schwarzschild or Hayward metric formulas in by hand):
  - Test 43: Emergent exterior Schwarzschild metric (A_i, a_i), vacuum reciprocity a_i * A_i -> 1,
             and emergent 1PN + 2PN gravitational light bending (2.000 vs. 1.000 for 1911 time-only)
             with O(1/N^2) convergence across N = 200 -> 400 -> 800 -> 1600.
  - Test 44: Emergent Karl Schwarzschild (1916) interior constant-density star solution (A_i, a_i, p_i)
             from coupled discrete Hamiltonian + radial momentum + hydrostatic balance equations,
             O(1/N^2) convergence across N = 100 -> 200 -> 400 -> 800, and Buchdahl limit (2GM/c^2 R -> 8/9).
  - Test 45: Emergent Anamorphic-to-River Duality across black-hole horizons: when Q_i / R_i >= 1
             (inside [R_-, R_+] where static bars A_i become imaginary), transforming to a freely
             falling flat slice converts the emergent discrete curvature charge Q_i / R_i into the
             horizon-penetrating Painleve-Gullstrand river velocity J_i = -c sqrt(Q_i / R_i).
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


def test_43_emergent_exterior_schwarzschild_and_light_bending_from_discrete_constraints() -> dict:
    """
    Test 43: Starts ONLY from a compact stellar density profile rho(R) on [0, R_star] and solves
    the discrete Hamiltonian constraint for the radial-to-transverse bar aspect ratio A_i = b_{r,i}/b_{perp,i}
    and the discrete radial momentum constraint for the local clock rate a_i across N = 200, 400, 800, 1600.
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


def test_44_emergent_interior_schwarzschild_constant_density_star_and_buchdahl_limit() -> dict:
    """
    Test 44: Solves the coupled discrete Hamiltonian + radial momentum + hydrostatic balance
    equations inside a constant-density star (2GM/(c^2 R_star) = 0.5) across N = 100, 200, 400, 800,
    and verifies the Buchdahl limit as compaction approaches 8/9 = 0.8888...
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


def test_45_emergent_painleve_gullstrand_river_and_lqc_core_from_anamorphic_duality() -> dict:
    """
    Test 45: Verifies that when Q_i / R_i >= 1 (inside a black hole's trapped region [R_-, R_+]
    where the static radial bar stretch A_i = 1 / sqrt(1 - Q_i / R_i) becomes imaginary),
    transforming to a freely falling flat slice (Gamma_i = 1) converts the emergent discrete
    curvature charge Q_i / R_i directly into the horizon-penetrating Painleve-Gullstrand river
    velocity J_i = -c sqrt(Q_i / R_i) across all three zones.
    """
    res = solve_emergent_anamorphic_to_river_duality(N=2000, M_tot=50.0, ell=1.0, R_max=200.0)
    passed = bool(
        res["three_zones_emerged"]
        and res["duality_identity_err"] < 1e-14
        and res["max_emergent_vs_exact_metric_err"] < 1e-5
        and res["R_minus_emergent"] < res["R_core"] < res["R_plus_emergent"]
    )
    return {
        "passed": passed,
        **res,
    }


def test_46_3d_anamorphic_18pt_spin2_transverse_traceless_gravitational_waves() -> dict:
    """
    Test 46: Audits Step 2 (True Spin-2 Transverse-Traceless Gravitational Waves on the 3D 18-Point
    Anamorphic Lattice):
      1) 18-Bar Strain Reconstruction Identity: The 6 axial bars and 12 face-diagonal bars of the
         18-point stencil have full rank 6 on symmetric 3x3 tensors and uniquely reconstruct all
         6 components of h_ij to < 1e-15, whereas a 6-bar simple cubic truss has rank 3 (blind to
         off-diagonal x-polarized shear).
      2) Helicity-2 (Spin-2) Quadrupole Rotation Law: (h_+, h_x) transform as (cos 2*theta, sin 2*theta),
         reversing sign at theta = pi/2 and returning to +1 at theta = pi (< 1e-15).
      3) 3D Linearized Ricci Tensor & Symplectic Cayley Evolution: On the 16^3 3D 18-point lattice,
         the TT-projected 3x3 tensor field satisfies delta G^0_0 = -(1/2) delta R^k_k == 0 (< 1e-14)
         and delta R_ij[h^TT] == -(1/2) Delta_18 h_ij^TT (< 1e-14), and 80 symplectic Cayley steps
         preserve Tr(h) == 0 (< 1e-13), D_j h_ij == 0 (< 1e-13), h_ij == h_ji (< 1e-15), and
         Hamiltonian tensor wave energy (< 1e-13).
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
    )
    return {
        "passed": passed,
        **res,
    }


def test_47_emergent_spin2_regge_wheeler_barrier_vs_scalar_on_anamorphic_grid() -> dict:
    """
    Test 47: Constructs the quadrupole (l = 2) Regge-Wheeler potential V_RW^(s)(R) directly on the
    EMERGENT discrete anamorphic grid (R_i, A_i, a_i, Q_i) solved from the discrete Hamiltonian and
    radial momentum constraints (with NO analytical metric formula input), verifying that genuine
    spin-2 axial gravitational perturbations (s = 2, factor 1 - s^2 = -3) match exact GR
    perturbation theory (< 1e-4) and are distinct from spin-0 scalar perturbations (s = 0).
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
    t43 = test_43_emergent_exterior_schwarzschild_and_light_bending_from_discrete_constraints()
    t44 = test_44_emergent_interior_schwarzschild_constant_density_star_and_buchdahl_limit()
    t45 = test_45_emergent_painleve_gullstrand_river_and_lqc_core_from_anamorphic_duality()
    t46 = test_46_3d_anamorphic_18pt_spin2_transverse_traceless_gravitational_waves()
    t47 = test_47_emergent_spin2_regge_wheeler_barrier_vs_scalar_on_anamorphic_grid()

    all_passed = bool(
        t43["passed"]
        and t44["passed"]
        and t45["passed"]
        and t46["passed"]
        and t47["passed"]
    )
    return {
        "module": "Module 13: Emergent General Relativity & True 3D Spin-2 Gravitational Waves on the Anamorphic Grid",
        "test_43_emergent_exterior_schwarzschild_and_light_bending_from_discrete_constraints": t43,
        "test_44_emergent_interior_schwarzschild_constant_density_star_and_buchdahl_limit": t44,
        "test_45_emergent_painleve_gullstrand_river_and_lqc_core_from_anamorphic_duality": t45,
        "test_46_3d_anamorphic_18pt_spin2_transverse_traceless_gravitational_waves": t46,
        "test_47_emergent_spin2_regge_wheeler_barrier_vs_scalar_on_anamorphic_grid": t47,
        "module_13_passed": all_passed,
        "all_passed": all_passed,
    }


if __name__ == "__main__":
    results = run_module_13()
    print(json.dumps(results, indent=2))
    print(f"\nMODULE 13 PASSED: {results['all_passed']}")
    if not results["all_passed"]:
        sys.exit(1)
