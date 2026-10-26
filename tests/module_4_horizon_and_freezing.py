"""
Module 4: Horizon Crossing, Cayley Symplectic Evolution, Local Saturation, & Inner-Horizon Blueshift
====================================================================================================
Tests:
  Test 9:  Hayward River Profile: Horizon Locations & Core Scale (Benchmark 2)
  Test 10: Fixed-Background Skew-Symmetric & Cayley Symplectic Conservation + Inward Trapping (Benchmark 4)
  Test 11: 3D Local Saturation Damping: Energy Budget Closure & What the Cap Does (Benchmark 5b)
  Test 12: Trapped Region: Outgoing Characteristics c + J(r) < 0 for r_- < r < r_+ (Benchmark 5a)
  Test 13: Inner-Horizon Blueshift vs. Resolution, and What a Saturation Cap Does With the Excess

Pass/fail flags in this module check NUMERICAL CORRECTNESS (conservation, stability, finiteness,
algebraic identities). Physical conclusions are reported as measured values and summarized in the
claims audit printed by verify_core_finite_bar_gr.py; several of them do NOT support the original
claims of the manuscript, and that is reported as such.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulate_3d_18pt_freezing_core import run_3d_simulation
from pg_wave_1d import run_inner_horizon_packet, boundary_mode_growth_rate, hayward_horizons


def test_09_hayward_river_profile_horizons_and_core_scale():
    """
    Test 9:
    - Evaluates the Hayward river speed |J(r)| = c sqrt(r_s r^2 / (r^3 + r_s ell^2)) and the roots
      r_- < r_+ of f(r) = 1 - J^2/c^2 (cubic r^3 - r_s r^2 + r_s ell^2 = 0).
    - Confirms, across a mass sweep r_s in [3, 1000] ell, that the core scale R_core = (r_s ell^2)^(1/3)
      lies strictly between the inner horizon r_- (~ ell) and the outer horizon r_+ (~ r_s).
    - NOTE: r_- lies inside R_core, but it is NOT removed ("excised") from the spacetime: the Hayward
      geometry used everywhere in this repository still contains the inner (Cauchy) horizon at r_-.
      Only the old frozen-wall code (now removed) stopped evolving the test field there.
    """
    c = 1.0
    ell = 1.0
    r_s = 27.0 * ell
    R_core = (r_s * (ell**2)) ** (1.0 / 3.0)  # 3.0 * ell

    J_sharp_at_rs = c * np.sqrt(r_s / r_s)
    J_sharp_at_Rcore = c * np.sqrt(r_s / R_core)
    J_hayward_at_Rcore = c * np.sqrt(r_s * R_core**2 / (R_core**3 + r_s * ell**2))

    r_minus, r_plus, kappa_minus, kappa_plus = hayward_horizons(r_s, ell)
    r_minus_asymp = float(ell * (1.0 + ell / (2.0 * r_s)))

    rs_sweep = np.geomspace(3.0 * ell, 1000.0 * ell, 40)
    ordering_all = True
    for rs_val in rs_sweep:
        rc_val = (rs_val * (ell**2)) ** (1.0 / 3.0)
        rts = np.sort(np.real(np.roots([1.0, -rs_val, 0.0, rs_val * (ell**2)])))
        if not (rts[1] < rc_val < rts[2]):
            ordering_all = False

    passed = (
        abs(J_sharp_at_rs - 1.0) < 1e-15
        and abs(J_sharp_at_Rcore - 3.0) < 1e-15
        and abs(J_hayward_at_Rcore - 3.0 / np.sqrt(2.0)) < 1e-14
        and r_minus < R_core < r_plus
        and ordering_all
    )
    return {
        "passed": bool(passed),
        "r_s_over_ell": r_s,
        "R_core_over_ell": R_core,
        "J_over_c_at_R_core_sharp_profile": float(J_sharp_at_Rcore),
        "J_over_c_at_R_core_hayward_profile": float(J_hayward_at_Rcore),
        "inner_horizon_r_minus_over_ell": r_minus,
        "inner_horizon_asymptotic_over_ell": r_minus_asymp,
        "outer_horizon_r_plus_over_ell": r_plus,
        "inner_horizon_surface_gravity_kappa_minus": kappa_minus,
        "outer_horizon_surface_gravity_kappa_plus": kappa_plus,
        "r_minus_lt_R_core_lt_r_plus_all_masses": bool(ordering_all),
        "inner_horizon_still_present_in_geometry": True,
    }


def test_10_cayley_symplectic_and_skew_symmetric_conservation():
    """
    Test 10:
    - Builds the skew-symmetric advection matrix D_J (D_J^T = -D_J) for the Painleve-Gullstrand Killing
      Hamiltonian H = sum_i [Pi_i^2/(2 a_i) - Pi_i (D_J Phi)_i + 1/4 sum w_d (Phi_i - Phi_j)^2] with
      generator F = [[-D_J, A^-1], [-L_K, -D_J]] and Cayley map S(dt) = (I - dt/2 F)^-1 (I + dt/2 F).
    - Verifies: skew-symmetry, S^T Omega S = Omega, Killing-energy conservation over 200 steps, and that a
      packet released from rest at r_0 = 4.0 (inside r_+ ~ 5.9) is swept INWARD (black hole, not white hole).
    """
    N = 120
    r = np.linspace(-15.0, 15.0, N)
    ell = float(r[1] - r[0])
    r_s = 6.0
    c = 1.0

    abs_r = np.abs(r)
    J = -np.sign(r) * c * np.sqrt((r_s * (abs_r**2)) / ((abs_r**3) + r_s * (ell**2)))
    a = np.ones_like(r)

    K0 = 1.0
    L_K = np.zeros((N, N))
    D_J = np.zeros((N, N))
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        L_K[i, i] += 2.0 * K0 / (ell**2)
        L_K[i, ip] -= K0 / (ell**2)
        L_K[i, im] -= K0 / (ell**2)
        D_J[i, ip] = 0.5 * (J[i] + J[ip]) / (2.0 * ell)
        D_J[i, im] = -0.5 * (J[i] + J[im]) / (2.0 * ell)

    skew_err = float(np.max(np.abs(D_J + D_J.T)))
    A_inv = np.diag(1.0 / a)
    F_mat = np.block([[-D_J, A_inv], [-L_K, D_J.T]])
    dt = 0.02
    I_2N = np.eye(2 * N)
    S_step = np.linalg.solve(I_2N - 0.5 * dt * F_mat, I_2N + 0.5 * dt * F_mat)
    Omega = np.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])
    symplectic_err = float(np.max(np.abs(S_step.T @ Omega @ S_step - Omega)))

    phi0 = np.zeros(N)
    pi0 = np.zeros(N)
    r0_in = 4.0
    mask = np.abs(r - r0_in) < 2.0
    phi0[mask] = np.exp(-((r[mask] - r0_in) ** 2) / 0.50)

    def killing_hamiltonian(phi, pi):
        return float(0.5 * np.sum((pi**2) / a) - np.sum(pi * (D_J @ phi)) + 0.5 * float(phi @ (L_K @ phi)))

    def comoving_centroid(phi, pi):
        dphi = np.roll(phi, -1) - phi
        e_loc = 0.5 * (pi**2) + 0.5 * (dphi / ell) ** 2
        right = r > 0
        return float(np.sum((r * e_loc)[right]) / (np.sum(e_loc[right]) + 1e-15))

    H0 = killing_hamiltonian(phi0, pi0)
    centroid_t0 = comoving_centroid(phi0, pi0)
    z = np.concatenate([phi0, pi0])
    max_H_drift = 0.0
    centroid_t60 = centroid_t0
    for step_i in range(200):
        z = S_step @ z
        Ht = killing_hamiltonian(z[:N], z[N:])
        max_H_drift = max(max_H_drift, abs(Ht - H0) / abs(H0))
        if step_i == 59:
            centroid_t60 = comoving_centroid(z[:N], z[N:])

    passed = (
        skew_err < 1e-15
        and symplectic_err < 1e-13
        and max_H_drift < 1e-13
        and centroid_t60 < centroid_t0 - 1.0
    )
    return {
        "passed": bool(passed),
        "river_matrix_skew_symmetry_error": skew_err,
        "cayley_symplectic_condition_error": symplectic_err,
        "relative_killing_energy_drift_200_steps": float(max_H_drift),
        "inside_horizon_initial_centroid_r": centroid_t0,
        "inside_horizon_centroid_after_60_steps": centroid_t60,
    }


def test_11_3d_local_saturation_energy_budget():
    """
    Test 11:
    - Runs the 3D 25^3 Cayley lattice on the Hayward river profile with local saturation damping
      (comoving cell energy capped at E_voxel_max, excess moved into a static core reservoir).
    - Numerical checks: fixed-background Killing energy conserved to < 1e-13; the Killing-energy budget
      H(t) = H(0) + (change from background updates) - (removed by damping) closes to < 1e-10 in every run;
      the cap is respected in the capped runs.
    - Reported physics (not pass conditions): on a FIXED Hayward background the wave never exceeds the
      cap in this short run; the pile-up above the cap appears only when J is recomputed every step from
      the wave's own energy (an ad-hoc self-gravity update, not Einstein's equations), and that update
      changes the Killing energy by several percent.
    """
    res_fixed_bg = run_3d_simulation(cap_density=False, stencil_mode="18pt", fixed_background=True)
    res_fixed_cap = run_3d_simulation(cap_density=True, stencil_mode="18pt", fixed_background=True)
    res_18 = run_3d_simulation(cap_density=True, stencil_mode="18pt")
    res_27 = run_3d_simulation(cap_density=True, stencil_mode="27pt")

    passed = (
        res_fixed_bg["killing_budget_closure_residual"] < 1e-13
        and res_fixed_cap["killing_budget_closure_residual"] < 1e-10
        and res_18["killing_budget_closure_residual"] < 1e-10
        and res_27["killing_budget_closure_residual"] < 1e-10
        and res_18["peak_ratio_to_ceiling"] <= 1.000001
        and res_27["peak_ratio_to_ceiling"] <= 1.000001
    )
    return {
        "passed": bool(passed),
        "fixed_bg_uncapped_killing_drift": res_fixed_bg["killing_budget_closure_residual"],
        "fixed_bg_uncapped_peak_over_cap": res_fixed_bg["peak_ratio_to_ceiling"],
        "fixed_bg_capped_cells_that_hit_cap": res_fixed_cap["final_frozen_core_cells"] - res_fixed_cap["initial_frozen_core_cells"],
        "dynamic_bg_18pt_cells_that_hit_cap": res_18["final_frozen_core_cells"] - res_18["initial_frozen_core_cells"],
        "dynamic_bg_27pt_cells_that_hit_cap": res_27["final_frozen_core_cells"] - res_27["initial_frozen_core_cells"],
        "dynamic_bg_18pt_killing_change_from_background_updates_rel": res_18["killing_change_from_background_updates_rel"],
        "dynamic_bg_18pt_killing_removed_by_damping_rel": res_18["killing_removed_by_damping_rel"],
        "dynamic_bg_18pt_reservoir_gain_over_initial_wave_comoving_energy": res_18["reservoir_gain_over_initial_wave_comoving_energy"],
        "budget_closure_residual_18pt": res_18["killing_budget_closure_residual"],
        "budget_closure_residual_27pt": res_27["killing_budget_closure_residual"],
    }


def test_12_trapped_region_outgoing_characteristics():
    """
    Test 12 (replaces the former 'Vaidya horizon vise / bounced-wave pinning' test):
    - For the Hayward profile, the outgoing characteristic speed v_+(r) = c + J(r) is negative for every
      r_- < r < r_+. Any outward-moving signal there (including one 'bounced' at R_core) moves inward.
      This follows from the background mass alone; the extra step r_s(M) -> r_s(M + dM) used in the old
      'horizon vise' argument is not needed, and the old picture of a wave 'pinned' on a static surface
      at R_core is not possible, because r = const surfaces inside r_+ are spacelike (g_tt > 0 there).
    - Also checks the interior identity 2GM(r)/(c^2 r) = (r_s / R_core^3) r^2 of a uniform rho_max sphere.
    """
    c = 1.0
    ell = 1.0
    max_v_plus_in_trapped = -np.inf
    min_gtt_sign_in_trapped = np.inf
    for r_s in np.geomspace(3.0, 1000.0, 25):
        r_minus, r_plus, _, _ = hayward_horizons(r_s, ell)
        rr = np.linspace(r_minus, r_plus, 2002)[1:-1]
        J = -c * np.sqrt(r_s * rr**2 / (rr**3 + r_s * ell**2))
        max_v_plus_in_trapped = max(max_v_plus_in_trapped, float(np.max(c + J)))
        g_tt = -(c**2 - J**2)  # metric component of a static (r = const) worldline
        min_gtt_sign_in_trapped = min(min_gtt_sign_in_trapped, float(np.min(g_tt)))

    G = 1.0
    r_s = 10.0
    R_core = (r_s * ell**2) ** (1.0 / 3.0)
    rho_max = 3.0 * c**2 / (8.0 * np.pi * G * ell**2)
    r_test = 0.5 * R_core
    M_enc = (4.0 / 3.0) * np.pi * (r_test**3) * rho_max
    err_interior = float(abs(2.0 * G * M_enc / ((c**2) * r_test) - (r_s / (R_core**3)) * (r_test**2)))

    passed = max_v_plus_in_trapped < 0.0 and min_gtt_sign_in_trapped > 0.0 and err_interior < 1e-14
    return {
        "passed": bool(passed),
        "max_outgoing_speed_c_plus_J_in_trapped_region": max_v_plus_in_trapped,
        "min_g_tt_of_static_worldline_in_trapped_region_positive_means_spacelike": min_gtt_sign_in_trapped,
        "interior_quadratic_metric_identity_error": err_interior,
    }


def test_13_inner_horizon_blueshift_and_saturation_cap():
    """
    Test 13 (rewritten):
    Clean 1D test-field study on the Hayward background with the outer boundary far outside r_+
    (see pg_wave_1d.py). A packet is launched between r_- and r_+ moving outward relative to the river,
    so it is carried toward the inner horizon r_- where c + J -> 0.

    Measured (reported, not pass conditions):
      1. Uncapped: the peak comoving energy near r_- grows like 1/h when the lattice spacing h is halved
         (blueshift limited only by the grid; it diverges in the continuum limit).
      2. Capped (comoving density <= 2x the initial peak, excess moved into a static reservoir, as in the
         3D simulator): the field stays below the cap by construction, but the energy deposited in the
         reservoir is several times the packet's initial comoving energy and keeps growing as h -> 0.
         In the 3D model that reservoir is added to the Misner-Sharp mass, so the cap relabels the
         blueshifted energy as core mass rather than removing it. Whether this amounts to mass inflation
         requires dynamical geometry (backreaction), which this repository does not have.
      3. Numerical artifact found in the previous version of this test: if the domain is truncated
         inside the trapped region with J != 0 at the boundary, boundary-localized growing modes appear
         with growth rate ~ 1/h. Tapering J to zero in an outer sponge removes them.

    Pass conditions (numerical correctness): exact Killing conservation without sponge and cap,
    finiteness, tapered-boundary stability, and the cap being respected.
    """
    r_s = 12.0
    resolutions = [0.04, 0.02, 0.01]
    uncapped = [run_inner_horizon_packet(h, r_s=r_s, T=40.0) for h in resolutions]
    capped = [run_inner_horizon_packet(h, r_s=r_s, T=40.0, cap_factor=2.0) for h in resolutions]
    conservation = run_inner_horizon_packet(0.04, r_s=r_s, T=10.0, use_sponge=False)

    peaks = np.array([u["max_comoving_energy_near_r_minus_over_E0"] for u in uncapped])
    hs = np.array(resolutions)
    blueshift_scaling_exponent = float(-np.polyfit(np.log(hs), np.log(peaks), 1)[0])
    reservoir = [cp["reservoir_comoving_energy_over_E0"] for cp in capped]

    rate_untapered_h04, lo_u4, hi_u4 = boundary_mode_growth_rate(0.04, r_s=r_s, taper=False)
    rate_untapered_h02, _, _ = boundary_mode_growth_rate(0.02, r_s=r_s, taper=False)
    rate_tapered_h04, _, _ = boundary_mode_growth_rate(0.04, r_s=r_s, taper=True)
    rate_tapered_h02, _, _ = boundary_mode_growth_rate(0.02, r_s=r_s, taper=True)

    passed = (
        conservation["max_relative_killing_drift_no_sponge_no_cap"] < 1e-10
        and all(u["all_finite"] for u in uncapped + capped)
        and all(cp["max_density_over_initial_peak"] <= 2.0 * (1.0 + 1e-9) for cp in capped)
        and rate_tapered_h04 < 0.05
        and rate_tapered_h02 < 0.05
    )
    return {
        "passed": bool(passed),
        "r_minus_over_ell": uncapped[0]["r_minus"],
        "r_plus_over_ell": uncapped[0]["r_plus"],
        "kappa_minus": uncapped[0]["kappa_minus"],
        "lattice_spacings_h": resolutions,
        "uncapped_peak_comoving_energy_near_r_minus_over_E0": [float(p) for p in peaks],
        "uncapped_blueshift_scaling_exponent_peak_propto_h_to_minus": blueshift_scaling_exponent,
        "capped_reservoir_deposit_over_E0": reservoir,
        "capped_reservoir_fraction_within_1ell_of_r_minus": [cp["reservoir_fraction_deposited_within_1ell_of_r_minus"] for cp in capped],
        "cap_respected_max_density_over_initial_peak": [cp["max_density_over_initial_peak"] for cp in capped],
        "killing_drift_no_sponge_no_cap": conservation["max_relative_killing_drift_no_sponge_no_cap"],
        "artifact_untapered_boundary_growth_rate_h0p04": rate_untapered_h04,
        "artifact_untapered_boundary_growth_rate_h0p02": rate_untapered_h02,
        "artifact_untapered_mode_support_r": [lo_u4, hi_u4],
        "tapered_boundary_growth_rate_h0p04": rate_tapered_h04,
        "tapered_boundary_growth_rate_h0p02": rate_tapered_h02,
        "finding_cap_bounds_field_energy": True,
        "finding_reservoir_exceeds_infalling_energy": bool(min(reservoir) > 1.0),
        "finding_reservoir_converged_in_h": bool(abs(reservoir[-1] - reservoir[-2]) / reservoir[-1] < 0.05),
    }


def run_module_4():
    res = {
        "test_09_hayward_river_profile_horizons_and_core_scale": test_09_hayward_river_profile_horizons_and_core_scale(),
        "test_10_cayley_symplectic_and_skew_symmetric_conservation": test_10_cayley_symplectic_and_skew_symmetric_conservation(),
        "test_11_3d_local_saturation_energy_budget": test_11_3d_local_saturation_energy_budget(),
        "test_12_trapped_region_outgoing_characteristics": test_12_trapped_region_outgoing_characteristics(),
        "test_13_inner_horizon_blueshift_and_saturation_cap": test_13_inner_horizon_blueshift_and_saturation_cap(),
    }
    res["module_4_passed"] = all(v["passed"] for v in res.values())
    return res


if __name__ == "__main__":
    out = run_module_4()
    print(json.dumps(out, indent=2))
    print(f"\nMODULE 4 PASSED: {out['module_4_passed']}")
