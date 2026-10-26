"""
Module 8: Extensions & Stress Tests (Tests 22-25)
=================================================
  Test 22: Off-Center 3D Wave Packet with Local Saturation Damping (numerics + energy budget)
  Test 23: Two-Component Scalar Toy With Ad-Hoc Rotation Coupling (NOT linearized GR / not spin-2)
  Test 24: Regular Schwarzschild-de Sitter Horizon Structure & Lattice Brillouin-Zone Group Velocity
  Test 25: Analytic Comparison of POSTULATED Completions (V1 transition vs V2 bounce) + Entropy Bound.
           Nothing here derives which completion the lattice rule produces; that needs a dynamical
           spherically symmetric Einstein-matter code with the saturation rule inside the dynamics.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulate_3d_18pt_freezing_core import (
    build_27pt_offsets_and_weights,
    cayley_midpoint_step_3d,
    compute_comoving_and_killing_energies,
    compute_hayward_gravity_from_cells,
    apply_local_saturation_damping,
)


def test_22_off_center_asymmetric_3d_stefan_accretion():
    """
    Test 22 (Extension 1: Breaking Spherical Symmetry during 3D Core Accretion):
    - Fires an OFF-CENTER, lopsided 3D wave packet (shifted toward +X, -Y) into the
      seed black hole on a 19x19x19 27-point lattice using the implicit Cayley midpoint
      Painleve-Gullstrand integrator and well-posed local saturation damping.
    - Verifies that removing spherical symmetry does NOT break the numerics:
      1. Active cell co-moving energy density remains capped at E_i <= E_{voxel,max} (by construction),
      2. The saturated region grows asymmetrically on the side impacted by the wave,
      3. The Killing-energy budget H(t) = H(0) + (ad-hoc background-update change) - (damping removal)
         closes to round-off. The two non-conservative terms are reported, not hidden.
    - Not tested: this is a test field on a fixed-form river profile; the enclosed-mass update is
      spherical (shell sums) even though the wave is not, so it is not a non-spherical gravity solution.
    """
    n_grid = 19
    n_steps = 38
    dt = 0.03
    G = 0.025
    c = 1.0
    ell = 1.0
    E_voxel_max = (3.0 / (8.0 * np.pi)) * 20.0
    offsets = build_27pt_offsets_and_weights()

    coords = np.arange(n_grid) - (n_grid // 2)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)

    seed_core = R <= 1.8
    E_core = np.where(seed_core, E_voxel_max, 0.0)
    saturated_visit_mask = seed_core.copy()
    initial_frozen = int(np.sum(saturated_visit_mask))

    # Off-center asymmetric wave packet biased toward (+X, -Y)
    phi = np.zeros_like(R)
    pi = np.zeros_like(R)
    R_shifted = np.sqrt((X - 0.75) ** 2 + (Y + 0.45) ** 2 + (Z - 0.25) ** 2)
    shell_mask = (R >= 2.0) & (R <= 5.8)
    env = np.exp(-((R_shifted[shell_mask] - 3.6) ** 2) / 0.85)
    phi[shell_mask] = 1.48 * env
    pi[shell_mask] = 1.88 * env

    Jx, Jy, Jz, _ = compute_hayward_gravity_from_cells(E_core, R, X, Y, Z, G=G, c=c, ell=ell)
    E_com_0, H_kill_0 = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)
    bg_change = 0.0
    removed_by_damping = 0.0
    max_budget_residual = 0.0
    peak_active_energy = float(np.max(E_com_0))

    for _ in range(n_steps):
        E_com_now, H_old_J = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)
        Jx, Jy, Jz, _ = compute_hayward_gravity_from_cells(E_core + E_com_now, R, X, Y, Z, G=G, c=c, ell=ell)
        _, H_new_J = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)
        bg_change += H_new_J - H_old_J
        phi, pi = cayley_midpoint_step_3d(phi, pi, Jx, Jy, Jz, offsets, dt)

        _, H_k_before = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)
        phi, pi, E_com_after = apply_local_saturation_damping(
            phi, pi, E_core, saturated_visit_mask, offsets, E_voxel_max
        )
        _, H_k_after = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)
        removed_by_damping += H_k_before - H_k_after

        peak_active_energy = max(peak_active_energy, float(np.max(E_com_after)))
        residual = abs(H_k_after - (H_kill_0 + bg_change - removed_by_damping)) / abs(H_kill_0)
        max_budget_residual = max(max_budget_residual, residual)

    final_frozen = int(np.sum(saturated_visit_mask))
    # Check that saturated core grew asymmetrically toward the +X impact side
    frozen_x_cm = float(np.sum(X[saturated_visit_mask]) / np.sum(saturated_visit_mask))

    passed = (
        final_frozen > initial_frozen
        and peak_active_energy <= E_voxel_max * 1.000001
        and max_budget_residual < 1e-10
        and frozen_x_cm > 0.0
    )
    return {
        "passed": bool(passed),
        "initial_frozen_cells": initial_frozen,
        "final_frozen_cells_asymmetric": final_frozen,
        "frozen_core_center_of_mass_x_shift": frozen_x_cm,
        "peak_active_energy_over_ceiling": float(peak_active_energy / E_voxel_max),
        "killing_budget_closure_residual": float(max_budget_residual),
        "killing_change_from_ad_hoc_background_updates_rel": float(bg_change / abs(H_kill_0)),
        "killing_removed_by_saturation_damping_rel": float(removed_by_damping / abs(H_kill_0)),
    }


def test_23_two_component_scalar_toy_with_ad_hoc_rotation_coupling():
    """
    Test 23 (relabelled; previously described as a 'spin-2 gravitational-wave doublet'):
    - Evolves TWO COPIES of the scalar lattice field (labelled h_+, h_x) coupled by an ad-hoc
      skew-symmetric SO(2) rotation term with rate
        Omega(r) = s * (a_spin * r_s * c) / (r^3 + r_s ell^2 + a_spin^2 r)   (s = 2),
      on the ad-hoc spinning river profile of Benchmark 10.
    - THIS IS NOT LINEARIZED GENERAL RELATIVITY. Genuine spin-2 perturbations would need the
      Regge-Wheeler / Zerilli (or Teukolsky) equations on the regular background, which are not
      implemented here. The test only checks that this toy Hamiltonian system is integrated correctly.
    - Verifies:
      1. 4N x 4N symplectic structure preservation S(dt)^T Omega_{4N} S(dt) = Omega_{4N} (< 1e-13),
      2. Conservation of the toy two-component energy (< 1e-13),
      3. The coupling rotates h_+ into h_x when a_spin > 0 (and trivially does nothing when the
         coupling is zero, a_spin = 0).
    """
    n_grid = 7
    n_cells = n_grid**3
    ell = 1.0
    c = 1.0
    r_s = 2.8
    spin_weight_s = 2.0

    coords = np.arange(n_grid) - (n_grid // 2)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    offsets = build_27pt_offsets_and_weights()

    def run_spin2_doublet(a_spin_val):
        denom = (R**3) + r_s * (ell**2) + (a_spin_val**2) * R
        J_r_mag = c * np.sqrt(np.divide(r_s * (R**2), denom, out=np.zeros_like(R), where=denom > 1e-12))
        J_phi_over_rho = (J_r_mag * a_spin_val) / (R**2 + ell**2)
        inv_R = np.divide(1.0, R, out=np.zeros_like(R), where=R > 1e-12)
        Jx = -J_r_mag * X * inv_R - J_phi_over_rho * Y
        Jy = -J_r_mag * Y * inv_R + J_phi_over_rho * X
        Jz = -J_r_mag * Z * inv_R

        # Spin-2 frame-dragging axial precession rate Omega_LT = s * a_spin * r_s / denom
        omega_lt = (spin_weight_s * a_spin_val * r_s * c / denom).ravel()

        L_K = np.zeros((n_cells, n_cells), dtype=float)
        M_adv = np.zeros((n_cells, n_cells), dtype=float)
        Jx_f, Jy_f, Jz_f = Jx.ravel(), Jy.ravel(), Jz.ravel()

        def idx3(ix, iy, iz):
            return (ix % n_grid) * (n_grid * n_grid) + (iy % n_grid) * n_grid + (iz % n_grid)

        for ix in range(n_grid):
            for iy in range(n_grid):
                for iz in range(n_grid):
                    i = idx3(ix, iy, iz)
                    for dx, dy, dz, w in offsets:
                        j = idx3(ix + dx, iy + dy, iz + dz)
                        L_K[i, i] += w
                        L_K[i, j] -= w
                        J_dot_d = 0.5 * (
                            (Jx_f[i] + Jx_f[j]) * dx
                            + (Jy_f[i] + Jy_f[j]) * dy
                            + (Jz_f[i] + Jz_f[j]) * dz
                        )
                        M_adv[i, j] += 0.25 * w * J_dot_d

        # 2-polarization block advection + spin-2 Faraday rotation matrix M_2pol (2N x 2N):
        # M_2pol = [[M_adv, -diag(omega_lt)], [+diag(omega_lt), M_adv]]
        # Notice M_2pol^T = [[-M_adv, +diag(omega_lt)], [-diag(omega_lt), -M_adv]] = -M_2pol!
        W_rot = np.diag(omega_lt)
        M_2pol = np.block([[M_adv, -W_rot], [W_rot, M_adv]])
        L_2pol = np.block([[L_K, np.zeros_like(L_K)], [np.zeros_like(L_K), L_K]])
        I_2N = np.eye(2 * n_cells)

        skew_2pol_err = float(np.max(np.abs(M_2pol + M_2pol.T)))

        # Full 4N x 4N Hamiltonian phase-space generator for state Z = [h_+, h_x, Pi_+, Pi_x]
        F_4N = np.block([[-M_2pol, I_2N], [-L_2pol, M_2pol.T]])
        dt = 0.04
        I_4N = np.eye(4 * n_cells)
        S_4N = np.linalg.solve(I_4N - 0.5 * dt * F_4N, I_4N + 0.5 * dt * F_4N)

        Omega_4N = np.block([[np.zeros((2 * n_cells, 2 * n_cells)), I_2N], [-I_2N, np.zeros((2 * n_cells, 2 * n_cells))]])
        symp_err = float(np.max(np.abs(S_4N.T @ Omega_4N @ S_4N - Omega_4N)))

        # Initialize purely linearly polarized (+ polarization only: h_+ != 0, h_x = 0) wave packet
        h_plus_0 = np.exp(-(((X - 1.5) ** 2 + Y**2 + Z**2) / 1.4)).ravel()
        h_cross_0 = np.zeros(n_cells, dtype=float)
        pi_plus_0 = np.zeros(n_cells, dtype=float)
        pi_cross_0 = np.zeros(n_cells, dtype=float)
        z = np.concatenate([h_plus_0, h_cross_0, pi_plus_0, pi_cross_0])

        def calc_H(z_vec):
            h = z_vec[: 2 * n_cells]
            p = z_vec[2 * n_cells :]
            return float(0.5 * (p @ p) - (p @ (M_2pol @ h)) + 0.5 * (h @ (L_2pol @ h)))

        H0 = calc_H(z)
        max_drift = 0.0
        for _ in range(30):
            z = S_4N @ z
            Ht = calc_H(z)
            max_drift = max(max_drift, abs(Ht - H0) / abs(H0))

        h_plus_end = z[:n_cells]
        h_cross_end = z[n_cells : 2 * n_cells]
        cross_pol_fraction = float(np.sum(h_cross_end**2) / (np.sum(h_plus_end**2) + np.sum(h_cross_end**2)))
        return skew_2pol_err, symp_err, max_drift, cross_pol_fraction

    skew_0, symp_0, drift_0, cross_nonspinning = run_spin2_doublet(0.0)
    skew_s, symp_s, drift_s, cross_spinning = run_spin2_doublet(0.85 * r_s)

    passed = (
        skew_s < 1e-15
        and symp_s < 1e-13
        and drift_s < 1e-13
        and cross_nonspinning == 0.0
        and cross_spinning > 0.05
    )
    return {
        "passed": bool(passed),
        "toy_operator_skew_symmetry_error": skew_s,
        "toy_4N_symplectic_condition_error": symp_s,
        "toy_relative_energy_drift_30steps": drift_s,
        "component_mixing_fraction_zero_coupling": cross_nonspinning,
        "component_mixing_fraction_with_ad_hoc_coupling": cross_spinning,
    }


def test_24_cosmological_de_sitter_embedding_and_brillouin_uv_limit():
    """
    Test 24 (Extension 3: Cosmological Schwarzschild-de Sitter & Exact Brillouin UV Limit):
    - Part A (Cosmological Lambda > 0 Embedding):
      Embeds the nonsingular Finite-Bar black hole inside a cosmological de Sitter background:
        f_SdS(r) = 1 - r_s r^2 / (r^3 + r_s ell^2) - (Lambda / 3) r^2.
      Verifies that all three horizons exist simultaneously (inner horizon r_- < R_core, still present,
      black hole event horizon r_+ ~ r_s, and cosmological de Sitter horizon r_c ~ sqrt(3/Lambda)),
      while the origin Kretschmann scalar K(0) = 24 * (1/ell^2 + Lambda/3)^2 stays strictly finite.
    - Part B (Finding the Exact UV Breaking Point at the Brillouin Cutoff lambda = 2*ell):
      Sweeps wavevector k from continuum IR scales (lambda = 1000*ell) to the Planck lattice
      Brillouin boundary k_max = pi / ell (lambda_min = 2*ell).
      Proves that while continuum GR wave propagation holds to < 1e-5 for lambda >> ell,
      right at the lattice cutoff lambda = 2*ell the axis group velocity v_g(pi/ell) = 0
      (standard Brillouin-zone behaviour of any nearest-neighbour lattice; no frequency above
      omega_max = 2c/ell propagates). Inside a horizon this subluminal dispersion is exactly what
      drives Corley-Jacobson mode conversion; it does not by itself make horizon physics unitary.
    """
    # Part A: Regular Schwarzschild-de Sitter (SdS) triple-root & curvature check
    ell = 1.0
    r_s = 10.0 * ell
    R_core = (r_s * (ell**2)) ** (1.0 / 3.0)
    Lambda = 3.0 / ((100.0 * ell) ** 2)  # Cosmological horizon r_c ~ 100 * ell

    def f_sds(r):
        return 1.0 - (r_s * (r**2)) / ((r**3) + (R_core**3)) - (Lambda / 3.0) * (r**2)

    # Locate the 3 positive horizon roots: r_- (inner), r_+ (BH event), r_c (cosmological)
    # via bisection on [0.5, 2.0], [5.0, 15.0], and [80.0, 120.0]
    def bisect_root(a, b):
        for _ in range(60):
            m = 0.5 * (a + b)
            if f_sds(a) * f_sds(m) <= 0.0:
                b = m
            else:
                a = m
        return float(0.5 * (a + b))

    r_minus = bisect_root(0.8 * ell, 1.5 * ell)
    r_plus = bisect_root(8.0 * ell, 11.0 * ell)
    r_cosmo = bisect_root(80.0 * ell, 110.0 * ell)

    # Origin Kretschmann scalar for f_SdS(r) = 1 - (1/ell^2 + Lambda/3) r^2 + O(r^5):
    # K(0) = 24 * (1/ell^2 + Lambda/3)^2
    eps = 1e-4 * ell
    fpp_0 = (f_sds(eps) - 2.0 * f_sds(0.0) + f_sds(-eps)) / (eps**2)
    K_0_num = float(6.0 * (fpp_0**2))
    K_0_exact = float(24.0 * ((1.0 / (ell**2) + Lambda / 3.0) ** 2))
    K_0_rel_err = abs(K_0_num - K_0_exact) / K_0_exact

    # Part B: Brillouin Zone UV Cutoff Sweep (lambda = 1000*ell down to lambda_min = 2*ell)
    c = 1.0
    # Along a lattice axis, omega(k) = (2 c / ell) * sin(k * ell / 2), v_g(k) = d(omega)/dk = c * cos(k * ell / 2)
    k_ir = 2.0 * np.pi / (1000.0 * ell)
    vg_ir = float(c * np.cos(0.5 * k_ir * ell))
    k_uv_cutoff = np.pi / ell  # lambda_min = 2 * ell (Nyquist-Brillouin limit)
    vg_at_brillouin_cutoff = float(c * np.cos(0.5 * k_uv_cutoff * ell))
    omega_max_planck = float((2.0 * c / ell) * np.sin(0.5 * k_uv_cutoff * ell))

    passed = (
        r_minus < R_core < r_plus < r_cosmo
        and abs(f_sds(r_minus)) < 1e-13
        and abs(f_sds(r_plus)) < 1e-13
        and abs(f_sds(r_cosmo)) < 1e-13
        and K_0_rel_err < 1e-7
        and abs(vg_ir - c) < 1e-5
        and abs(vg_at_brillouin_cutoff) < 1e-15
        and abs(omega_max_planck - 2.0 * c / ell) < 1e-15
    )
    return {
        "passed": bool(passed),
        "sds_inner_horizon_r_minus_over_ell": r_minus,
        "sds_core_scale_R_core_over_ell": float(R_core),
        "sds_bh_event_horizon_r_plus_over_ell": r_plus,
        "sds_cosmological_horizon_r_c_over_ell": r_cosmo,
        "sds_origin_kretschmann_K0": K_0_num,
        "sds_origin_kretschmann_relative_error": K_0_rel_err,
        "group_velocity_over_c_at_lambda_1000ell_IR": vg_ir,
        "group_velocity_over_c_at_lambda_2ell_Brillouin_cutoff": vg_at_brillouin_cutoff,
        "max_lattice_frequency_omega_max_times_tP": omega_max_planck,
    }


def test_25_postulated_completions_comparison_and_entropy_bound():
    """
    Test 25 (renamed; previously 'Dynamical 1+1 PG Collapse & Trilemma Classifier'):
    IMPORTANT: this is NOT a dynamical collapse simulation and does NOT classify anything.
    The lattice saturation rule is not used. Each completion below is POSTULATED and its defining
    equation is evaluated or integrated (rho <= rho_max = 3 c^2 / (8*pi*G*ell^2), r_s = 27 ell):
        * Baseline (Classical Oppenheimer-Snyder, rho_max -> inf):
          Collapses to R -> 0 (singularity) in exact proper time tau_OS = (2/3) R_0^(3/2)/sqrt(c^2 r_s).
        * Version 1 (POSTULATED limiting-curvature phase transition, Frolov-Markov-Mukhanov 1989):
          the collapsing interior is assumed to change phase across a spacelike surface (J^2(R_core)/c^2 = r_s/R_core = 9.0 > 1) at R_core = (r_s ell^2)^(1/3),
          where the Schwarzschild curvature 12 r_s^2 / r^6 matches 24 / ell^4 at
          r_match = 2^(-1/6) R_core ~ 0.890899 R_core.
        * Version 2 (POSTULATED bounce, effective-LQC form as in Kelly-Santacruz-Wilson-Ewing):
          Energy-conserving holonomy-like saturation dR/dtau = -sqrt((c^2 r_s / R)*(1 - rho/rho_max))
          with d^2R/dtau^2 = -(c^2 r_s / (2 R^2))*(1 - 4 R_core^3 / R^3) halts collapse at
          R_bounce = R_core = (r_s ell^2)^(1/3) BY CONSTRUCTION (the factor (1 - rho/rho_max) with
          rho/rho_max = R_core^3/R^3 is an input: it is the LQC effective Friedmann equation) and rebounds outward
          with positive bounce acceleration d^2R/dtau^2(R_core) = +3 c^2 R_core / (2 ell^2) > 0.
    - Also verifies the Bekenstein core entropy bound S_core / k_B <= pi (r_s / ell_P) vs.
      S_BH / k_B = pi (r_s / ell_P)^2, confirming the exact factor r_s / ell_P ~ 1.83e38
      information deficit of a 1D-scaling static core for a 1 M_sun black hole.
    """
    c = 1.0
    ell = 1.0
    r_s = 27.0 * ell
    R_core = (r_s * (ell**2)) ** (1.0 / 3.0)  # 3.0 * ell
    R0 = 2.0 * r_s  # 54.0 * ell

    # 1. Exact analytical Oppenheimer-Snyder collapse proper time from R0 to R_core:
    tau_os_to_Rcore = float((2.0 / (3.0 * np.sqrt(c**2 * r_s))) * ((R0**1.5) - (R_core**1.5)))

    # 2. Version 1 (FMM spacelike limiting-curvature transition at R_core):
    J2_over_c2_at_Rcore = float(r_s / R_core)  # 9.0 > 1 (strictly spacelike constant-r surface)
    r_curv_match_over_Rcore = float(2.0 ** (-1.0 / 6.0))  # 0.890898718...
    K_schw_at_match = float(12.0 * (r_s**2) / ((r_curv_match_over_Rcore * R_core) ** 6))
    K_de_sitter_core = float(24.0 / (ell**4))
    curv_match_err = abs(K_schw_at_match - K_de_sitter_core)

    # 3. Version 2 (Hamiltonian / KSW bounce integration of d^2R/dtau^2 = -(c^2 r_s / (2 R^2))*(1 - 4 R_core^3 / R^3)):
    dtau = 0.002
    R_val = R0
    v_val = -c * np.sqrt((r_s / R0) * (1.0 - (R_core / R0) ** 3))

    def accel_ksw(r_now):
        return -0.5 * (c**2) * r_s / (r_now**2) * (1.0 - 4.0 * (R_core**3) / (r_now**3))

    R_min_reached = R0
    bounced_outward = False
    max_energy_invariant_err = 0.0
    for _ in range(45000):
        # Velocity Verlet symplectic step
        a_now = accel_ksw(R_val)
        R_next = R_val + v_val * dtau + 0.5 * a_now * (dtau**2)
        a_next = accel_ksw(R_next)
        v_next = v_val + 0.5 * (a_now + a_next) * dtau
        R_val, v_val = R_next, v_next
        R_min_reached = min(R_min_reached, R_val)
        # First integral check: v^2 - (c^2 r_s / R)*(1 - R_core^3 / R^3) == 0
        inv_err = abs((v_val**2) - (c**2 * r_s / R_val) * (1.0 - (R_core / R_val) ** 3))
        max_energy_invariant_err = max(max_energy_invariant_err, inv_err)
        if v_val > 0.0 and R_val > 1.5 * R_core:
            bounced_outward = True
            break

    bounce_radius_rel_err = abs(R_min_reached - R_core) / R_core
    bounce_accel_at_Rcore = float(accel_ksw(R_core))
    expected_bounce_accel = float(1.5 * (c**2) * R_core / (ell**2))

    # 4. Bekenstein core entropy bound vs. Bekenstein-Hawking black hole entropy for 1 M_sun:
    rs_over_ellP_solar = 1.8264e38
    N_core_solar = (4.0 * np.pi / 3.0) * rs_over_ellP_solar
    # With E_{voxel,max} = (3 / (8*pi)) E_P and R_cell = ell_P, S_cell / k_B <= 2*pi * E_{voxel,max} * ell_P / (hbar c) = 3/4
    S_core_max_solar = float(N_core_solar * 0.75)  # = pi * (r_s / ell_P)
    S_BH_solar = float(np.pi * (rs_over_ellP_solar**2))
    entropy_deficit_factor = float(S_BH_solar / S_core_max_solar)

    passed = (
        tau_os_to_Rcore > 0.0
        and J2_over_c2_at_Rcore > 1.0
        and curv_match_err < 1e-13
        and bounced_outward
        and bounce_radius_rel_err < 1e-5
        and abs(bounce_accel_at_Rcore - expected_bounce_accel) < 1e-14
        and max_energy_invariant_err < 1e-5
        and abs(entropy_deficit_factor - rs_over_ellP_solar) / rs_over_ellP_solar < 1e-14
    )
    return {
        "passed": bool(passed),
        "oppenheimer_snyder_proper_time_to_Rcore": tau_os_to_Rcore,
        "version1_fmm_J2_over_c2_at_Rcore_spacelike": J2_over_c2_at_Rcore,
        "version1_curvature_matching_radius_over_Rcore": r_curv_match_over_Rcore,
        "version1_curvature_match_error": curv_match_err,
        "version2_ksw_bounce_min_radius_over_ell": float(R_min_reached),
        "version2_ksw_expected_Rcore_over_ell": float(R_core),
        "version2_ksw_bounce_radius_relative_error": bounce_radius_rel_err,
        "version2_ksw_outward_bounce_acceleration": bounce_accel_at_Rcore,
        "version2_ksw_energy_invariant_error": float(max_energy_invariant_err),
        "solar_mass_core_voxel_count_N_core": float(N_core_solar),
        "solar_mass_core_bekenstein_entropy_bound": S_core_max_solar,
        "solar_mass_bh_bekenstein_hawking_entropy": S_BH_solar,
        "bekenstein_entropy_deficit_ratio_S_BH_over_S_core": entropy_deficit_factor,
    }


def run_module_8():
    res = {
        "test_22_off_center_asymmetric_3d_stefan_accretion": test_22_off_center_asymmetric_3d_stefan_accretion(),
        "test_23_two_component_scalar_toy_with_ad_hoc_rotation_coupling": test_23_two_component_scalar_toy_with_ad_hoc_rotation_coupling(),
        "test_24_cosmological_de_sitter_embedding_and_brillouin_uv_limit": test_24_cosmological_de_sitter_embedding_and_brillouin_uv_limit(),
        "test_25_postulated_completions_comparison_and_entropy_bound": test_25_postulated_completions_comparison_and_entropy_bound(),
    }
    res["module_8_passed"] = all(v["passed"] for v in res.values())
    return res


if __name__ == "__main__":
    out = run_module_8()
    print(json.dumps(out, indent=2))
    print(f"\nMODULE 8 PASSED: {out['module_8_passed']}")
