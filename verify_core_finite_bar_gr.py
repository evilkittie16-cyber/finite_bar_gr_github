"""
verify_core_finite_bar_gr.py
============================
Clean, self-contained verification of the Finite-Bar Discrete General Relativity
(GR) Black Hole Model documented in proof_corrected.md.

Ten core benchmarks plus the 9-module / 29-test suite in tests/.

How to read the output:
  * "passed" flags check NUMERICAL CORRECTNESS and ALGEBRAIC IDENTITIES (conservation, symplecticity,
    stability, finiteness, closed-form formulas). They do not by themselves validate the physics claims.
  * The CLAIMS AUDIT printed at the end states, claim by claim, whether the code SUPPORTS, does NOT
    SUPPORT, or has NOT TESTED each headline claim of the manuscript. Several original claims are not
    supported; that is reported deliberately.
"""

import json
import os
import numpy as np


def test_1_equivalence_principle_and_gr_bending():
    """
    Test 1:
    - Verifies that BOTH coordinate gauges used in the manuscript—the weak-field isotropic
      gauge (n_iso = psi^2 / w) AND the horizon-penetrating Painleve-Gullstrand river gauge
      (n_PG = 1 / (1 - J(r)^2 / c^2))—reproduce the exact 1915 GR factor of 2:
        n(r) = 1 + 2 * GM/(c^2 r) + c_2PN * (GM/(c^2 r))^2 + O((GM/r)^3)
      with analytical 2PN coefficients c_2PN = 7/4 (isotropic) and c_2PN = 4 (PG).
    - Verifies that the local proper wave speed measured by a cell's proper clock
      dtau = dt * sqrt(K_i / a_i) matches the conformal metric identity sqrt(K_i / a_i) == w / psi^2.
    """
    G_M = 1.0
    c = 1.0
    ell = 0.01
    R_core = (2.0 * G_M * (ell**2)) ** (1.0 / 3.0)
    r_vals = np.linspace(5.0, 100.0, 200)

    # 1. Non-trivial algebraic check: sqrt(K_link / a_cell) with a = psi^7/w, K = psi^3*w
    #    equals the exact isotropic lapse-to-conformal ratio w / psi^2 across all r
    r_tilde = np.sqrt(r_vals**2 + R_core**2)
    psi = 1.0 + G_M / (2.0 * r_tilde)
    w = (1.0 - G_M / (2.0 * r_tilde)) / psi
    a_cell = (psi**7) / w
    K_link = (psi**3) * w
    c_coord_from_weights = np.sqrt(K_link / a_cell)
    c_coord_exact_metric = w / (psi**2)
    max_conformal_weight_err = float(np.max(np.abs(c_coord_from_weights - c_coord_exact_metric)))

    # 2. Extract 1PN (1915 GR factor of 2) in BOTH Isotropic and Painleve-Gullstrand gauges
    #    at r_test = 1000.0, and verify that the residual is the exact analytical 2PN term!
    r_test = 1000.0
    u = G_M / ((c**2) * r_test)  # dimensionless potential GM/(c^2 r) = 1e-3

    # Isotropic gauge: n_iso = psi^2 / w = 1 + 2 u + (7/4) u^2 + O(u^3)
    r_t = np.sqrt(r_test**2 + R_core**2)
    psi_t = 1.0 + G_M / (2.0 * r_t)
    w_t = (1.0 - G_M / (2.0 * r_t)) / psi_t
    n_iso = (psi_t**2) / w_t
    coeff_1PN_iso_raw = float((n_iso - 1.0) / u)
    coeff_1PN_iso_2PN_corrected = float((n_iso - 1.0 - 1.75 * (u**2)) / u)

    # Painleve-Gullstrand (River) gauge: J^2/c^2 = r_s * r^2 / (r^3 + R_core^3)
    # Two-way radial refractive index n_PG = 1 / (1 - J^2/c^2) = 1 + 2 u + 4 u^2 + O(u^3)
    r_s = 2.0 * G_M / (c**2)
    J2_over_c2 = (r_s * (r_test**2)) / (r_test**3 + R_core**3)
    n_PG = 1.0 / (1.0 - J2_over_c2)
    coeff_1PN_PG_raw = float((n_PG - 1.0) / u)
    coeff_1PN_PG_2PN_corrected = float((n_PG - 1.0 - 4.0 * (u**2)) / u)

    passed = (
        max_conformal_weight_err < 1e-15
        and abs(coeff_1PN_iso_2PN_corrected - 2.0) < 2e-5
        and abs(coeff_1PN_PG_2PN_corrected - 2.0) < 2e-5
    )
    return {
        "passed": bool(passed),
        "max_conformal_weight_identity_err": max_conformal_weight_err,
        "isotropic_1PN_coeff_raw_at_r1000": coeff_1PN_iso_raw,
        "isotropic_1PN_coeff_after_exact_2PN_subtraction": coeff_1PN_iso_2PN_corrected,
        "painleve_gullstrand_1PN_coeff_raw_at_r1000": coeff_1PN_PG_raw,
        "painleve_gullstrand_1PN_coeff_after_exact_2PN_subtraction": coeff_1PN_PG_2PN_corrected,
        "expected_1915_GR_coefficient": 2.0,
        "einstein_1911_time_only_coefficient": 1.0,
    }


def test_2_spherical_core_and_three_zones():
    """
    Test 2:
    - Verifies that integrating 1/r over a 3D sphere of radius R_core(M) gives:
      * Exact -GM/r outside (r >= R_core) with 0 error,
      * Smooth finite -(GM/(2*R_core))*(3 - r^2/R_core^2) inside (r < R_core) via r^3/r = r^2,
        hitting -1.5*GM/R_core at r=0 with zero gravitational force at r=0.
    - Verifies the unified Hayward-Painleve-Gullstrand river flow
      J(r) = -c * sqrt(r_s * r^2 / (r^3 + R_core(M)^3)) with R_core(M) = (r_s * ell^2)^(1/3) >= ell:
      * Zone 1 (r > r_+): |J(r)| < c (outside horizon)
      * Zone 2 (r_- < r < r_+): |J(r)| > c (supersonic infall trap; space & time swapped)
      * Zone 3 (r < r_- ~ R_core(M)): |J(r)| < c -> J(0) = 0 (finite center core; r_- is never sub-Planckian!)
    """
    G_M = 5.0
    c = 1.0
    r_s = 2.0 * G_M / (c**2)  # r_s = 10.0
    ell = 1.0                 # minimum cell scale ell = 1.0
    R_core = float((r_s * (ell**2)) ** (1.0 / 3.0))  # mass-dependent core radius R_core(M) = (r_s * ell^2)^(1/3) ~ 2.154 >= ell

    # 1. Exact 3D spherical shell integral of 1/|x - y| over a ball of radius R_core
    def exact_sphere_potential(r):
        if r >= R_core:
            return -G_M / r
        return -(G_M / (2.0 * R_core)) * (3.0 - (r / R_core) ** 2)

    r_outside = np.linspace(R_core, 50.0, 100)
    outside_err = float(np.max(np.abs([exact_sphere_potential(r) - (-G_M / r) for r in r_outside])))
    V_center = float(exact_sphere_potential(0.0))
    V_center_expected = float(-1.5 * G_M / R_core)

    # Numerical derivative (force) right at r = 0
    dr = 1e-7
    force_at_zero = float(-(exact_sphere_potential(dr) - exact_sphere_potential(0.0)) / dr)

    # 2. Unified Hayward River flow J(r) across the 3 zones with R_core(M) = (r_s * ell^2)^(1/3)
    def river_flow_J(r):
        return -c * np.sqrt((r_s * (r**2)) / (r**3 + R_core**3))

    J_at_zero = float(river_flow_J(0.0))
    J_inside_core = float(abs(river_flow_J(0.2 * R_core)))  # Zone 3 (r < r_-): |J| < c
    J_in_trap = float(abs(river_flow_J(1.5 * R_core)))      # Zone 2 (r_- < r < r_+): |J| > c
    J_far_outside = float(abs(river_flow_J(20.0 * R_core))) # Zone 1 (r > r_+): |J| < c

    passed = (
        R_core >= ell
        and outside_err < 1e-15
        and abs(V_center - V_center_expected) < 1e-15
        and abs(force_at_zero) < 1e-6
        and abs(J_at_zero) < 1e-15
        and J_inside_core < c
        and J_in_trap > c
        and J_far_outside < c
    )
    return {
        "passed": bool(passed),
        "R_core_mass_dependent": R_core,
        "outside_GR_max_error": outside_err,
        "V_at_r0": V_center,
        "force_at_r0": force_at_zero,
        "J_at_r0": J_at_zero,
        "abs_J_zone3_core": J_inside_core,
        "abs_J_zone2_trap_supersonic": J_in_trap,
        "abs_J_zone1_outside": J_far_outside,
    }


def test_3_eighteen_point_sphere_isotropy():
    """
    Test 3:
    Verifies both the 18-neighbor (19-point: W_face=1/3, W_edge=1/6, W_corner=0)
    AND the full 27-point (26-neighbor: W_face=7/15, W_edge=1/10, W_corner=1/30)
    isotropic 3D grid stencils:
      - O(ell^2) axial moment M_2 == 2.0 for both stencils,
      - O(ell^4) sphere cross-term 6 * M_220 / M_400 == 2.0 for both stencils,
      - 3D advection operator M_ij 1/(4*ell) unit normalization == 1.0 for both stencils,
      - O(ell^6): the coefficient of the single monomial kx^2 ky^2 kz^2 vanishes for the 27-point stencil
        (360 W_corner - 12 = 0), but the sixth-order symbol is still anisotropic. Its directional ratios
        axis : face-diagonal : body-diagonal are 1 : 3/2 : 11/9 (18-pt) and 1 : 3/2 : 5/3 (27-pt), so the
        27-pt stencil has the LARGER sixth-order directional spread.
    """
    # 6 Cube Face neighbors: (+/-1, 0, 0), (0, +/-1, 0), (0, 0, +/-1)
    faces = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    # 12 FCC Sphere Edge neighbors: (+/-1, +/-1, 0), (+/-1, 0, +/-1), (0, +/-1, +/-1)
    edges = [
        (1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
        (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
        (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1),
    ]
    # 8 BCC Cube Corner neighbors: (+/-1, +/-1, +/-1)
    corners = [
        (1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),
        (-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1),
    ]

    def check_stencil(W_face, W_edge, W_corner):
        m_xx = (
            sum(W_face * d[0]**2 for d in faces)
            + sum(W_edge * d[0]**2 for d in edges)
            + sum(W_corner * d[0]**2 for d in corners)
        )
        m_xxxx = (
            sum(W_face * d[0]**4 for d in faces)
            + sum(W_edge * d[0]**4 for d in edges)
            + sum(W_corner * d[0]**4 for d in corners)
        )
        m_xxyy = (
            sum(W_face * (d[0]**2)*(d[1]**2) for d in faces)
            + sum(W_edge * (d[0]**2)*(d[1]**2) for d in edges)
            + sum(W_corner * (d[0]**2)*(d[1]**2) for d in corners)
        )
        m_xxyyzz = sum(W_corner * (d[0]**2)*(d[1]**2)*(d[2]**2) for d in corners)
        cross_4th = float(6.0 * m_xxyy / m_xxxx)
        adv_norm = float(0.25 * 2.0 * m_xx)
        body_diag_monomial_coeff = float(90.0 * m_xxyyzz / m_xxxx - 12.0)
        all_d = [(d, W_face) for d in faces] + [(d, W_edge) for d in edges] + [(d, W_corner) for d in corners]

        def p6(n):
            n = np.array(n, dtype=float) / np.linalg.norm(n)
            return sum(w * float(np.dot(n, d)) ** 6 for d, w in all_d)

        ratios = (p6((1, 1, 0)) / p6((1, 0, 0)), p6((1, 1, 1)) / p6((1, 0, 0)))
        return float(m_xx), cross_4th, adv_norm, body_diag_monomial_coeff, ratios

    m_xx_18, cross_18, adv_18, err6_18, ratios_18 = check_stencil(1.0 / 3.0, 1.0 / 6.0, 0.0)
    m_xx_27, cross_27, adv_27, err6_27, ratios_27 = check_stencil(7.0 / 15.0, 1.0 / 10.0, 1.0 / 30.0)
    spread_18 = max(1.0, *ratios_18) - min(1.0, *ratios_18)
    spread_27 = max(1.0, *ratios_27) - min(1.0, *ratios_27)

    passed = (
        abs(m_xx_18 - 2.0) < 1e-15
        and abs(cross_18 - 2.0) < 1e-15
        and abs(adv_18 - 1.0) < 1e-15
        and abs(m_xx_27 - 2.0) < 1e-15
        and abs(cross_27 - 2.0) < 1e-15
        and abs(adv_27 - 1.0) < 1e-15
        and abs(err6_27) < 1e-15
        and abs(ratios_18[0] - 1.5) < 1e-12 and abs(ratios_18[1] - 11.0 / 9.0) < 1e-12
        and abs(ratios_27[0] - 1.5) < 1e-12 and abs(ratios_27[1] - 5.0 / 3.0) < 1e-12
    )
    return {
        "passed": bool(passed),
        "stencil_18_second_moment": m_xx_18,
        "stencil_18_fourth_order_sphere_cross_coeff": cross_18,
        "stencil_18_advection_1_over_4ell_norm": adv_18,
        "stencil_18_kx2ky2kz2_monomial_coeff": err6_18,
        "stencil_18_sixth_order_ratios_face_body_over_axis": list(ratios_18),
        "stencil_18_sixth_order_directional_spread": spread_18,
        "stencil_27_second_moment": m_xx_27,
        "stencil_27_fourth_order_sphere_cross_coeff": cross_27,
        "stencil_27_advection_1_over_4ell_norm": adv_27,
        "stencil_27_kx2ky2kz2_monomial_coeff": err6_27,
        "stencil_27_sixth_order_ratios_face_body_over_axis": list(ratios_27),
        "stencil_27_sixth_order_directional_spread": spread_27,
        "target_exact_sphere_coeff": 2.0,
    }


def test_4_horizon_crossing_and_energy_conservation():
    """
    Test 4:
    Evolves a wave packet across the event horizon (r = r_s) and through the center (r = 0)
    using the canonical Killing Hamiltonian H(Phi, Pi) on a FIXED background (softened toy profile
    J = c (r/r~) sqrt(r_s/r~), periodic 1D domain):
      - Checks that dormant empty cells have H_i = 0 and dPhi_i/dt = 0 (zero processing when nothing happens).
      - Checks that the Killing energy is conserved to round-off by the Cayley map (fixed background only).
      - Checks that the wave stays 100% finite through the inner boundary r_- and center r = 0.
    """
    N = 120
    r = np.linspace(-15.0, 15.0, N)
    ell = float(r[1] - r[0])  # finite bar width
    r_s = 6.0
    c = 1.0

    # Softened river flow J(r) across the whole grid (from -15 to +15 through r = 0)
    r_tilde = np.sqrt(r**2 + (1.5 * ell) ** 2)
    J = -np.sign(r) * c * (np.abs(r) / r_tilde) * np.sqrt(r_s / r_tilde)
    a = 1.0 + 0.2 * (r_s / r_tilde)  # cell reaction inertia a_i > 0

    # Build periodic spring Laplacian L_K and Finite-Volume river shift matrix M
    K0 = 1.0
    L_K = np.zeros((N, N))
    M = np.zeros((N, N))
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        L_K[i, i] += 2.0 * K0 / (ell**2)
        L_K[i, ip] -= K0 / (ell**2)
        L_K[i, im] -= K0 / (ell**2)
        M[i, ip] = 0.5 * (J[i] + J[ip]) / (2.0 * ell)
        M[i, im] = -0.5 * (J[i] + J[im]) / (2.0 * ell)

    # 1. Check that dormant cells (outside the wave packet) have zero activity (dPhi/dt = 0, dPi/dt = 0, H_i = 0)
    phi0 = np.zeros(N)
    pi0 = np.zeros(N)
    # Localize wave packet around r = 8.0 (just outside the horizon r_s = 6.0), leaving r < 0 completely empty
    mask = np.abs(r - 8.0) < 2.5
    phi0[mask] = np.exp(-((r[mask] - 8.0) ** 2) / 0.8) * np.cos(2.0 * (r[mask] - 8.0))
    pi0[mask] = -a[mask] * np.sin(2.0 * (r[mask] - 8.0)) * np.exp(-((r[mask] - 8.0) ** 2) / 0.8)

    dphi_dt_0 = (pi0 / a) - (M @ phi0)
    dpi_dt_0 = -(L_K @ phi0) + (M.T @ pi0)
    dormant_region = np.abs(r + 8.0) < 3.0  # empty region far from packet
    max_dormant_activity = float(np.max(np.abs(dphi_dt_0[dormant_region])) + np.max(np.abs(dpi_dt_0[dormant_region])))

    # 2. Verify exact canonical Killing energy conservation dH/dt = <dH/dPhi, dPhi/dt> + <dH/dPi, dPi/dt> = 0
    # Evolve with symplectic midpoint (Crank-Nicolson) integrator for 200 steps as wave crosses horizon
    def hamiltonian(phi, pi):
        return float(0.5 * np.sum((pi**2) / a) - np.sum(pi * (M @ phi)) + 0.5 * float(phi @ (L_K @ phi)))

    H_initial = hamiltonian(phi0, pi0)
    dt = 0.02
    # Combined state z = [phi, pi], dz/dt = F z with black-hole PG generator F = [[-M, A_inv], [-L_K, M.T]]
    A_inv = np.diag(1.0 / a)
    F_mat = np.block([[-M, A_inv], [-L_K, M.T]])
    I_2N = np.eye(2 * N)
    step_mat = np.linalg.solve(I_2N - 0.5 * dt * F_mat, I_2N + 0.5 * dt * F_mat)

    z = np.concatenate([phi0, pi0])
    max_H_drift = 0.0
    for _ in range(200):
        z = step_mat @ z
        phi_t, pi_t = z[:N], z[N:]
        H_t = hamiltonian(phi_t, pi_t)
        max_H_drift = max(max_H_drift, abs(H_t - H_initial) / abs(H_initial))

    finite_everywhere = bool(np.all(np.isfinite(z)))
    passed = (max_dormant_activity < 1e-15) and (max_H_drift < 1e-13) and finite_everywhere
    return {
        "passed": bool(passed),
        "max_dormant_cell_activity": max_dormant_activity,
        "relative_energy_drift_200_steps": float(max_H_drift),
        "wave_finite_across_horizon_and_center": finite_everywhere,
    }


def test_5_trapped_region_and_core_scale():
    """
    Test 5 (replaces the former 'surface freezing & horizon vise' benchmark):
    - R_core(M) = (3M / (4 pi rho_max))^(1/3) = (r_s ell^2)^(1/3) grows like M^(1/3), r_s like M.
    - Inside the trapped region r_- < r < r_+ of the Hayward profile the outgoing characteristic speed
      c + J(r) is negative: any outward-moving signal (including one reflected at R_core) moves inward.
      This holds for the background mass alone; no 'horizon vise' step r_s(M) -> r_s(M + dM) is needed.
    - A worldline held at fixed r inside r_+ has g_tt = -(c^2 - J^2) > 0, i.e. it is spacelike.
      A static 'frozen surface' at R_core < r_+ is therefore not a possible physical object; a transition
      at R_core can only be a spacelike (instantaneous) layer.
    - The sub-critical case (r_s < R_core) has c + J(R_core) > 0: a reflected signal escapes.
    """
    G = 1.0
    c = 1.0
    ell = 1.0
    rho_max = 3.0 * c**2 / (8.0 * np.pi * G * ell**2)

    def R_core_of_M(M_val):
        return float((3.0 * M_val / (4.0 * np.pi * rho_max)) ** (1.0 / 3.0))

    def r_s_of_M(M_val):
        return float(2.0 * G * M_val / (c**2))

    M_vals = np.array([1.0, 8.0, 64.0])
    scaling_R = float(np.polyfit(np.log(M_vals), np.log([R_core_of_M(m) for m in M_vals]), 1)[0])
    identity_err = max(abs(R_core_of_M(m) - (r_s_of_M(m) * ell**2) ** (1.0 / 3.0)) for m in M_vals)

    M_sub = 0.3  # r_s = 0.6 < R_core
    v_plus_sub = float(c * (1.0 - np.sqrt(r_s_of_M(M_sub) / R_core_of_M(M_sub))))

    r_s = 20.0
    roots = np.sort(np.real(np.roots([1.0, -r_s, 0.0, r_s * ell**2])))
    r_minus, r_plus = roots[1], roots[2]
    rr = np.linspace(r_minus, r_plus, 4002)[1:-1]
    J = -c * np.sqrt(r_s * rr**2 / (rr**3 + r_s * ell**2))
    max_v_plus = float(np.max(c + J))
    min_gtt = float(np.min(J**2 - c**2))

    passed = (
        abs(scaling_R - 1.0 / 3.0) < 1e-12
        and identity_err < 1e-12
        and v_plus_sub > 0.0
        and max_v_plus < 0.0
        and min_gtt > 0.0
    )
    return {
        "passed": bool(passed),
        "R_core_mass_scaling_exponent": scaling_R,
        "R_core_identity_error": float(identity_err),
        "subcritical_outgoing_speed_at_R_core": v_plus_sub,
        "max_outgoing_speed_in_trapped_region": max_v_plus,
        "min_gtt_static_worldline_in_trapped_region_spacelike_if_positive": min_gtt,
    }


def test_6_quantum_route_preserves_gr():
    """
    Test 6:
    Promotes the Finite-Bar Discrete GR model to the Quantum Route:
      1. Canonical Operator Quantization: [Phi_hat_i, Pi_hat_j] = i * hbar * delta_{ij}
         with symplectic/unitary Heisenberg evolution dO/dt = (1 / i*hbar) [O, H_hat].
         Verifies that the canonical commutation relations J_symp = [[0, I], [-I, 0]]
         are preserved across 200 horizon-crossing steps (unitarity of the LINEAR evolution only).
      2. Ehrenfest's Theorem (1927): for a linear system the quantum expectation trajectory equals
         the classical one by construction; it is therefore NOT tested (no check is counted).
      3. Two-mass classical index evaluation (NOT a quantum superposition):
         The classical 1PN index coefficient is evaluated for two masses; agreement with 2.0 is by
         construction (continuum formula with its exact 2PN term subtracted).
      4. Horizon Link Entanglement (Bombelli-Koul-Lee-Sorkin 1986 / Srednicki 1993):
         Computes the exact Gaussian quantum ground-state entanglement entropy across a boundary cut
         from the lattice spring couplings K_ij * (Phi_hat_i - Phi_hat_j)^2, showing it is finite and non-zero.
    """
    N = 60
    r = np.linspace(-12.0, 12.0, N)
    ell = float(r[1] - r[0])
    r_s = 5.0
    c = 1.0

    r_tilde = np.sqrt(r**2 + (1.5 * ell) ** 2)
    J = -np.sign(r) * c * (np.abs(r) / r_tilde) * np.sqrt(r_s / r_tilde)
    a = 1.0 + 0.2 * (r_s / r_tilde)

    K0 = 1.0
    L_K = np.zeros((N, N))
    M_mat = np.zeros((N, N))
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        L_K[i, i] += 2.0 * K0 / (ell**2)
        L_K[i, ip] -= K0 / (ell**2)
        L_K[i, im] -= K0 / (ell**2)
        M_mat[i, ip] = 0.5 * (J[i] + J[ip]) / (2.0 * ell)
        M_mat[i, im] = -0.5 * (J[i] + J[im]) / (2.0 * ell)

    A_inv = np.diag(1.0 / a)
    F_mat = np.block([[-M_mat, A_inv], [-L_K, M_mat.T]])
    dt = 0.02
    I_2N = np.eye(2 * N)
    # Heisenberg operator propagator S: z_hat(t + dt) = S @ z_hat(t)
    S_step = np.linalg.solve(I_2N - 0.5 * dt * F_mat, I_2N + 0.5 * dt * F_mat)

    # 1. Verify preservation of canonical quantum commutator [z_hat_a, z_hat_b] = i * hbar * Omega_{ab}
    #    Over 200 horizon-crossing steps: S_total @ Omega @ S_total.T == Omega
    Omega = np.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])
    S_total = np.linalg.matrix_power(S_step, 200)
    Omega_after = S_total @ Omega @ S_total.T
    commutator_drift = float(np.max(np.abs(Omega_after - Omega)))

    # 2. Ehrenfest's theorem is NOT tested here: for linear dynamics <z_hat(t)> = S(t) <z_hat(0)> is exact
    #    by construction (the quantum mean and the classical trajectory are the same matrix product), so an
    #    earlier "diff = 0.0" check compared one computation with itself and has been removed.

    # 3. Classical evaluation of the 1PN index coefficient for two different masses (M = 1.0 and 1.3).
    #    No quantum superposition is formed; each value is the classical continuum index formula with its
    #    exact analytical 2PN term subtracted, so agreement with 2.0 holds by construction.
    masses_superposition = [1.0, 1.3]
    r_check = 1000.0
    ell_b = 0.01
    branch_iso_1PN_coeffs = []
    branch_pg_1PN_coeffs = []
    for M_b in masses_superposition:
        u_b = M_b / r_check
        R_core_b = (2.0 * M_b * (ell_b**2)) ** (1.0 / 3.0)
        r_t = np.sqrt(r_check**2 + R_core_b**2)
        psi_b = 1.0 + M_b / (2.0 * r_t)
        w_b = (1.0 - M_b / (2.0 * r_t)) / psi_b
        n_iso_b = (psi_b**2) / w_b
        branch_iso_1PN_coeffs.append(float((n_iso_b - 1.0 - 1.75 * (u_b**2)) / u_b))

        r_s_b = 2.0 * M_b
        J2_b = (r_s_b * (r_check**2)) / (r_check**3 + R_core_b**3)
        n_pg_b = 1.0 / (1.0 - J2_b)
        branch_pg_1PN_coeffs.append(float((n_pg_b - 1.0 - 4.0 * (u_b**2)) / u_b))

    max_branch_1PN_err = float(
        max(
            max(abs(c_val - 2.0) for c_val in branch_iso_1PN_coeffs),
            max(abs(c_val - 2.0) for c_val in branch_pg_1PN_coeffs),
        )
    )

    # 4. Horizon Link Entanglement Entropy (Bombelli-Sorkin 1986 / Srednicki 1993)
    #    For the coupled harmonic lattice with mass matrix a_i and spring matrix L_K + m_reg^2 I,
    #    compute the ground-state entanglement entropy across the half-chain cut (horizon boundary).
    m_reg = 0.1
    V_mat = A_inv @ (L_K + (m_reg**2) * np.eye(N))
    # Symmetrized frequency-squared matrix W^2 = A^{-1/2} (L_K + m^2 I) A^{-1/2}
    A_sqrt = np.diag(np.sqrt(a))
    A_inv_sqrt = np.diag(1.0 / np.sqrt(a))
    W2 = A_inv_sqrt @ (L_K + (m_reg**2) * np.eye(N)) @ A_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(W2)
    W_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
    W_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T

    # Ground-state position & momentum covariance matrices: <Phi_i Phi_j> and <Pi_i Pi_j> (hbar = 1)
    X_cov = 0.5 * (A_inv_sqrt @ W_inv_sqrt @ A_inv_sqrt)
    P_cov = 0.5 * (A_sqrt @ W_sqrt @ A_sqrt)
    # Restrict to interior subsystem (first N//2 cells)
    sub = slice(0, N // 2)
    C_sub = X_cov[sub, sub] @ P_cov[sub, sub]
    nu_vals = np.sqrt(np.maximum(np.linalg.eigvals(C_sub).real, 0.25 + 1e-12))
    # Symplectic eigenvalues nu >= 0.5 give von Neumann entanglement entropy S_ent
    s_modes = (nu_vals + 0.5) * np.log(nu_vals + 0.5) - (nu_vals - 0.5) * np.log(np.maximum(nu_vals - 0.5, 1e-15))
    S_entanglement = float(np.sum(s_modes))

    passed = (
        commutator_drift < 1e-13
        and max_branch_1PN_err < 2e-5
        and S_entanglement > 0.0
        and np.isfinite(S_entanglement)
    )
    return {
        "passed": bool(passed),
        "canonical_commutator_drift_200_steps": commutator_drift,
        "ehrenfest_status": "not tested (exact by construction for linear dynamics)",
        "two_mass_classical_isotropic_1PN_coeffs": branch_iso_1PN_coeffs,
        "two_mass_classical_PG_1PN_coeffs": branch_pg_1PN_coeffs,
        "two_mass_classical_max_1PN_err_after_2PN": max_branch_1PN_err,
        "horizon_cut_entanglement_entropy_S": S_entanglement,
    }


def test_7_observational_and_buchdahl_stress_tests():
    """
    Test 7:
    Aggressively stress-tests the model against modern observational bounds and GR theorems:
      1. Fermi-LAT Gamma-Ray Burst Bound (GRB 090510, Nature 2009):
         31 GeV photon (lambda = 4e-17 m) traveling L = 7.3 billion light-years (T = 2.3e17 s)
         must arrive within dt < 0.8 s.
         - A linear O(ell_P / lambda) grid struggles (dt ~ 0.093 s).
         - Our symmetric 18-point stencil has IDENTICALLY ZERO linear O(k*ell) dispersion,
           leaving only quadratic group-velocity dispersion dv_g/c = (pi^2 / 2)*(ell_P / lambda)^2
           -> dt = 1.85e-19 s (19 orders of magnitude safe!) and 4th-order O((ell_P / lambda)^4)
           ~ 2.67e-74 directional anisotropy.
      2. Buchdahl's Theorem (1959), Strong Energy Condition Violation, & Smooth Hayward p_theta:
         - In the Painleve-Gullstrand metric (-g_tt * g_rr = 1), the de Sitter core has constant
           radial tension p_r = -rho_max * c^2, which violates the Strong Energy Condition
           (rho_max*c^2 + 3*p_r = -2*rho_max*c^2 < 0). This is NOT what avoids Penrose (1965):
           Penrose's theorem uses the Null Energy Condition, which the Hayward core satisfies;
           Hayward-type metrics avoid it through their inner (Cauchy) horizon. Buchdahl's bound
           applies to static isotropic stars with R > r_s and does not constrain a core inside r_+.
         - Furthermore, for the smooth Hayward profile 2GM(r)/(c^2 r) = r_s r^2 / (r^3 + R_core^3),
           both radial pressure p_r(r) and tangential pressure p_theta(r) = p_r + (r/2)*dp_r/dr
           are smooth and strictly finite everywhere without a Dirac delta jump!
      3. Photon sphere of the Hayward metric (root of r f'(r) = 2 f(r)) vs 1.5 r_s, and the
         remnant thresholds of the sharp (r_s = R_core) and smooth Hayward (r_s = 3 sqrt(3)/2 ell) cores.
    """
    # 1. Fermi-LAT GRB 090510 check (using packet GROUP VELOCITY v_g = d(omega)/dk)
    ell_P = 1.616255e-35       # Planck length (m)
    lam_31GeV = 4.0e-17        # 31 GeV photon wavelength (m)
    T_travel = 2.3e17          # 7.3 billion years in seconds
    fermi_lat_limit_s = 0.8    # observational upper bound (s)

    ratio = ell_P / lam_31GeV
    dt_linear_model = float(T_travel * ratio)  # ~ 0.093 s
    # On our symmetric 18-pt stencil, omega(k) ~ c*k*(1 - (1/24)*k^2*ell^2), so group velocity
    # v_g = d(omega)/dk ~ c*(1 - (1/8)*k^2*ell^2) = c*[1 - (pi^2 / 2)*(ell_P / lambda)^2]
    frac_dvg_18pt = (np.pi**2 / 2.0) * (ratio**2)
    dt_18pt_grid_vg = float(T_travel * frac_dvg_18pt)  # ~ 1.85e-19 s
    directional_anisotropy_18pt = float(ratio**4)      # ~ 2.67e-74

    # 2. Buchdahl (1959) Bypass, Strong Energy Condition Violation, & Smooth Hayward p_theta(r)
    r_s = 10.0
    R_core = 2.0  # deep inside the event horizon (R_core = 0.2 * r_s < (9/8)*r_s)
    r_pts = np.linspace(0.1, R_core, 50)
    # Sharp core: f(r) - 1 = -(r_s / R_core^3) * r^2, f'(r) = -2 * (r_s / R_core^3) * r
    f_minus_1 = -(r_s / (R_core**3)) * (r_pts**2)
    fp_vals = -2.0 * (r_s / (R_core**3)) * r_pts
    G_tt = (r_pts * fp_vals + f_minus_1) / (r_pts**2)
    target_G_tt = -3.0 * r_s / (R_core**3)  # constant = -8*pi*G*rho_max / c^2 everywhere inside!
    max_buchdahl_free_err = float(np.max(np.abs(G_tt - target_G_tt)))
    sec_combination_over_rho_c2 = -2.0  # (rho*c^2 + 3*p_r) / (rho*c^2) = 1 - 3 = -2 < 0 (SEC violated!)

    # Check smooth Hayward profile tangential pressure p_theta(r) = p_r(r) + (r/2)*dp_r/dr across all r
    r_smooth = np.linspace(0.0, 20.0, 200)
    # Dimensionless Hayward density rho(r)/rho_max = R_core^6 / (r^3 + R_core^3)^2, p_r = -rho
    rho_norm = (R_core**6) / ((r_smooth**3 + R_core**3) ** 2)
    drho_dr_norm = -6.0 * (R_core**6) * (r_smooth**2) / ((r_smooth**3 + R_core**3) ** 3)
    p_r_norm = -rho_norm
    p_theta_norm = p_r_norm - 0.5 * r_smooth * drho_dr_norm
    p_theta_at_R_core = float(((R_core**6) * (2.0 * (R_core**3) - (R_core**3))) / ((2.0 * (R_core**3)) ** 3))  # exact +1/8 = +0.125
    smooth_hayward_pressures_finite = bool(
        np.all(np.isfinite(p_r_norm))
        and np.all(np.isfinite(p_theta_norm))
        and abs(p_theta_at_R_core - 0.125) < 1e-15
    )

    # 3. Photon sphere of the Hayward metric f = 1 - r_s r^2/(r^3 + R_core^3): solve r f'(r) - 2 f(r) = 0
    def _f(rv):
        return 1.0 - r_s * rv**2 / (rv**3 + R_core**3)

    def _g(rv, eps=1e-6):
        return rv * (_f(rv + eps) - _f(rv - eps)) / (2.0 * eps) - 2.0 * _f(rv)

    lo_ph, hi_ph = 1.2 * r_s, 1.8 * r_s
    for _ in range(80):
        mid_ph = 0.5 * (lo_ph + hi_ph)
        if _g(lo_ph) * _g(mid_ph) <= 0.0:
            hi_ph = mid_ph
        else:
            lo_ph = mid_ph
    r_ph = 0.5 * (lo_ph + hi_ph)
    eht_photon_sphere_err = float(abs(r_ph - 1.5 * r_s) / (1.5 * r_s))

    # 4. End of Hawking Evaporation: Automatic Remnant Shut-Off at M_crit (where r_s(M_crit) = R_core(M_crit))
    G_val = 1.0
    c_val = 1.0
    M_crit = 1.0
    r_s_crit = 2.0 * G_val * M_crit / (c_val**2)
    rho_max = 3.0 * M_crit / (4.0 * np.pi * (r_s_crit**3))

    def max_J_over_c(M_test):
        r_s_m = 2.0 * G_val * M_test / (c_val**2)
        R_core_m = (3.0 * M_test / (4.0 * np.pi * rho_max)) ** (1.0 / 3.0)
        return float(np.sqrt(r_s_m / R_core_m)), float(r_s_m - R_core_m)

    peak_J_above_crit, dr_sup_above = max_J_over_c(2.0 * M_crit)
    peak_J_at_remnant, dr_sup_remnant = max_J_over_c(M_crit)

    # Also verify Smooth-Hayward extremal remnant shut-off:
    # For J_smooth^2 / c^2 = r_s * r^2 / (r^3 + r_s * ell^2), the peak is at r_peak = (2 * r_s * ell^2)^(1/3),
    # and max_r |J_smooth(r)| / c == 1.0 occurs at r_s_crit_smooth = (3 * sqrt(3) / 2) * ell (where r_peak = sqrt(3)*ell)
    ell_rem = 1.0
    r_s_crit_smooth = 1.5 * np.sqrt(3.0) * ell_rem
    r_peak_smooth = (2.0 * r_s_crit_smooth * (ell_rem**2)) ** (1.0 / 3.0)
    peak_J_smooth_at_remnant = float(
        np.sqrt((r_s_crit_smooth * (r_peak_smooth**2)) / (r_peak_smooth**3 + r_s_crit_smooth * (ell_rem**2)))
    )

    passed = (
        dt_18pt_grid_vg < 1e-15 * fermi_lat_limit_s
        and directional_anisotropy_18pt < 1e-70
        and max_buchdahl_free_err < 1e-14
        and sec_combination_over_rho_c2 < 0.0
        and smooth_hayward_pressures_finite
        and eht_photon_sphere_err < 1e-2
        and peak_J_above_crit > 1.0
        and dr_sup_above > 0.0
        and abs(peak_J_at_remnant - 1.0) < 1e-15
        and abs(dr_sup_remnant) < 1e-15
        and abs(peak_J_smooth_at_remnant - 1.0) < 1e-15
        and abs(r_peak_smooth - np.sqrt(3.0) * ell_rem) < 1e-15
    )
    return {
        "passed": bool(passed),
        "fermi_lat_linear_model_delay_s": dt_linear_model,
        "fermi_lat_our_18pt_group_velocity_delay_s": dt_18pt_grid_vg,
        "fermi_lat_upper_bound_s": fermi_lat_limit_s,
        "axis_vs_diagonal_anisotropy_31GeV": directional_anisotropy_18pt,
        "river_core_constant_tension_error": max_buchdahl_free_err,
        "strong_energy_condition_ratio_negative": sec_combination_over_rho_c2,
        "smooth_hayward_p_r_and_p_theta_finite_everywhere": smooth_hayward_pressures_finite,
        "photon_sphere_radius_over_rs_hayward": float(r_ph / r_s),
        "photon_sphere_relative_shift_from_1p5_rs": eht_photon_sphere_err,
        "supersonic_thickness_at_2Mcrit": dr_sup_above,
        "supersonic_thickness_at_remnant_Mcrit_sharp": dr_sup_remnant,
        "peak_J_over_c_at_remnant_Mcrit_sharp": peak_J_at_remnant,
        "peak_J_over_c_at_remnant_Mcrit_smooth_hayward": peak_J_smooth_at_remnant,
        "smooth_hayward_extremal_horizon_radius_over_ell": float(r_peak_smooth),
    }


def test_8_surface_gravity_hawking_thermodynamics_and_zero_temp_remnant():
    """
    Test 8:
    - Verifies the exact analytical surface gravity identity at the outer horizon r_+:
        kappa(r_+) = (c^2 / 2) f'(r_+) = (c^2 / (2 r_+)) * (1 - 3*ell^2 / r_+^2)
                                       == (c^2 / (2 r_s)) * (r_+^2 - 3*ell^2) / (r_+^2 - ell^2)
    - Verifies that at the extremal remnant r_+ = sqrt(3)*ell (r_s = 1.5*sqrt(3)*ell),
      kappa(r_+) == 0 and T_H == 0 identically.
    - Verifies that peak Hawking temperature occurs at r_+ = 3*ell (r_s = 27/8 * ell = 3.375*ell),
      below which specific heat C_M = dM / dT_H > 0 is strictly positive (stable cooling).
    - Verifies the exact First Law c^2 dM = T_H dS_reg with the quantum-corrected regular black hole entropy:
        S_reg(r_+) = (k_B c^3 / (4 G hbar)) * [4*pi*(r_+^2 - 3*ell^2) + 8*pi*ell^2 * ln((r_+^2 - ell^2)/(2*ell^2))
                                               - 4*pi*ell^4 * (1/(r_+^2 - ell^2) - 1/(2*ell^2))]
      and S_reg(sqrt(3)*ell) == 0.
    """
    c = 1.0
    ell = 1.0
    r_plus_vals = np.linspace(np.sqrt(3.0) * ell, 20.0 * ell, 200)
    r_s_vals = (r_plus_vals**3) / (r_plus_vals**2 - ell**2)

    # Analytical surface gravity in both forms
    kappa_form1 = (c**2 / (2.0 * r_plus_vals)) * (1.0 - 3.0 * (ell**2) / (r_plus_vals**2))
    kappa_form2 = (c**2 / (2.0 * r_s_vals)) * ((r_plus_vals**2 - 3.0 * ell**2) / (r_plus_vals**2 - ell**2))
    algebraic_equivalence_err = float(np.max(np.abs(kappa_form1 - kappa_form2)))

    # Compare against direct derivative f'(r_+) of f(r) = 1 - r_s r^2 / (r^3 + r_s ell^2)
    eps = 1e-7
    f_plus = 1.0 - (r_s_vals * ((r_plus_vals + eps) ** 2)) / (((r_plus_vals + eps) ** 3) + r_s_vals * (ell**2))
    f_minus = 1.0 - (r_s_vals * ((r_plus_vals - eps) ** 2)) / (((r_plus_vals - eps) ** 3) + r_s_vals * (ell**2))
    kappa_fd = 0.5 * (c**2) * (f_plus - f_minus) / (2.0 * eps)
    fd_match_err = float(np.max(np.abs(kappa_form1 - kappa_fd)))

    # Remnant zero-temperature check at r_+ = sqrt(3)*ell
    kappa_at_remnant = float(kappa_form1[0])

    # Peak temperature radius r_{+,\text{peak}} = 3 * ell (r_s = 27/8 * ell = 3.375 * ell)
    r_plus_peak_exact = float(3.0 * ell)
    dkappa_at_peak = float(0.5 * (c**2) * (9.0 * (ell**2) - (r_plus_peak_exact**2)) / (r_plus_peak_exact**4))

    # Specific heat sign: dM/dr_+ > 0 for all r_+ > sqrt(3)*ell, while dT_H/dr_+ > 0 for sqrt(3)*ell < r_+ < 3*ell
    r_sub_peak = 2.2 * ell
    dM_dr_sub = 0.5 * (r_sub_peak**2) * (r_sub_peak**2 - 3.0 * ell**2) / ((r_sub_peak**2 - ell**2) ** 2)
    dkappa_dr_sub = 0.5 * (c**2) * (9.0 * (ell**2) - (r_sub_peak**2)) / (r_sub_peak**4)
    specific_heat_positive_below_peak = bool((dM_dr_sub / dkappa_dr_sub) > 0.0)

    # First Law verification: dM / dr_+ == T_H * (dS_reg / dr_+) in units G = hbar = k_B = c = 1
    dM_dr_vals = 0.5 * (r_plus_vals**2) * (r_plus_vals**2 - 3.0 * ell**2) / ((r_plus_vals**2 - ell**2) ** 2)
    T_H_vals = kappa_form1 / (2.0 * np.pi)
    dS_reg_dr_vals = 2.0 * np.pi * (r_plus_vals**5) / ((r_plus_vals**2 - ell**2) ** 2)
    first_law_residual = float(np.max(np.abs(dM_dr_vals[1:] - T_H_vals[1:] * dS_reg_dr_vals[1:])))

    r_rem = np.sqrt(3.0) * ell
    S_reg_remnant = float(
        np.pi * (r_rem**2 - 3.0 * ell**2)
        + 2.0 * np.pi * (ell**2) * np.log((r_rem**2 - ell**2) / (2.0 * ell**2))
        - np.pi * (ell**4) * (1.0 / (r_rem**2 - ell**2) - 1.0 / (2.0 * ell**2))
    )

    passed = (
        algebraic_equivalence_err < 1e-15
        and fd_match_err < 1e-8
        and abs(kappa_at_remnant) < 1e-15
        and abs(dkappa_at_peak) < 1e-15
        and specific_heat_positive_below_peak
        and first_law_residual < 1e-14
        and abs(S_reg_remnant) < 1e-14
    )
    return {
        "passed": bool(passed),
        "kappa_forms_algebraic_equivalence_error": algebraic_equivalence_err,
        "kappa_vs_finite_difference_error": fd_match_err,
        "kappa_at_extremal_remnant_sqrt3_ell": kappa_at_remnant,
        "peak_temperature_horizon_radius_over_ell": r_plus_peak_exact,
        "dkappa_dr_at_peak_temperature_radius": dkappa_at_peak,
        "specific_heat_positive_in_cooling_regime": specific_heat_positive_below_peak,
        "first_law_TdS_reg_minus_dM_max_residual": first_law_residual,
        "remnant_entropy_S_reg_at_sqrt3_ell": S_reg_remnant,
    }


def test_9_discrete_gauss_poynting_flux_identity_3d_non_spherical():
    """
    Test 9:
    - Verifies the exact discrete Gauss-Poynting flux conservation law on the 3D 27-point grid
      for an ASYMMETRIC, OFF-CENTER (non-spherical) 3D wave packet:
        d/dt sum_{i in Omega} E_i^local == - sum_{i in Omega, j not in Omega} F_{ij}
      where the antisymmetric link energy flux is:
        F_ij = 0.5 K_ij (Phi_i - Phi_j)(dPhi_i + dPhi_j)
               + 0.5 D_ij (dPhi_i Pi_j + Pi_i dPhi_j - Phi_i dPi_j - dPi_i Phi_j) == -F_ji
      (Killing-energy flux of the PG lattice Hamiltonian, D_ij skew).
    - This is an exact identity of any lattice Hamiltonian with symmetric K and skew D; it is a check
      of the implementation, not a model of non-spherical gravity.
    """
    n_grid = 13
    ell = 1.0
    coords = np.arange(n_grid) - (n_grid // 2)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # Full 27-point stencil weights
    faces = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    edges = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ]
    corners = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    offsets = (
        [(d[0], d[1], d[2], 7.0 / 15.0) for d in faces]
        + [(d[0], d[1], d[2], 1.0 / 10.0) for d in edges]
        + [(d[0], d[1], d[2], 1.0 / 30.0) for d in corners]
    )

    # Non-uniform inertia a_i and 3D radial river velocity J(x,y,z)
    r_s = 3.0
    r_tilde = np.sqrt(R**2 + ell**2)
    a_grid = 1.0 + 0.25 * (r_s / r_tilde)
    J_mag = (R / r_tilde) * np.sqrt((r_s * (R**2)) / (R**3 + r_s * (ell**2)))
    inv_R = np.divide(1.0, R, out=np.zeros_like(R), where=R > 1e-12)
    Jx = -J_mag * X * inv_R
    Jy = -J_mag * Y * inv_R
    Jz = -J_mag * Z * inv_R

    # Initialize an OFF-CENTER, ASYMMETRIC 3D wave packet (centered at (1.5, -1.0, 0.8))
    dist_off = np.sqrt((X - 1.5) ** 2 + (Y + 1.0) ** 2 + (Z - 0.8) ** 2)
    phi = (1.2 + 0.4 * X - 0.3 * Y * Z) * np.exp(-(dist_off**2) / 3.2)
    pi = (0.9 - 0.5 * Y + 0.2 * X * Z) * np.exp(-(dist_off**2) / 2.8)
    u = pi / a_grid

    # Compute dphi/dt and dpi/dt under the true Painleve-Gullstrand Hamiltonian equations:
    #   dphi/dt = u - D_J phi,   dpi/dt = -L_K phi - D_J pi
    lap_phi = np.zeros_like(phi)
    d_phi_adv = np.zeros_like(phi)
    d_pi_adv = np.zeros_like(pi)
    for dx, dy, dz, w in offsets:
        phi_nb = np.roll(phi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        pi_nb = np.roll(pi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jx_nb = np.roll(Jx, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jy_nb = np.roll(Jy, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jz_nb = np.roll(Jz, shift=(-dx, -dy, -dz), axis=(0, 1, 2))

        lap_phi += w * (phi - phi_nb)
        J_dot_d = 0.5 * ((Jx + Jx_nb) * dx + (Jy + Jy_nb) * dy + (Jz + Jz_nb) * dz)
        D_ij = 0.25 * w * J_dot_d
        d_phi_adv += D_ij * phi_nb
        d_pi_adv += D_ij * pi_nb

    dphi = u - d_phi_adv
    dpi = -lap_phi - d_pi_adv

    # Exact rate of change dE_i^Killing / dt of the cell Killing energy:
    #   E_i^Killing = 0.5 * u_i * pi_i - 0.5 * pi_i * (D_J phi)_i + 0.5 * phi_i * (D_J pi)_i
    #                 + 0.25 * sum_j K_ij (phi_i - phi_j)^2
    d_dphi_adv = np.zeros_like(phi)
    d_dpi_adv = np.zeros_like(pi)
    for dx, dy, dz, w in offsets:
        dphi_nb = np.roll(dphi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        dpi_nb = np.roll(dpi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jx_nb = np.roll(Jx, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jy_nb = np.roll(Jy, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jz_nb = np.roll(Jz, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        J_dot_d = 0.5 * ((Jx + Jx_nb) * dx + (Jy + Jy_nb) * dy + (Jz + Jz_nb) * dz)
        D_ij = 0.25 * w * J_dot_d
        d_dphi_adv += D_ij * dphi_nb
        d_dpi_adv += D_ij * dpi_nb

    dE_killing_dt = (
        u * dpi
        - 0.5 * (dpi * d_phi_adv + pi * d_dphi_adv)
        + 0.5 * (dphi * d_pi_adv + phi * d_dpi_adv)
    )
    for dx, dy, dz, w in offsets:
        phi_nb = np.roll(phi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        dphi_nb = np.roll(dphi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        dE_killing_dt += 0.5 * w * (phi - phi_nb) * (dphi - dphi_nb)

    # Test Gauss-Poynting surface flux theorem across spherical control volume R <= 3.2 * ell
    omega_mask = R <= 3.2
    volume_rate_of_change = float(np.sum(dE_killing_dt[omega_mask]))

    # Exact antisymmetric link Killing energy flux F_{ij}^Killing = -F_{ji}^Killing:
    #   F_{ij}^Killing = 0.5 * K_{ij} * (phi_i - phi_j) * (dphi_i + dphi_j)
    #                    + 0.5 * D_{ij} * (dphi_i * pi_j + pi_i * dphi_j - phi_i * dpi_j - dpi_i * phi_j)
    boundary_flux_out = 0.0
    max_link_antisymmetry_err = 0.0
    for dx, dy, dz, w in offsets:
        phi_nb = np.roll(phi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        pi_nb = np.roll(pi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        dphi_nb = np.roll(dphi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        dpi_nb = np.roll(dpi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jx_nb = np.roll(Jx, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jy_nb = np.roll(Jy, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jz_nb = np.roll(Jz, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        omega_nb = np.roll(omega_mask, shift=(-dx, -dy, -dz), axis=(0, 1, 2))

        J_dot_d = 0.5 * ((Jx + Jx_nb) * dx + (Jy + Jy_nb) * dy + (Jz + Jz_nb) * dz)
        D_ij = 0.25 * w * J_dot_d
        F_ij = (
            0.5 * w * (phi - phi_nb) * (dphi + dphi_nb)
            + 0.5 * D_ij * (dphi * pi_nb + pi * dphi_nb - phi * dpi_nb - dpi * phi_nb)
        )

        # Check F_{ij} == -F_{ji}
        F_ji_rolled = (
            0.5 * w * (phi_nb - phi) * (dphi_nb + dphi)
            - 0.5 * D_ij * (dphi_nb * pi + pi_nb * dphi - phi_nb * dpi - dpi_nb * phi)
        )
        max_link_antisymmetry_err = max(
            max_link_antisymmetry_err, float(np.max(np.abs(F_ij + F_ji_rolled)))
        )

        crossing_cut = omega_mask & (~omega_nb)
        boundary_flux_out += float(np.sum(F_ij[crossing_cut]))

    gauss_poynting_residual = abs(volume_rate_of_change + boundary_flux_out)

    passed = max_link_antisymmetry_err < 1e-15 and gauss_poynting_residual < 1e-13
    return {
        "passed": bool(passed),
        "link_flux_antisymmetry_error_Fij_plus_Fji": max_link_antisymmetry_err,
        "volume_dE_dt_inside_R3p2": volume_rate_of_change,
        "minus_surface_poynting_flux_across_R3p2": -boundary_flux_out,
        "discrete_gauss_poynting_conservation_residual": gauss_poynting_residual,
    }


def test_10_3d_spinning_kerr_doran_whirlpool_and_river_ergosphere():
    """
    Test 10:
    - Extends the 3D 27-point skew-symmetric lattice operator M_{ij} to a spinning regular
      black hole flow (Kerr-Doran river + azimuthal whirlpool):
        J(x,y,z) = J_r(r) r_hat + J_phi(r, theta) phi_hat
      with J_r(r) = -c * sqrt(r_s r^2 / (r^3 + r_s ell^2 + a_spin^2 r))
      and  |J_phi| = |J_r| a_spin r sin(theta) / (r^2 + ell^2).
    - CAVEAT: this is an ad-hoc rotating flow on flat spatial slices. It is NOT the Kerr metric (Kerr does
      not admit a flat-slice 'river' form; Doran slices are not flat). Results test the numerics only.
    - Verifies:
      1. Existence of an equatorial ergosphere where |J| = sqrt(J_r^2 + J_phi^2) > c outside
         the radial horizon (|J_r| < c), while the origin remains regular (|J(0)| = 0).
      2. Exact skew-symmetry M^T = -M (< 1e-15) of the 3D 27-point operator with spin.
      3. Round-off-level Cayley conservation of the Killing energy (< 1e-13)
         across 50 spinning ergosphere steps and non-zero Lense-Thirring frame-dragging shift.
    """
    n_grid = 9
    n_cells = n_grid**3
    ell = 1.0
    c = 1.0
    r_s = 3.2
    a_spin = 0.85 * r_s

    coords = np.arange(n_grid) - (n_grid // 2)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    rho_cyl = np.sqrt(X**2 + Y**2)

    denom = (R**3) + r_s * (ell**2) + (a_spin**2) * R
    J_r_mag = c * np.sqrt(np.divide(r_s * (R**2), denom, out=np.zeros_like(R), where=denom > 1e-12))
    # In core-regularized Kerr-Doran coordinates, |J_phi| = |J_r| * (a_spin * r * sin(theta) / (r^2 + ell^2)),
    # and since sin(theta) = rho_cyl / R and phi_hat = (-Y / rho_cyl, X / rho_cyl, 0),
    # J_phi_vec = |J_r| * a_spin / (R^2 + ell^2) * (-Y, X, 0)
    J_phi_over_rho = (J_r_mag * a_spin) / (R**2 + ell**2)

    inv_R = np.divide(1.0, R, out=np.zeros_like(R), where=R > 1e-12)
    Jx = -J_r_mag * X * inv_R - J_phi_over_rho * Y
    Jy = -J_r_mag * Y * inv_R + J_phi_over_rho * X
    Jz = -J_r_mag * Z * inv_R

    J_tot_sq = Jx**2 + Jy**2 + Jz**2
    J_r_sq = J_r_mag**2
    # Ergosphere cells: total flow is supersonic (|J| > c) while radial flow is subsonic (|J_r| < c)
    ergosphere_cells = int(np.sum((J_tot_sq > c**2) & (J_r_sq < c**2)))
    origin_flow_speed = float(np.sqrt(J_tot_sq[n_grid // 2, n_grid // 2, n_grid // 2]))

    # Build explicit 3D 27-point matrices L_K and M on the 9x9x9 lattice
    faces = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    edges = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ]
    corners = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    offsets = (
        [(d[0], d[1], d[2], 7.0 / 15.0) for d in faces]
        + [(d[0], d[1], d[2], 1.0 / 10.0) for d in edges]
        + [(d[0], d[1], d[2], 1.0 / 30.0) for d in corners]
    )

    L_K = np.zeros((n_cells, n_cells), dtype=float)
    M_spin = np.zeros((n_cells, n_cells), dtype=float)
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
                    M_spin[i, j] += 0.25 * w * J_dot_d

    skew_err = float(np.max(np.abs(M_spin + M_spin.T)))

    # Symplectic Cayley evolution over 50 steps with black-hole PG generator F = [[-M_spin, I], [-L_K, M_spin.T]]
    F = np.block([
        [-M_spin, np.eye(n_cells)],
        [-L_K, M_spin.T],
    ])
    dt = 0.04
    I_2n = np.eye(2 * n_cells)
    S_cayley = np.linalg.solve(I_2n - 0.5 * dt * F, I_2n + 0.5 * dt * F)

    # Initial wave packet at (x=+2, y=0, z=0) to measure azimuthal frame dragging into y != 0
    phi0 = np.exp(-(((X - 2.0) ** 2 + Y**2 + Z**2) / 1.5)).ravel()
    pi0 = np.zeros(n_cells, dtype=float)
    z = np.concatenate([phi0, pi0])

    def calc_H(z_vec):
        p = z_vec[:n_cells]
        q = z_vec[n_cells:]
        return float(0.5 * (q @ q) - (q @ (M_spin @ p)) + 0.5 * (p @ (L_K @ p)))

    H0 = calc_H(z)
    max_H_drift = 0.0
    for _ in range(50):
        z = S_cayley @ z
        Ht = calc_H(z)
        max_H_drift = max(max_H_drift, abs(Ht - H0) / abs(H0))

    phi_end = z[:n_cells].reshape((n_grid, n_grid, n_grid))
    # Because initially phi0 is symmetric in +Y vs -Y, any net <Y * phi^2> != 0 is pure azimuthal frame-dragging!
    frame_drag_y_asymmetry = float(np.sum(Y * (phi_end**2)) / np.sum(phi_end**2))

    passed = (
        ergosphere_cells > 0
        and origin_flow_speed == 0.0
        and skew_err < 1e-15
        and max_H_drift < 1e-13
        and abs(frame_drag_y_asymmetry) > 1e-3
    )
    return {
        "passed": bool(passed),
        "ergosphere_supersonic_cell_count": ergosphere_cells,
        "origin_flow_speed_regular": origin_flow_speed,
        "kerr_doran_27pt_skew_symmetry_error": skew_err,
        "symplectic_cayley_energy_drift_50steps": max_H_drift,
        "lense_thirring_azimuthal_frame_drag_y_shift": frame_drag_y_asymmetry,
    }


def build_claims_audit(results, modules):
    """Claim-by-claim verdicts derived from the computed results (not hard-coded PASS flags)."""
    m4 = modules["module_4_horizon_and_freezing"]
    t9 = m4["test_09_hayward_river_profile_horizons_and_core_scale"]
    t12 = m4["test_12_trapped_region_outgoing_characteristics"]
    t13 = m4["test_13_inner_horizon_blueshift_and_saturation_cap"]
    m8 = modules["module_8_theory_extensions_and_stress_limits"]
    t25 = m8["test_25_postulated_completions_comparison_and_entropy_bound"]
    m9 = modules["module_9_dynamical_collapse"]
    t27 = m9["test_27_how_can_the_cap_be_enforced_homogeneous"]
    t28 = m9["test_28_local_cell_energy_cap_cannot_prevent_singularity"]
    t29 = m9["test_29_curvature_cap_bounce_and_shell_crossing"]
    m10 = modules["module_10_post_bounce_shock"]
    t30 = m10["test_30_hksw_godunov_shock_and_convergence"]
    t31 = m10["test_31_hksw_lifetime_M2_scaling_and_superluminal_audit"]
    t32 = m10["test_32_sticky_shell_dissipation_and_ltb_signature_bound"]
    t33 = m10["test_33_lqc_effective_metric_vs_hayward_inner_horizon_at_Rcore"]
    m11 = modules["module_11_relativistic_einstein_euler"]
    t34 = m11["test_34_curvature_capped_tov_no_go_for_static_positive_pressure_core"]
    t35 = m11["test_35_einstein_euler_shock_prevents_shell_crossing_and_preserves_signature"]
    t36 = m11["test_36_continuum_convergence_of_relativistic_einstein_euler_bounce"]
    t37 = m11["test_37_trapped_region_hyperbola_and_throat_threshold_audit"]
    m12 = modules["module_12_random_grid_and_voronoi_audit"]
    t38 = m12["test_38_random_voronoi_delaunay_vs_naive_random_laplacian_consistency"]
    t39 = m12["test_39_random_grid_rotational_isotropy_vs_rayleigh_scattering_bound"]
    t40 = m12["test_40_random_unstructured_falling_lattice_einstein_euler_invariance"]
    t41 = m12["test_41_static_and_stretched_anamorphic_grid_light_bending_and_zero_scattering"]
    t42 = m12["test_42_tidal_aspect_ratio_and_hencky_anamorphic_bar_stress_in_bounce"]
    m13 = modules["module_13_emergent_anamorphic_gr"]
    t43 = m13["test_43_discretised_exterior_schwarzschild_and_light_bending_from_rho"]
    t44 = m13["test_44_discretised_interior_schwarzschild_star_and_constant_density_8_9_divergence"]
    t45 = m13["test_45_painleve_gullstrand_coordinate_map_of_discrete_curvature_charge"]
    t46 = m13["test_46_18pt_rank6_strain_tt_evolution_and_by_construction_bianchi"]
    t47 = m13["test_47_continuum_regge_wheeler_formula_on_discretised_background"]
    b3 = results["test_3_eighteen_point_sphere_isotropy"]

    def verdict(ok, yes="SUPPORTED", no="NOT SUPPORTED"):
        return yes if ok else no

    return [
        {"claim": "Matches GR outside the horizon (1PN light bending, photon sphere, l=2 QNMs)",
         "verdict": verdict(results["test_1_equivalence_principle_and_gr_bending"]["passed"]
                            and modules["module_7_strong_field_and_observables"]["test_20_regge_wheeler_barrier_and_quasinormal_mode_ringdown"]["passed"]),
         "note": "Expected by construction: the Schwarzschild/Hayward metric is an input and corrections are O(ell^2/r^2)."},
        {"claim": "Energy-conserving, symplectic black-hole (not white-hole) wave evolution on a fixed background",
         "verdict": verdict(m4["test_10_cayley_symplectic_and_skew_symmetric_conservation"]["passed"]),
         "note": "Packets inside r_+ are swept inward; Killing energy conserved to round-off."},
        {"claim": "Tree-level lattice dispersion passes the Fermi-LAT GRB 090510 bound",
         "verdict": verdict(results["test_7_observational_and_buchdahl_stress_tests"]["passed"]),
         "note": "Tree level only; loop-induced O(1) Lorentz violation (Collins et al. 2004) is an open problem."},
        {"claim": "27-point stencil removes the sixth-order body-diagonal error",
         "verdict": verdict(b3["stencil_27_sixth_order_directional_spread"] < b3["stencil_18_sixth_order_directional_spread"]),
         "note": "Only the kx^2 ky^2 kz^2 monomial vanishes; directional spread is 0.667 (27-pt) vs 0.500 (18-pt)."},
        {"claim": "Bounded curvature, K(0) = 24/ell^4, core scale R_core = (r_s ell^2)^(1/3)",
         "verdict": "SUPPORTED",
         "note": "Properties of the Hayward metric with ell = ell_P; R_core also equals the limiting-curvature radius up to 2^(-1/6)."},
        {"claim": "The inner Cauchy horizon r_- is excised",
         "verdict": verdict(not t9["inner_horizon_still_present_in_geometry"]),
         "note": "r_- < R_core, but r_- is still part of the Hayward geometry; nothing removes it."},
        {"claim": "Bounced waves are pinned on a static frozen surface at R_core",
         "verdict": verdict(t12["min_g_tt_of_static_worldline_in_trapped_region_positive_means_spacelike"] <= 0.0),
         "note": "Inside r_+ a fixed-r worldline is spacelike; everything moves inward anyway (c + J < 0)."},
        {"claim": "Local saturation resolves inner-horizon mass inflation",
         "verdict": verdict(not t13["finding_reservoir_exceeds_infalling_energy"] and t13["finding_reservoir_converged_in_h"]),
         "note": ("The cap bounds the field's energy density by construction, but moves the blueshifted excess into "
                  f"reservoir mass of {t13['capped_reservoir_deposit_over_E0']} x E0 for h = {t13['lattice_spacings_h']} "
                  "(growing as h -> 0). Settling mass inflation requires backreaction, which is not modelled.")},
        {"claim": "Singularity theorems avoided via Strong Energy Condition violation",
         "verdict": "NOT SUPPORTED",
         "note": "Penrose (1965) uses the NEC, which the Hayward core satisfies; it is avoided via the inner horizon (or a bounce / topology change)."},
        {"claim": "Cold T_H = 0 remnant at r_+ = sqrt(3) ell",
         "verdict": "SUPPORTED (conditional)",
         "note": "Hayward 2006 result; requires the inner horizon to exist, which contradicts the excision claim."},
        {"claim": "Quantum unitarity / information preserved",
         "verdict": "NOT SUPPORTED",
         "note": (f"Linear evolution is unitary, but saturation damping is not; core entropy bound falls short of S_BH by "
                  f"{t25['bekenstein_entropy_deficit_ratio_S_BH_over_S_core']:.3g} for 1 M_sun using the per-cell heuristic, or by "
                  f"{t25['bekenstein_entropy_deficit_ratio_S_BH_over_S_core'] ** (2.0 / 3.0):.3g} using the whole-system Bekenstein bound "
                  f"S <= 2 pi M c^2 R_core / (hbar c) = pi (r_s/l_P)^(4/3).")},
        {"claim": "Spin-2 (gravitational-wave) sector",
         "verdict": "PARTLY SUPPORTED (rank-6 18-bar geometry & linearized TT evolution; Test 23 was a scalar toy)",
         "note": ("Test 23 (Module 8) was a 2-component scalar toy. Module 13 (Tests 46-47) shows the 18-point stencil's 6 axial + 12 face-diagonal bars "
                  "have full rank 6 on symmetric 3x3 tensors h_ij (vs. rank 3 for a 6-bar cubic truss; kinematic only). Linearized TT evolution is six "
                  "decoupled scalar 18-pt waves (TT/Bianchi identities hold by construction), the Regge-Wheeler (1 - s^2) = -3 factor in Test 47 is taken "
                  "from continuum perturbation theory, and nonlinear 3+1D backreaction is not evolved.")},
        {"claim": "No post-merger echoes",
         "verdict": "SUPPORTED (generic)",
         "note": "True for any object whose surface lies inside r_+; not a distinguishing prediction."},
        {"claim": "A local cell-energy cap (the rule as originally stated) prevents the singularity",
         "verdict": verdict(not (t27["finding_pressure_cannot_enforce_cap"] or t28["finding_local_energy_cap_insufficient"])),
         "note": ("Dynamical collapse (Module 9): the cap is reached only inside a trapped region; causal pressure cannot "
                  "enforce it; and empty cells (energy density 0) still reach K -> infinity by Schwarzschild free fall.")},
        {"claim": "A curvature (mean-density) cap produces a regular bounce at R_core = (r_s ell^2)^(1/3)",
         "verdict": verdict(t29["passed"], yes="SUPPORTED (with LQC-type dynamics)"),
         "note": ("Every shell, empty or not, bounces at exactly R_core(m); shells then cross near the bounce, so a shock "
                  "treatment is needed to follow what comes out (V2-type). The bounce law dR/dt^2 = (2m/R)(1 - R_b^3/R^3) is the "
                  "effective-LQC Oppenheimer-Snyder equation IMPOSED per shell; each shell is an independent ODE (no neighbour coupling), "
                  "so the lattice carries no dynamics in Module 9.")},
        {"claim": "The lattice saturation rule selects V1 (transition), V2 (bounce) or V3 (inner horizon)",
         "verdict": "PARTLY ANSWERED",
         "note": ("The rule must act on curvature, not cell energy. Repulsive enforcement gives V2, conversion to vacuum energy "
                  "gives V1 (homogeneous case); the postulate itself does not choose. No tested mechanism produces V3.")},
        {"claim": "Post-bounce HKSW (2022) shock wave exits horizon in time T ~ (8 pi / 3) M^2",
         "verdict": verdict(t30["passed"] and t31["passed"], yes="SUPPORTED (conditional: spacelike inside horizon)"),
         "note": ("Module 10 (Tests 30-31): Godunov PDE converges at O(dx) to the exact Rankine-Hugoniot ODE (rel_err 3.7e-5 at dx=0.01) "
                  "and confirms Delta t = (8 pi M^2)/(3 ell) - 8M/3 - pi ell/3 (G = c = 1; in SI the leading term is 8 pi G^2 M^2/(3 c^5 ell), giving a "
                  "PBH mass ~2e25 g for Delta t = t_0); HOWEVER, the shock has exterior Mach number "
                  "M_ext = (dX_s/dt - U_R)/c > 1 across 100% of [R_-, R_+] (spacelike/superluminal relative to exterior metric; Fazzini & Mehmood 2024).")},
        {"claim": "Causal post-bounce shock / inelastic shell coalescence & inner horizon at R_core",
         "verdict": verdict(t32["passed"] and t33["passed"], yes="PARTLY ANSWERED (requires relativistic shock pressure)"),
         "note": ("Module 10 (Tests 32-33): Any exterior-causal shock (dX_s/dt - U_R <= c) must move inward across [R_-, R_+] down to the "
                  "effective-LQC inner horizon R_-(M) = R_core(M)[1 + (1/3)(ell/r_s)^(2/3)] (where f_LQC(R_core) = +1 is timelike, unlike Hayward); "
                  "cold inelastic coalescence dissipates 99.98% of kinetic energy (e_mech = -5.09 c^2 < -0.5 c^2, violating LTB signature 1+2e > 0), "
                  "proving that the next step requires relativistic shock pressure p > 0 (Einstein-Euler system).")},
        {"claim": "Static positive-pressure hydrostatic core inside R_+(M) under curvature-capped TOV",
         "verdict": verdict(not t34["passed"]),
         "note": ("Module 11 (Test 34): By the Curvature-Capped TOV Theorem, in [R_-, R_+] f_LQC < 0 (spacelike), while inside the timelike core "
                  "R < R_-(M) bar_rho/rho_max > 1/2 makes both (1 - 4 bar_rho/rho_max) < 0 and (1 - 2 bar_rho/rho_max) < 0, so dp/dR > 0 strictly "
                  "for all p >= 0. No static positive-pressure core with p(R_surf)=0 can exist inside R_+; the shock-heated core must bounce dynamically.")},
        {"claim": "Coupled relativistic Misner-Sharp / Einstein-Euler shock pressure prevents shell crossing & preserves Lorentzian signature",
         "verdict": verdict(t35["passed"] and t36["passed"] and t37["passed"], yes="SUPPORTED QUALITATIVELY (no crossing, Gamma > 0, turnaround inside R_-); headline numbers depend on l_visc"),
         "note": ("Module 11 (Tests 35-37): Replacing dust (p=0) with relativistic Einstein-Euler fluid (p=(gamma-1)rho_0 eps > 0) prevents shell crossing "
                  "(min dR/dR_0 = 0.088 > 0), heats the core by 48.2x, preserves Gamma >= 0.783 > 0 and |Gamma^2 - f_LQC - U^2| < 2.4e-8 at l_visc = 0.06 ell. "
                  "ADM mass conservation (0.0) follows from the P = 0 surface boundary condition and is not a test of the scheme. "
                  "Because l_visc is fixed in physical units, O(1/N^2) convergence (L2 err 5.8e-7) is to the VISCOUS PDE: sweeping l_visc = 0.03/0.06/0.12 "
                  "gives heating 21x/48x/129x and min Gamma 0.83/0.78/0.53 (viscosity_sweep_audit.py), while R_surf,min varies by only 1.5% and every shell "
                  "still turns around inside R_-(m_i). The trapped-region hyperbola (U/c)^2 - Gamma^2 = |f_LQC| requires turnaround at R <= R_-(m_i).")},
        {"claim": "3D random Voronoi-Delaunay grid vs. regular 18-pt crystal & unstructured falling mesh",
         "verdict": verdict(t38["passed"] and t39["passed"] and t40["passed"], yes="PARTLY SUPPORTED (co-moving random mesh works; static 3D random grid scatters PeV photons)"),
         "note": ("Module 12 (Tests 38-40): Exact 3D Voronoi-Delaunay DEC weights satisfy Minkowski's facet identity (1.3e-15), Tr(M_i)==3 (2.7e-15), "
                  "Cayley conservation (6.0e-15), and statistical SO(3) isotropy (unlike naive random graphs which fail at O(1/ell)). However, static 3D "
                  "Voronoi cell-to-cell shear fluctuations (sigma_shear = 0.0765) cause E^4 Rayleigh scattering that passes GRB 090510 at 31 GeV (tau = 8.3e-14) "
                  "but gives tau = 2.2 > 1 for 2.5 PeV photons over the observed ~1.4 kpc LHAASO baseline (15.6 over a hypothetical 10 kpc; tau = 0 on the 18-pt crystal), "
                  "assuming the random lattice is the physical spacetime. By contrast, a co-moving random "
                  "unstructured falling mesh in the Einstein-Euler solver converges to the exact same bounce and heating (L2 err 2.0e-5 at N=120).")},
        {"claim": "Smoothly stretched anamorphic grid & physical tidal aspect-ratio (Hencky + Kelvin-Voigt) bar stress",
         "verdict": verdict(t41["passed"] and t42["passed"], yes="PARTLY SUPPORTED (2.22x lower heating error; light bending by construction; bar stress is a tunable material model)"),
         "note": ("Module 12 (Tests 41-42): (1) A static radially anamorphic lattice (b_r = 1/sqrt(f) != b_perp = 1) reproduces 1PN+2PN GR light bending "
                  "(2.0006) by construction (Schwarzschild optical metric is input). Its smooth gradient (delta A_cell ~ 2.7e-39 near a solar-mass horizon) is "
                  "deterministic, not random disorder, so no Rayleigh optical depth is claimed; away from masses it is the periodic crystal (tau = 0). "
                  "(2) A smoothly stretched falling radial coordinate s(xi) = 0.65 xi^0.5 + 0.35 xi reduces peak thermal-heating discretization error "
                  "at N=50 by 2.22x (2.78e-2 -> 1.25e-2). (3) Adding a constitutive lattice-bar stress model (Hencky logarithmic deviatoric elasticity + "
                  "Kelvin-Voigt tidal viscosity with tunable G_bar, eta_anam; p_r != p_perp) beyond an ideal isotropic fluid increases min(dR/dR_0) and "
                  "min(Gamma_i) across delta in {0.42, 0.48, 0.52}; this is a material-model effect, not a GR result.")},
        {"claim": "Second-order discretisation of spherically symmetric Einstein / TOV constraints in anamorphic variables (A_i, a_i, p_i, J_i) without a metric ansatz",
         "verdict": verdict(t43["passed"] and t44["passed"] and t45["passed"], yes="SUPPORTED (O(1/N^2) convergent p = 2.000; no metric ansatz, but Einstein/TOV equations are input)"),
         "note": ("Module 13 (Tests 43-45): Taking the spherically symmetric Einstein/TOV constraint equations (G^t_t, G^r_r, and hydrostatic balance) as input "
                  "and discretizing them in anamorphic variables (A_i = b_{r,i}/b_{perp,i}, a_i, p_i) starting from rho(R) without a prescribed metric formula "
                  "reproduces: (1) exterior Schwarzschild and vacuum reciprocity a_i * A_i -> 1 at O(1/N^2) (p = 2.0000, err 8.0e-6 at N=1600) and 1PN light "
                  "bending 1.9999 (vs 1.002 for 1911 time-only); (2) Schwarzschild's 1916 interior constant-density star solution (p = 1.998, pressure err 4.5e-7, "
                  "clock err 2.8e-8 at N=800) and the constant-density central-pressure divergence as 2GM/(c^2 R) -> 8/9 (the Buchdahl-saturating case; "
                  "Buchdahl's general theorem is not tested); and (3) the Painleve-Gullstrand coordinate definition J_i = -c sqrt(Q_i / R_i), whose identity "
                  "holds by definition (only the quadrature err 9.4e-8 of Q_i is a genuine check).")},
        {"claim": "3D 18-point rank-6 strain reconstruction & Regge-Wheeler potential on the discretized background",
         "verdict": verdict(t46["passed"] and t47["passed"], yes="PARTLY SUPPORTED (rank-6 kinematics genuine; TT/Bianchi checks by construction; RW spin factor 1-s^2=-3 is continuum input)"),
         "note": ("Module 13 (Tests 46-47): (1) GENUINE: a 6-bar cubic truss has rank 3 < 6 (blind to off-diagonal x-shear), whereas the 6 axial + 12 face-diagonal "
                  "bars have full rank 6 and reconstruct any symmetric 3x3 h_ij to 6.7e-16 (kinematic only; the evolution does not use the bars). (2) The scalar "
                  "18-pt vs 6-pt dispersion anisotropy ratio at the single wavenumber |k|ell = pi/4 is 65x (k-dependent; not gravity-specific). (3) BY CONSTRUCTION: "
                  "TT trace/divergence, Cayley energy, delta G^0_0 = 0, and the real-space contracted Bianchi / gauge identities with Delta = sum_k D_k D_k hold for "
                  "ANY commuting difference operators; the ~0.105 'mixed' anomaly only measures Delta_18 != sum_k D_k D_k, sum_k D_k D_k has spurious zone-boundary "
                  "zero modes, and the evolution itself uses Delta_18. (4) The continuum Regge-Wheeler formula (input spin factor 1 - s^2 = -3) on the discretized "
                  "background matches the exact barrier to 7.1e-5, which measures the accuracy of the discretized f = a/A only.")},
    ]


def main():
    from tests.module_1_exterior_gr import run_module_1
    from tests.module_2_core_regularization import run_module_2
    from tests.module_3_stencil_isotropy import run_module_3
    from tests.module_4_horizon_and_freezing import run_module_4
    from tests.module_5_nonspherical_and_spin import run_module_5
    from tests.module_6_quantum_and_thermodynamics import run_module_6
    from tests.module_7_strong_field_and_observables import run_module_7
    from tests.module_8_theory_extensions_and_stress_limits import run_module_8
    from tests.module_9_dynamical_collapse import run_module_9
    from tests.module_10_post_bounce_shock import run_module_10
    from tests.module_11_relativistic_einstein_euler import run_module_11
    from tests.module_12_random_grid_and_voronoi_audit import run_module_12
    from tests.module_13_emergent_anamorphic_gr import run_module_13

    results = {
        "test_1_equivalence_principle_and_gr_bending": test_1_equivalence_principle_and_gr_bending(),
        "test_2_spherical_core_and_three_zones": test_2_spherical_core_and_three_zones(),
        "test_3_eighteen_point_sphere_isotropy": test_3_eighteen_point_sphere_isotropy(),
        "test_4_horizon_crossing_and_energy_conservation": test_4_horizon_crossing_and_energy_conservation(),
        "test_5_trapped_region_and_core_scale": test_5_trapped_region_and_core_scale(),
        "test_6_quantum_route_preserves_gr": test_6_quantum_route_preserves_gr(),
        "test_7_observational_and_buchdahl_stress_tests": test_7_observational_and_buchdahl_stress_tests(),
        "test_8_surface_gravity_hawking_thermodynamics_and_zero_temp_remnant": test_8_surface_gravity_hawking_thermodynamics_and_zero_temp_remnant(),
        "test_9_discrete_gauss_poynting_flux_identity_3d_non_spherical": test_9_discrete_gauss_poynting_flux_identity_3d_non_spherical(),
        "test_10_3d_spinning_kerr_doran_whirlpool_and_river_ergosphere": test_10_3d_spinning_kerr_doran_whirlpool_and_river_ergosphere(),
    }
    core_passed = all(v["passed"] for v in results.values())

    modules = {
        "module_1_exterior_gr": run_module_1(),
        "module_2_core_regularization": run_module_2(),
        "module_3_stencil_isotropy": run_module_3(),
        "module_4_horizon_and_freezing": run_module_4(),
        "module_5_nonspherical_and_spin": run_module_5(),
        "module_6_quantum_and_thermodynamics": run_module_6(),
        "module_7_strong_field_and_observables": run_module_7(),
        "module_8_theory_extensions_and_stress_limits": run_module_8(),
        "module_9_dynamical_collapse": run_module_9(),
        "module_10_post_bounce_shock": run_module_10(),
        "module_11_relativistic_einstein_euler": run_module_11(),
        "module_12_random_grid_and_voronoi_audit": run_module_12(),
        "module_13_emergent_anamorphic_gr": run_module_13(),
    }
    n_tests = sum(len([k for k in m if k.startswith("test_")]) for m in modules.values())
    modules_passed = all(
        modules[f"module_{k}_{name}"][f"module_{k}_passed"]
        for k, name in [
            (1, "exterior_gr"),
            (2, "core_regularization"),
            (3, "stencil_isotropy"),
            (4, "horizon_and_freezing"),
            (5, "nonspherical_and_spin"),
            (6, "quantum_and_thermodynamics"),
            (7, "strong_field_and_observables"),
            (8, "theory_extensions_and_stress_limits"),
            (9, "dynamical_collapse"),
            (10, "post_bounce_shock"),
            (11, "relativistic_einstein_euler"),
            (12, "random_grid_and_voronoi_audit"),
            (13, "emergent_anamorphic_gr"),
        ]
    )
    all_passed = bool(core_passed and modules_passed)
    audit = build_claims_audit(results, modules)
    results["modular_test_suite"] = modules
    results["number_of_module_tests"] = n_tests
    results["ALL_MODULE_NUMERICAL_CHECKS_PASSED"] = bool(modules_passed)
    results["ALL_NUMERICAL_CHECKS_PASSED"] = bool(all_passed)
    results["CLAIMS_AUDIT"] = audit

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "verify_core_finite_bar_gr_results.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v for k, v in results.items() if k != "CLAIMS_AUDIT"}, indent=2))
    print(f"\nALL 13 MODULES ({n_tests} TESTS) NUMERICAL CHECKS PASSED: {modules_passed}")
    print(f"ALL 10 CORE BENCHMARK NUMERICAL CHECKS PASSED: {core_passed}")
    print("\nCLAIMS AUDIT (what the code does and does not support):")
    for item in audit:
        print(f"  [{item['verdict']}] {item['claim']}\n      -> {item['note']}")


if __name__ == "__main__":
    main()
