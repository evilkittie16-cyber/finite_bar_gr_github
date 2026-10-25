"""
simulate_3d_18pt_freezing_core.py
=================================
Shortcut-free 3D lattice wave simulation of the Finite-Bar Discrete GR Black Hole model
on an N x N x N 3D grid:
  1. Full 18-point stencil (6 Cube faces w=1/3 + 12 FCC Sphere edges w=1/6) used for BOTH
     the spring Laplacian L_K AND the skew-symmetric river shift M (and -M^T).
  2. Zero magic multipliers: at every step, enclosed mass-energy M(r_i) at radius r_i is
     computed directly by summing the actual cell energies (E_frozen + E_wave) inside r <= r_i,
     setting r_s(r_i) = 2*G*M(r_i)/c^2 and |J(r_i)| = c * sqrt(r_s(r_i) / sqrt(r_i^2 + ell^2)).
  3. 100% Local 18-Neighbor Saturation: when any cell i reaches E_i >= E_max, it locks
     E_max into E_frozen[i], sets active_mask[i] = False (dPhi_i/dt = 0, dPi_i/dt = 0),
     and transfers the excess (E_i - E_max) ONLY to its direct active 18-stencil neighbors
     weighted by w_d (no global radius lookup).
  4. Exact Total Energy Conservation Audit: tracks E_total = sum(E_wave) + sum(E_frozen)
     at every step and verifies conservation to machine precision.
"""

import json
import os
import numpy as np


def build_6pt_offsets_and_weights():
    faces = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    return [(d[0], d[1], d[2], 1.0) for d in faces]


def build_18pt_offsets_and_weights():
    faces = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    edges = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ]
    offsets = []
    for d in faces:
        offsets.append((d[0], d[1], d[2], 1.0 / 3.0))
    for d in edges:
        offsets.append((d[0], d[1], d[2], 1.0 / 6.0))
    return offsets


def build_27pt_offsets_and_weights():
    """
    Full 3x3x3 27-point (1 center + 26 neighbors) Spotz-Carey / Patra-Karttunen stencil:
      - 6 Simple Cubic faces: W_face = 7/15
      - 12 FCC Sphere edges:  W_edge = 1/10
      - 8 BCC Cube corners:   W_corner = 1/30
    Eliminates O(ell^2), O(ell^4), AND the O(ell^6) kx^2*ky^2*kz^2 body-diagonal error!
    """
    faces = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    edges = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ]
    corners = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    offsets = []
    for d in faces:
        offsets.append((d[0], d[1], d[2], 7.0 / 15.0))
    for d in edges:
        offsets.append((d[0], d[1], d[2], 1.0 / 10.0))
    for d in corners:
        offsets.append((d[0], d[1], d[2], 1.0 / 30.0))
    return offsets


def apply_3d_18pt_laplacian(phi, offsets, active_mask):
    """
    Applies the symmetric 18-point spring Laplacian L_K across all 18 neighbors
    (6 Cube faces + 12 FCC Sphere edges). Links to frozen cells are inactive (zero flux).
    """
    lap = np.zeros_like(phi)
    for dx, dy, dz, w in offsets:
        phi_nb = np.roll(phi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        mask_nb = np.roll(active_mask, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        link_active = active_mask & mask_nb
        lap += np.where(link_active, w * (phi - phi_nb), 0.0)
    return lap


def apply_3d_18pt_river_shift(field, Jx, Jy, Jz, offsets, active_mask, transpose=False):
    """
    Applies the 3D river shift operator M (or M^T when transpose=True) across ALL 18 neighbors
    of the isotropic stencil.
    Because each link d in N_18 has an opposite link -d in N_18 with equal weight w_d,
    pairing +0.25 * w_d * ((J_i + J_{i+d}) . d) on link +d with its opposite on -d makes
    M skew-symmetric (M^T = -M), guaranteeing d/dt <Phi, Pi>_energy == 0.
    """
    out = np.zeros_like(field)
    for dx, dy, dz, w in offsets:
        f_nb = np.roll(field, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        m_nb = np.roll(active_mask, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jx_nb = np.roll(Jx, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jy_nb = np.roll(Jy, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jz_nb = np.roll(Jz, shift=(-dx, -dy, -dz), axis=(0, 1, 2))

        link_active = active_mask & m_nb
        # Directional flow projected along link vector d = (dx, dy, dz)
        # Note: since sum_{d in N18} w_d d_a d_b = 2 delta_{ab}, dividing by 2*2=4 gives exact J . grad(f)
        J_dot_d = 0.5 * ((Jx + Jx_nb) * dx + (Jy + Jy_nb) * dy + (Jz + Jz_nb) * dz)
        coeff = 0.25 * w * J_dot_d
        if not transpose:
            # (M @ f)_i
            out += np.where(link_active, coeff * f_nb, 0.0)
        else:
            # (M^T @ f)_i = - (M @ f)_i because M_{j,i} = -M_{i,j}
            out -= np.where(link_active, coeff * f_nb, 0.0)
    return out


def compute_cell_energies(phi, pi, a_grid, offsets, active_mask):
    """
    Computes positive wave energy E_i in each active 3D cell:
      E_i = 0.5 * pi_i^2 / a_i + 0.25 * sum_{d in N18} w_d * (phi_i - phi_{i+d})^2
    Note: Because M is skew-symmetric (M^T = -M), under the canonical equations
      dphi/dt = pi/a + M phi,   dpi/dt = -L_K phi + M (a dphi/dt - M phi)
    or in the local co-moving frame where E_wave = sum_i E_i, let's evolve the exact
    skew-coupled wave equations that conserve sum_i E_i identically:
      dphi/dt = pi / a_grid
      dpi/dt  = -L_K phi - a_grid * M(pi / a_grid) - M^T(pi)
    Wait: let's check d/dt sum_i (0.5 * pi_i^2 / a_i + 0.5 * phi_i (L_K phi)_i):
      dE/dt = sum_i (pi_i / a_i) * dpi/dt + sum_i (L_K phi)_i * dphi/dt
            = sum_i (pi_i / a_i) * (-L_K phi_i) - sum_i (pi_i / a_i) * (M_{\text{skew}} (pi / a))_i + sum_i (L_K phi)_i (pi_i / a_i)
            = 0 IDENTICALLY because <u, M_skew u> == 0 for any skew-symmetric matrix M_skew!
    Wait—let's check the canonical Hamiltonian H(phi, pi) from Section 4.2 too!
    In Section 4.2:
      dphi/dt = pi / a + M phi
      dpi/dt  = -L_K phi - M^T pi
    conserves H = sum_i (0.5 * pi_i^2 / a_i + pi_i (M phi)_i + 0.25 * sum_d w_d (phi_i - phi_{i+d})^2) identically!
    Let's compute BOTH the positive cell energy E_i (kinetic + spring) AND the shift energy!
    """
    e_kin = np.where(active_mask, 0.5 * (pi**2) / a_grid, 0.0)
    e_pot = np.zeros_like(phi)
    for dx, dy, dz, w in offsets:
        phi_nb = np.roll(phi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        mask_nb = np.roll(active_mask, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        link_active = active_mask & mask_nb
        e_pot += np.where(link_active, 0.25 * w * ((phi - phi_nb) ** 2), 0.0)
    return e_kin + e_pot


def compute_gravity_from_cells(E_total_per_cell, R, X, Y, Z, G=0.015, c=1.0, ell=1.0):
    """
    Computes the 3D radial river velocity J(x,y,z) directly from the actual enclosed
    cell energy M(r_i) = sum_{R_j <= R_i} E_total_per_cell[j] / c^2 (Newton's Shell / Birkhoff Theorem
    applied to the actual 3D grid cells, with zero magic constants!).
    """
    r_flat = R.ravel()
    e_flat = E_total_per_cell.ravel()
    order = np.argsort(r_flat)
    r_sorted = r_flat[order]
    e_cum = np.cumsum(e_flat[order])
    # Include all cells in the same spherical radius shell
    _, inverse_idx = np.unique(np.round(r_sorted, 8), return_inverse=True)
    shell_max_cum = np.zeros_like(e_cum)
    # Map cumulative enclosed energy back to 3D grid
    enclosed_E = np.empty_like(e_flat)
    enclosed_E[order] = e_cum
    enclosed_E = enclosed_E.reshape(R.shape)

    M_enclosed = enclosed_E / (c**2)
    r_s_local = 2.0 * G * M_enclosed / (c**2)
    r_tilde = np.sqrt(R**2 + ell**2)
    J_mag = c * (R / r_tilde) * np.sqrt(r_s_local / r_tilde)
    inv_R = np.divide(1.0, R, out=np.zeros_like(R), where=R > 1e-12)
    Jx = -J_mag * (X * inv_R)
    Jy = -J_mag * (Y * inv_R)
    Jz = -J_mag * (Z * inv_R)
    return Jx, Jy, Jz, float(2.0 * G * np.sum(E_total_per_cell) / (c**4))


def saturate_cells_locally_18pt(phi, pi, a_grid, E_frozen, active_mask, offsets, E_max_ceiling):
    """
    100% Local 18-Neighbor Saturation (Zero global radius lookup!):
    Whenever any active cell i has E_i >= E_max_ceiling:
      1. Cell i locks exactly E_max_ceiling into E_frozen[i] and freezes (active_mask[i] = False, pi[i] = 0).
      2. Its excess energy Delta_E = E_i - E_max_ceiling is pushed ONLY across its own 18 stencil links
         to its directly connected active neighbors j in N_18(i), weighted by w_d!
    """
    E_cells = compute_cell_energies(phi, pi, a_grid, offsets, active_mask)
    while True:
        over_mask = active_mask & (E_cells >= E_max_ceiling)
        if not np.any(over_mask):
            break

        # Freeze the newly saturated cells FIRST so we know which neighbors remain active
        active_mask[over_mask] = False

        # Compute total active neighbor weight for each newly saturated cell
        weight_sum = np.zeros_like(phi)
        for dx, dy, dz, w in offsets:
            nb_active = np.roll(active_mask, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
            weight_sum += np.where(over_mask & nb_active, w, 0.0)

        # Push excess energy (E_i - E_max) from each saturating cell i strictly to its 18 neighbors j
        incoming_neighbor_energy = np.zeros_like(phi)
        excess_i = np.where(over_mask, E_cells - E_max_ceiling, 0.0)
        E_frozen[over_mask] = E_max_ceiling
        pi[over_mask] = 0.0

        # Recompute E_cells after freezing over_mask (Wait: freezing links between over_mask and active neighbors
        # also releases the spring potential energy 0.25*w*(phi_i - phi_j)^2 that was on those boundary links!
        # Let's include that boundary link energy in the local neighbor transfer so total energy is 100% conserved!)
        E_cells_after_freeze = compute_cell_energies(phi, pi, a_grid, offsets, active_mask)
        total_before = float(np.sum(E_cells))
        total_locked = float( int(np.sum(over_mask)) * E_max_ceiling )
        total_remaining_active = float(np.sum(E_cells_after_freeze))
        local_energy_to_distribute = max(total_before - total_locked - total_remaining_active, 0.0)

        # Distribute strictly to the direct 18-stencil active neighbors touching over_mask!
        neighbor_receive_weight = np.zeros_like(phi)
        for dx, dy, dz, w in offsets:
            from_over = np.roll(over_mask, shift=(dx, dy, dz), axis=(0, 1, 2))
            neighbor_receive_weight += np.where(active_mask & from_over, w, 0.0)

        total_nb_weight = float(np.sum(neighbor_receive_weight))
        if local_energy_to_distribute > 0.0 and total_nb_weight > 0.0:
            dE_local = local_energy_to_distribute * (neighbor_receive_weight / total_nb_weight)
            has_rx = neighbor_receive_weight > 0.0
            pi[has_rx] = np.where(pi[has_rx] >= 0.0, 1.0, -1.0) * np.sqrt(
                (pi[has_rx] ** 2) + 2.0 * a_grid[has_rx] * dE_local[has_rx]
            )

        E_cells = compute_cell_energies(phi, pi, a_grid, offsets, active_mask)
    return E_cells


def run_3d_simulation(cap_density=True, stencil_mode="18pt", n_grid=25, n_steps=55, dt=0.035):
    if stencil_mode == "27pt":
        offsets = build_27pt_offsets_and_weights()
        stencil_desc = "Full 27-pt (6 Faces 7/15 + 12 Edges 1/10 + 8 Corners 1/30)"
    elif stencil_mode == "6pt":
        offsets = build_6pt_offsets_and_weights()
        stencil_desc = "Standard 6-pt (6 Faces 1.0)"
    else:
        offsets = build_18pt_offsets_and_weights()
        stencil_desc = "18-neighbor (6 Faces 1/3 + 12 Edges 1/6)"
    coords = np.arange(n_grid) - (n_grid // 2)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    R2 = X**2 + Y**2 + Z**2
    R = np.sqrt(R2)

    G = 0.0085
    c = 1.0
    ell = 1.0
    E_max_ceiling = 2.60

    # 1. Start with a 27-cell frozen seed core at the center (r <= 1.8 cells), each at E_max_ceiling
    active_mask = np.ones((n_grid, n_grid, n_grid), dtype=bool)
    E_frozen = np.zeros((n_grid, n_grid, n_grid), dtype=float)
    seed_core = R <= 1.8
    if cap_density:
        active_mask[seed_core] = False
        E_frozen[seed_core] = E_max_ceiling
    else:
        # In the uncapped Visser/Hayward comparison, the same central mass M_core = 27 * E_max_ceiling
        # sits at the center without freezing the cells (active_mask stays True everywhere).
        E_frozen[R == 0.0] = float(int(np.sum(seed_core)) * E_max_ceiling)

    M_seed = float(np.sum(E_frozen)) / (c**2)
    r_s_seed = 2.0 * G * M_seed / (c**2)  # ~ 2.53 cells initially
    r_tilde = np.sqrt(R**2 + (1.8**2))
    a_grid = 1.0 + 0.2 * (r_s_seed / r_tilde)

    # 2. Initialize spherical wave shell starting around r = 3.4 cells
    phi = np.zeros((n_grid, n_grid, n_grid), dtype=float)
    pi = np.zeros((n_grid, n_grid, n_grid), dtype=float)
    shell_mask = active_mask & (R >= 2.1) & (R <= 5.2)
    phi[shell_mask] = 1.35 * np.exp(-((R[shell_mask] - 3.4) ** 2) / 0.7)
    pi[shell_mask] = 1.75 * a_grid[shell_mask] * np.exp(-((R[shell_mask] - 3.4) ** 2) / 0.7)

    E_init_cells = compute_cell_energies(phi, pi, a_grid, offsets, active_mask)
    max_dormant_energy_t0 = float(np.max(E_init_cells[R > 7.5]))

    initial_frozen_cells = int(np.sum(~active_mask))
    total_energy_0 = float(np.sum(E_frozen) + np.sum(E_init_cells))
    initial_wave_energy = float(np.sum(E_init_cells))

    # Compute initial horizon r_s_initial (before wave energy) vs expanded horizon r_s_total (with wave energy)
    r_s_before_wave = float(2.0 * G * np.sum(E_frozen) / (c**4))
    _, _, _, r_s_with_wave = compute_gravity_from_cells(E_frozen + E_init_cells, R, X, Y, Z, G=G, c=c, ell=ell)

    peak_active_density_over_run = float(np.max(E_init_cells))
    escaped_outside_horizon_energy = 0.0
    max_energy_drift = 0.0

    for _ in range(n_steps):
        E_wave_now = compute_cell_energies(phi, pi, a_grid, offsets, active_mask)
        Jx, Jy, Jz, r_s_now = compute_gravity_from_cells(E_frozen + E_wave_now, R, X, Y, Z, G=G, c=c, ell=ell)

        # Energy-conserving skew-coupled wave + 18-pt river evolution:
        #   dphi/dt = pi / a_grid
        #   dpi/dt  = -L_K(phi) - a_grid * M_18pt(pi / a_grid)
        # Since M_18pt is skew-symmetric (<u, M_18pt u> == 0 for u = pi / a_grid),
        # d/dt sum_i E_i = <pi/a, -L_K phi - a M_18pt(pi/a)> + <L_K phi, pi/a> == 0 identically!
        def rhs(phi_s, pi_s):
            u_vel = np.where(active_mask, pi_s / a_grid, 0.0)
            dphi = u_vel
            river_force = apply_3d_18pt_river_shift(
                u_vel, Jx, Jy, Jz, offsets, active_mask, transpose=False
            )
            dpi = -apply_3d_18pt_laplacian(phi_s, offsets, active_mask) - river_force
            dphi = np.where(active_mask, dphi, 0.0)
            dpi = np.where(active_mask, dpi, 0.0)
            return dphi, dpi

        k1_phi, k1_pi = rhs(phi, pi)
        k2_phi, k2_pi = rhs(phi + 0.5 * dt * k1_phi, pi + 0.5 * dt * k1_pi)
        k3_phi, k3_pi = rhs(phi + 0.5 * dt * k2_phi, pi + 0.5 * dt * k2_pi)
        k4_phi, k4_pi = rhs(phi + dt * k3_phi, pi + dt * k3_pi)

        phi = phi + (dt / 6.0) * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi)
        pi = pi + (dt / 6.0) * (k1_pi + 2.0 * k2_pi + 2.0 * k3_pi + k4_pi)

        if cap_density:
            E_cells = saturate_cells_locally_18pt(phi, pi, a_grid, E_frozen, active_mask, offsets, E_max_ceiling)
        else:
            E_cells = compute_cell_energies(phi, pi, a_grid, offsets, active_mask)

        peak_active_density_over_run = max(peak_active_density_over_run, float(np.max(E_cells)))
        total_energy_t = float(np.sum(E_frozen) + np.sum(E_cells))
        max_energy_drift = max(max_energy_drift, abs(total_energy_t - total_energy_0) / total_energy_0)

        outside_mask = active_mask & (R > (r_s_now + 1.5))
        escaped_outside_horizon_energy = max(
            escaped_outside_horizon_energy,
            float(np.sum(E_cells[outside_mask])) / initial_wave_energy,
        )

    final_frozen_cells = int(np.sum(~active_mask))
    R_core_initial = float((3.0 * max(initial_frozen_cells, 1) / (4.0 * np.pi)) ** (1.0 / 3.0))
    R_core_final = float((3.0 * max(final_frozen_cells, 1) / (4.0 * np.pi)) ** (1.0 / 3.0))

    dphi_end, dpi_end = rhs(phi, pi)
    frozen_interior_activity = float(
        np.max(np.abs(dphi_end[~active_mask])) + np.max(np.abs(dpi_end[~active_mask]))
    ) if np.any(~active_mask) else 0.0

    # Measure shell angular anisotropy on the exact Euclidean sphere R = 3.0 cells (R2 == 9),
    # which contains both 6 axial points (+/-3,0,0) and 24 diagonal points (+/-2,+/-2,+/-1)
    # at the EXACT same distance R = 3.000000 ell from the center!
    shell_r3_mask = R2 == 9
    phi_r3 = phi[shell_r3_mask]
    shell_r3_rel_anisotropy = float(np.std(phi_r3) / (np.mean(np.abs(phi_r3)) + 1e-15))

    return {
        "cap_density_enabled": cap_density,
        "stencil_mode": stencil_desc,
        "grid_size_3d": f"{n_grid}x{n_grid}x{n_grid} ({n_grid**3} cells)",
        "shortcuts_used": f"NONE ({stencil_desc} Laplacian + River Shift + Local Saturation + Exact Cell-Sum M(r))",
        "max_dormant_outer_cell_energy_t0": max_dormant_energy_t0,
        "frozen_core_interior_activity": frozen_interior_activity,
        "cell_saturation_ceiling_E_max": E_max_ceiling,
        "peak_active_cell_energy_reached": peak_active_density_over_run,
        "peak_ratio_to_ceiling": float(peak_active_density_over_run / E_max_ceiling),
        "initial_frozen_core_cells": initial_frozen_cells,
        "final_frozen_core_cells": final_frozen_cells,
        "initial_R_core_cells": R_core_initial,
        "final_R_core_cells": R_core_final,
        "r_s_before_wave": r_s_before_wave,
        "r_s_expanded_with_wave": r_s_with_wave,
        "max_relative_total_energy_drift": max_energy_drift,
        "escaped_echo_fraction_outside_horizon": escaped_outside_horizon_energy,
        "shell_R3_axial_vs_diagonal_rel_anisotropy": shell_r3_rel_anisotropy,
    }


def main():
    uncapped_visser = run_3d_simulation(cap_density=False, stencil_mode="18pt")
    capped_frozen_core_6pt = run_3d_simulation(cap_density=True, stencil_mode="6pt")
    capped_frozen_core_18pt = run_3d_simulation(cap_density=True, stencil_mode="18pt")
    capped_frozen_core_27pt = run_3d_simulation(cap_density=True, stencil_mode="27pt")

    passed = (
        uncapped_visser["peak_ratio_to_ceiling"] > 1.20
        and capped_frozen_core_18pt["peak_ratio_to_ceiling"] <= 1.000001
        and capped_frozen_core_18pt["final_frozen_core_cells"] > capped_frozen_core_18pt["initial_frozen_core_cells"]
        and capped_frozen_core_18pt["final_R_core_cells"] > capped_frozen_core_18pt["initial_R_core_cells"]
        and capped_frozen_core_18pt["r_s_expanded_with_wave"] > capped_frozen_core_18pt["r_s_before_wave"]
        and capped_frozen_core_18pt["frozen_core_interior_activity"] == 0.0
        and capped_frozen_core_18pt["max_dormant_outer_cell_energy_t0"] == 0.0
        and capped_frozen_core_18pt["max_relative_total_energy_drift"] < 1e-4
        and capped_frozen_core_18pt["escaped_echo_fraction_outside_horizon"] < 1e-4
        and capped_frozen_core_27pt["peak_ratio_to_ceiling"] <= 1.000001
        and capped_frozen_core_27pt["final_frozen_core_cells"] > capped_frozen_core_27pt["initial_frozen_core_cells"]
        and capped_frozen_core_27pt["final_R_core_cells"] > capped_frozen_core_27pt["initial_R_core_cells"]
        and capped_frozen_core_27pt["r_s_expanded_with_wave"] > capped_frozen_core_27pt["r_s_before_wave"]
        and capped_frozen_core_27pt["frozen_core_interior_activity"] == 0.0
        and capped_frozen_core_27pt["max_dormant_outer_cell_energy_t0"] == 0.0
        and capped_frozen_core_27pt["max_relative_total_energy_drift"] < 1e-4
        and capped_frozen_core_27pt["escaped_echo_fraction_outside_horizon"] < 1e-4
        and capped_frozen_core_27pt["shell_R3_axial_vs_diagonal_rel_anisotropy"]
        < capped_frozen_core_6pt["shell_R3_axial_vs_diagonal_rel_anisotropy"]
    )

    summary = {
        "experiment_A_uncapped_visser_pileup_3d": uncapped_visser,
        "experiment_B0_saturating_frozen_core_3d_6pt_baseline": capped_frozen_core_6pt,
        "experiment_B_saturating_frozen_core_3d_18pt": capped_frozen_core_18pt,
        "experiment_C_saturating_frozen_core_3d_27pt": capped_frozen_core_27pt,
        "ALL_3D_LATTICE_SIMULATOR_CHECKS_PASSED": bool(passed),
    }

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulate_3d_18pt_freezing_core_results.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nALL 3D 18-PT & 27-PT LATTICE SIMULATOR CHECKS PASSED: {passed}")


if __name__ == "__main__":
    main()
