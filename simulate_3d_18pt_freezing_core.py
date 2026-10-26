"""
simulate_3d_18pt_freezing_core.py
=================================
3D lattice simulation of a scalar wave packet in the Painleve-Gullstrand / Hayward
river background with local saturation damping on a 25 x 25 x 25 grid:

  1. True Painleve-Gullstrand Hamiltonian Dynamics (Black-Hole Infall Sign):
       Pi = a * (dPhi/dt + J . grad(Phi))
       H_Killing = sum_i [ 0.5 * Pi_i^2 / a_i - Pi_i * (D_J Phi)_i
                           + 0.25 * sum_d w_d (Phi_i - Phi_{i+d})^2 ]
       dPhi/dt = -(D_J Phi) + Pi / a
       dPi/dt  = -L_K(Phi)  - (D_J Pi)
     where D_J^T = -D_J is the skew-symmetric 3D isotropic advection operator.
     Characteristics along the radial river flow J(r) < 0 are v_char = J(r) +/- c,
     so inside the horizon (|J| > c) both characteristics point strictly inward.

  2. Implicit Cayley (Crank-Nicolson) Symplectic Midpoint Propagation:
       z^{n+1} = z^n + (dt / 2) * F(z^n + z^{n+1})
     solved via 12 fixed-point iterations per step (converged to ~1e-15 for dt = 0.03).

  3. Hayward River Velocity Profile & Matched Initial Conditions:
       J(r) = -c * sqrt(r_s(r) * r^2 / (r^3 + r_s(r) * ell^2))
     Both the uncapped run (cap_density=False) and the saturated run (cap_density=True)
     use the EXACT same initial 27-cell seed core mass distribution, and are also
     compared against a flat-space (J = 0) control run to isolate gravitational
     infall compression and blueshift from pure spherical geometry focusing.

  4. Well-Posed Local Saturation Damping (No Ill-Posed Internal Reflecting Wall):
     The hyperbolic wave grid remains continuous everywhere (no internal Neumann cut).
     Whenever a voxel's local free-fall co-moving energy density
       E_i^comoving = 0.5 * Pi_i^2 / a_i + 0.25 * sum_d w_d (Phi_i - Phi_{i+d})^2
     exceeds the voxel ceiling E_voxel_max, its local kinetic momentum Pi_i is damped
     to the ceiling and the absorbed co-moving energy Delta_E_i is transferred into
     the static core reservoir E_core[i].

  5. Honest Energy Bookkeeping & Limitations (added after independent review):
     - On a FIXED background, H_Killing is conserved by the Cayley step to ~1e-15.
     - When J is recomputed every step from the enclosed cell energy (fixed_background=False), H_Killing
       changes at each update. This update is an ad-hoc self-gravity rule, NOT Einstein's equations.
     - The saturation damping removes Killing energy from the field and adds COMOVING energy to E_core.
     - The budget H(t) = H(0) + (background-update change) - (damping removal) is tracked and must close
       to round-off; the two non-conservative terms are reported separately instead of being hidden.
     - The deposited reservoir sits at fixed cells inside r_+; that is only consistent if it is read as
       vacuum-like matter (p_r = -rho c^2), whose stress tensor has no preferred velocity in the t-r plane.
     - The grid is periodic (np.roll), r_s ~ 3 cells, r_- ~ 1 cell (not resolved), and a run lasts only
       55 * 0.03 = 1.65 time units. Treat the 3D results as a qualitative demonstration only.
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


def apply_3d_laplacian(phi, offsets):
    """Applies the symmetric 3D stencil Laplacian L_K(phi)."""
    lap = np.zeros_like(phi)
    for dx, dy, dz, w in offsets:
        phi_nb = np.roll(phi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        lap += w * (phi - phi_nb)
    return lap


def apply_3d_skew_advection(field, Jx, Jy, Jz, offsets):
    """
    Applies the skew-symmetric 3D river advection operator D_J(field) ~ (J . grad)(field):
      (D_J f)_i = (1 / 4*ell) * sum_d w_d * 0.5 * ((J_i + J_{i+d}) . d) * f_{i+d}
    Satisfies D_J^T = -D_J identically on the periodic grid.
    """
    out = np.zeros_like(field)
    for dx, dy, dz, w in offsets:
        f_nb = np.roll(field, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jx_nb = np.roll(Jx, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jy_nb = np.roll(Jy, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        Jz_nb = np.roll(Jz, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        J_dot_d = 0.5 * ((Jx + Jx_nb) * dx + (Jy + Jy_nb) * dy + (Jz + Jz_nb) * dz)
        out += 0.25 * w * J_dot_d * f_nb
    return out


def compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets):
    """
    Returns:
      1. E_comoving_grid: local free-fall co-moving wave energy in each cell i:
           E_i^comoving = 0.5 * Pi_i^2 + 0.25 * sum_d w_d (Phi_i - Phi_{i+d})^2
      2. H_killing_total: global Killing Hamiltonian conserved under fixed-J evolution:
           H_Killing = sum_i [ E_i^comoving - Pi_i * (D_J Phi)_i ]
    """
    e_kin = 0.5 * (pi**2)
    e_pot = np.zeros_like(phi)
    for dx, dy, dz, w in offsets:
        phi_nb = np.roll(phi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        e_pot += 0.25 * w * ((phi - phi_nb) ** 2)
    e_comoving = e_kin + e_pot
    dj_phi = apply_3d_skew_advection(phi, Jx, Jy, Jz, offsets)
    h_killing = float(np.sum(e_comoving - pi * dj_phi))
    return e_comoving, h_killing


def compute_hayward_gravity_from_cells(E_total_per_cell, R, X, Y, Z, G=0.025, c=1.0, ell=1.0):
    """
    Computes the 3D Hayward radial river velocity:
      J(x,y,z) = -c * sqrt( r_s(r) * r^2 / (r^3 + r_s(r) * ell^2) ) * r_hat
    from the midpoint spherical shell sum M(r_i) = sum_{R_j < R_i} E_j + 0.5 * sum_{R_j == R_i} E_j.
    """
    r_flat = R.ravel()
    e_flat = E_total_per_cell.ravel()
    order = np.argsort(r_flat)
    r_sorted = r_flat[order]
    e_sorted = e_flat[order]
    e_cum = np.cumsum(e_sorted)

    unique_r, inverse_idx = np.unique(np.round(r_sorted, 8), return_inverse=True)
    shell_max_cum = np.zeros(len(unique_r), dtype=float)
    shell_sum = np.zeros(len(unique_r), dtype=float)
    np.maximum.at(shell_max_cum, inverse_idx, e_cum)
    np.add.at(shell_sum, inverse_idx, e_sorted)
    shell_enclosed = shell_max_cum - 0.5 * shell_sum

    enclosed_E = np.empty_like(e_flat)
    enclosed_E[order] = shell_enclosed[inverse_idx]
    enclosed_E = enclosed_E.reshape(R.shape)

    M_enclosed = enclosed_E / (c**2)
    r_s_local = 2.0 * G * M_enclosed / (c**2)
    denom = (R**3) + r_s_local * (ell**2)
    J_mag = c * np.sqrt(np.divide(r_s_local * (R**2), denom, out=np.zeros_like(R), where=denom > 1e-15))
    inv_R = np.divide(1.0, R, out=np.zeros_like(R), where=R > 1e-12)
    Jx = -J_mag * (X * inv_R)
    Jy = -J_mag * (Y * inv_R)
    Jz = -J_mag * (Z * inv_R)
    return Jx, Jy, Jz, float(2.0 * G * np.sum(E_total_per_cell) / (c**4))


def cayley_midpoint_step_3d(phi, pi, Jx, Jy, Jz, offsets, dt, n_iter=12):
    """
    Implicit Cayley (Crank-Nicolson) symplectic midpoint step for the true PG system:
      dPhi/dt = -D_J(Phi) + Pi
      dPi/dt  = -L_K(Phi) - D_J(Pi)
    Solves z^{n+1} = z^n + dt * F(0.5 * (z^n + z^{n+1})) to machine precision (~1e-15).
    """
    phi_next = phi.copy()
    pi_next = pi.copy()
    for _ in range(n_iter):
        phi_mid = 0.5 * (phi + phi_next)
        pi_mid = 0.5 * (pi + pi_next)
        dphi = -apply_3d_skew_advection(phi_mid, Jx, Jy, Jz, offsets) + pi_mid
        dpi = -apply_3d_laplacian(phi_mid, offsets) - apply_3d_skew_advection(pi_mid, Jx, Jy, Jz, offsets)
        phi_next = phi + dt * dphi
        pi_next = pi + dt * dpi
    return phi_next, pi_next


def apply_local_saturation_damping(phi, pi, E_core, saturated_visit_mask, offsets, E_voxel_max):
    """
    Well-posed local saturation damping (inelastic absorption into core mass reservoir):
    Whenever a cell's local free-fall co-moving energy E_i^comoving = E_kin,i + E_pot,i
    exceeds E_voxel_max, its excess energy Delta_E_i = E_i^comoving - E_voxel_max is
    locally damped out of the wave momentum Pi_i (and local amplitude if E_pot,i > E_voxel_max)
    and added directly to the cell's core mass reservoir E_core[i].
    """
    w_sum = sum(w for _, _, _, w in offsets)
    for _ in range(8):
        e_kin = 0.5 * (pi**2)
        e_pot = np.zeros_like(phi)
        phi_bar = np.zeros_like(phi)
        for dx, dy, dz, w in offsets:
            phi_nb = np.roll(phi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
            e_pot += 0.25 * w * ((phi - phi_nb) ** 2)
            phi_bar += (w / w_sum) * phi_nb
        e_comoving = e_kin + e_pot

        over = e_comoving > E_voxel_max
        if not np.any(over):
            break
        saturated_visit_mask |= over
        excess = np.where(over, e_comoving - E_voxel_max, 0.0)
        E_core += excess
        # Reduce kinetic energy 0.5 * pi^2 first to bring E_i^comoving down to E_voxel_max
        target_kin = np.maximum(E_voxel_max - e_pot, 0.0)
        scale_pi = np.divide(
            np.sqrt(2.0 * target_kin),
            np.abs(pi),
            out=np.zeros_like(pi),
            where=np.abs(pi) > 1e-15,
        )
        pi = np.where(over, pi * np.minimum(scale_pi, 1.0), pi)
        # If gradient energy e_pot itself exceeds E_voxel_max, relax phi toward neighbor average
        pot_over = e_pot > E_voxel_max
        if np.any(pot_over):
            scale_pot = np.sqrt(np.minimum(E_voxel_max / np.maximum(e_pot, 1e-15), 1.0))
            phi = np.where(pot_over, phi_bar + 0.85 * scale_pot * (phi - phi_bar), phi)

    e_kin = 0.5 * (pi**2)
    e_pot = np.zeros_like(phi)
    for dx, dy, dz, w in offsets:
        phi_nb = np.roll(phi, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        e_pot += 0.25 * w * ((phi - phi_nb) ** 2)
    e_comoving = e_kin + e_pot
    return phi, pi, e_comoving


def run_3d_simulation(
    cap_density=True,
    stencil_mode="18pt",
    n_grid=25,
    n_steps=55,
    dt=0.03,
    gravity_enabled=True,
    fixed_background=False,
    record_history=False,
):
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

    G = 0.025 if gravity_enabled else 0.0
    c = 1.0
    ell = 1.0
    # Voxel saturation ceiling in simulation units where E_P = 20.0 -> 3/(8*pi)*E_P ~ 2.387
    E_P_sim = 20.0
    E_voxel_max = (3.0 / (8.0 * np.pi)) * E_P_sim

    # Identical 27-cell seed core mass distribution at the center (R <= 1.8) for ALL runs
    seed_core = R <= 1.8
    E_core = np.where(seed_core, E_voxel_max, 0.0)
    E_core_initial_total = float(np.sum(E_core))
    saturated_visit_mask = seed_core.copy() if cap_density else np.zeros_like(seed_core)
    initial_saturated_cells = int(np.sum(saturated_visit_mask))

    # Spherical wave shell around r = 4.5 cells (starts below E_voxel_max at t = 0)
    phi = np.zeros((n_grid, n_grid, n_grid), dtype=float)
    pi = np.zeros((n_grid, n_grid, n_grid), dtype=float)
    r0 = 4.5
    sigma2 = 0.85
    shell_mask = (R >= 2.2) & (R <= 6.2)
    env = np.exp(-((R[shell_mask] - r0) ** 2) / sigma2)
    phi[shell_mask] = 1.50 * env
    pi[shell_mask] = 1.91 * env

    Jx0, Jy0, Jz0, r_s_before_wave = compute_hayward_gravity_from_cells(E_core, R, X, Y, Z, G=G, c=c, ell=ell)
    E_init_comoving, H_killing_0 = compute_comoving_and_killing_energies(phi, pi, Jx0, Jy0, Jz0, offsets)
    initial_wave_comoving = float(np.sum(E_init_comoving))
    max_dormant_energy_t0 = float(np.max(E_init_comoving[R > 8.0]))
    _, _, _, r_s_with_wave = compute_hayward_gravity_from_cells(
        E_core + E_init_comoving, R, X, Y, Z, G=G, c=c, ell=ell
    )

    # Killing-energy budget: H(t) = H(0) + bg_change - removed_by_damping  (+ round-off)
    bg_change = 0.0
    removed_by_damping = 0.0
    max_budget_residual = 0.0
    peak_active_density_over_run = float(np.max(E_init_comoving))
    shell_r3_rel_anisotropy = 0.0
    shell_r3_mask = R2 == 9

    history = {"t": [0.0], "H_killing": [float(H_killing_0)], "bg_change": [0.0], "removed_by_damping": [0.0],
               "reservoir_gain": [0.0], "wave_comoving": [initial_wave_comoving]}
    Jx, Jy, Jz = Jx0, Jy0, Jz0
    for step_idx in range(n_steps):
        if not fixed_background:
            E_com_now, H_old_J = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)
            Jx, Jy, Jz, _ = compute_hayward_gravity_from_cells(E_core + E_com_now, R, X, Y, Z, G=G, c=c, ell=ell)
            _, H_new_J = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)
            bg_change += H_new_J - H_old_J

        phi, pi = cayley_midpoint_step_3d(phi, pi, Jx, Jy, Jz, offsets, dt)

        if cap_density:
            _, H_k_before_damp = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)
            phi, pi, E_com_after = apply_local_saturation_damping(
                phi, pi, E_core, saturated_visit_mask, offsets, E_voxel_max
            )
            _, H_k_after_damp = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)
            removed_by_damping += H_k_before_damp - H_k_after_damp
        else:
            E_com_after, _ = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)

        peak_active_density_over_run = max(peak_active_density_over_run, float(np.max(E_com_after)))
        _, H_k_now = compute_comoving_and_killing_energies(phi, pi, Jx, Jy, Jz, offsets)
        budget_residual = abs(H_k_now - (H_killing_0 + bg_change - removed_by_damping)) / (abs(H_killing_0) + 1e-15)
        max_budget_residual = max(max_budget_residual, budget_residual)
        if record_history:
            history["t"].append((step_idx + 1) * dt)
            history["H_killing"].append(float(H_k_now))
            history["bg_change"].append(float(bg_change))
            history["removed_by_damping"].append(float(removed_by_damping))
            history["reservoir_gain"].append(float(np.sum(E_core)) - E_core_initial_total)
            history["wave_comoving"].append(float(np.sum(E_com_after)))

        if step_idx == 20:
            phi_r3 = phi[shell_r3_mask]
            shell_r3_rel_anisotropy = float(np.std(phi_r3) / (np.mean(np.abs(phi_r3)) + 1e-15))

    final_saturated_cells = int(np.sum(saturated_visit_mask))
    R_core_initial = float((3.0 * max(initial_saturated_cells, 1) / (4.0 * np.pi)) ** (1.0 / 3.0))
    R_core_final = float((3.0 * max(final_saturated_cells, 1) / (4.0 * np.pi)) ** (1.0 / 3.0))
    reservoir_gain = float(np.sum(E_core)) - E_core_initial_total

    out = {
        "cap_density_enabled": cap_density,
        "gravity_enabled": gravity_enabled,
        "fixed_background": fixed_background,
        "stencil_mode": stencil_desc,
        "grid_size_3d": f"{n_grid}x{n_grid}x{n_grid} ({n_grid**3} cells, periodic)",
        "simulated_time": float(n_steps * dt),
        "integrator": "Implicit Cayley (Crank-Nicolson) midpoint, 12 fixed-point iterations",
        "river_profile": "Hayward J(r) = -c * sqrt(r_s r^2 / (r^3 + r_s ell^2))",
        "max_dormant_outer_cell_energy_t0": max_dormant_energy_t0,
        "cell_saturation_ceiling_E_max": float(E_voxel_max),
        "peak_active_cell_energy_reached": peak_active_density_over_run,
        "peak_ratio_to_ceiling": float(peak_active_density_over_run / E_voxel_max),
        "initial_frozen_core_cells": initial_saturated_cells,
        "final_frozen_core_cells": final_saturated_cells,
        "initial_R_core_cells": R_core_initial,
        "final_R_core_cells": R_core_final,
        "r_s_before_wave": r_s_before_wave,
        "r_s_expanded_with_wave": r_s_with_wave,
        "initial_killing_energy": float(H_killing_0),
        "killing_change_from_background_updates_rel": float(bg_change / abs(H_killing_0)),
        "killing_removed_by_damping_rel": float(removed_by_damping / abs(H_killing_0)),
        "killing_budget_closure_residual": float(max_budget_residual),
        "reservoir_gain_over_initial_wave_comoving_energy": float(reservoir_gain / initial_wave_comoving),
        "shell_R3_axial_vs_diagonal_rel_anisotropy": shell_r3_rel_anisotropy,
    }
    if record_history:
        out["history"] = history
    return out


def main():
    flat_space_control = run_3d_simulation(cap_density=False, stencil_mode="18pt", gravity_enabled=False)
    uncapped_fixed_bg = run_3d_simulation(cap_density=False, stencil_mode="18pt", fixed_background=True)
    capped_fixed_bg = run_3d_simulation(cap_density=True, stencil_mode="18pt", fixed_background=True)
    uncapped_dynamic_bg = run_3d_simulation(cap_density=False, stencil_mode="18pt")
    capped_6pt = run_3d_simulation(cap_density=True, stencil_mode="6pt")
    capped_18pt = run_3d_simulation(cap_density=True, stencil_mode="18pt")
    capped_27pt = run_3d_simulation(cap_density=True, stencil_mode="27pt")
    all_runs = [flat_space_control, uncapped_fixed_bg, capped_fixed_bg, uncapped_dynamic_bg,
                capped_6pt, capped_18pt, capped_27pt]

    # Numerical-correctness checks only
    numerical_checks_passed = (
        uncapped_fixed_bg["killing_budget_closure_residual"] < 1e-13
        and flat_space_control["killing_budget_closure_residual"] < 1e-13
        and all(run["killing_budget_closure_residual"] < 1e-10 for run in all_runs)
        and all(run["peak_ratio_to_ceiling"] <= 1.000001 for run in (capped_6pt, capped_18pt, capped_27pt, capped_fixed_bg))
        and capped_18pt["max_dormant_outer_cell_energy_t0"] == 0.0
        and capped_18pt["shell_R3_axial_vs_diagonal_rel_anisotropy"]
        < 0.25 * capped_6pt["shell_R3_axial_vs_diagonal_rel_anisotropy"]
    )

    findings = {
        "peak_over_cap_flat_space_J0": flat_space_control["peak_ratio_to_ceiling"],
        "peak_over_cap_fixed_hayward_background": uncapped_fixed_bg["peak_ratio_to_ceiling"],
        "peak_over_cap_dynamic_self_gravity_update": uncapped_dynamic_bg["peak_ratio_to_ceiling"],
        "pileup_requires_ad_hoc_self_gravity_update": bool(
            uncapped_fixed_bg["peak_ratio_to_ceiling"] <= 1.0 < uncapped_dynamic_bg["peak_ratio_to_ceiling"]
        ),
        "killing_change_from_background_updates_rel_18pt": capped_18pt["killing_change_from_background_updates_rel"],
        "cells_hitting_cap_fixed_background": capped_fixed_bg["final_frozen_core_cells"] - capped_fixed_bg["initial_frozen_core_cells"],
        "cells_hitting_cap_dynamic_background_18pt": capped_18pt["final_frozen_core_cells"] - capped_18pt["initial_frozen_core_cells"],
        "faceting_6pt_18pt_27pt": [
            capped_6pt["shell_R3_axial_vs_diagonal_rel_anisotropy"],
            capped_18pt["shell_R3_axial_vs_diagonal_rel_anisotropy"],
            capped_27pt["shell_R3_axial_vs_diagonal_rel_anisotropy"],
        ],
    }

    summary = {
        "experiment_0_flat_space_J0_control": flat_space_control,
        "experiment_A0_fixed_hayward_background_uncapped": uncapped_fixed_bg,
        "experiment_A1_fixed_hayward_background_capped": capped_fixed_bg,
        "experiment_A2_dynamic_self_gravity_uncapped": uncapped_dynamic_bg,
        "experiment_B0_dynamic_capped_6pt": capped_6pt,
        "experiment_B_dynamic_capped_18pt": capped_18pt,
        "experiment_C_dynamic_capped_27pt": capped_27pt,
        "findings": findings,
        "NUMERICAL_CHECKS_PASSED": bool(numerical_checks_passed),
    }

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulate_3d_18pt_freezing_core_results.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\n3D LATTICE SIMULATOR NUMERICAL CHECKS PASSED: {numerical_checks_passed}")
    print("NOTE: these checks test the numerics (budget closure, cap respected, stencil isotropy).")
    print("      They do not show that saturation resolves mass inflation; see tests/module_4 Test 13.")


if __name__ == "__main__":
    main()
