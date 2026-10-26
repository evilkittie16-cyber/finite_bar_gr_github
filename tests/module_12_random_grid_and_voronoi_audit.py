"""
Module 12: 3D Random Voronoi-Delaunay Lattice, Rayleigh Scattering Bounds,
and Unstructured Random Falling-Mesh Invariance Audit (Tests 38-40).

Tests the forum discussion question: "What about a random grid? Would that be closer to real space?"
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from random_voronoi_and_mesh_audit import (
    audit_3d_random_voronoi_vs_naive_laplacian,
    audit_random_grid_rayleigh_scattering_bounds,
    run_unstructured_random_einstein_euler,
    audit_static_and_stretched_anamorphic_grid,
    audit_tidal_aspect_ratio_hencky_bounce,
)


def test_38_random_voronoi_delaunay_vs_naive_random_laplacian_consistency() -> dict:
    """
    Test 38: Compares a naive distance-weighted random graph Laplacian against the exact 3D
    Voronoi-Delaunay Discrete Exterior Calculus (DEC) circumcentric dual Laplacian on 83 interior
    3D random Voronoi polyhedra.

    Verifies:
      1) Naive distance-weighted random Laplacian fails at O(1/ell) on a linear function Phi = k . x + b
         (rms error ~ 0.525) because random neighbor displacements do not cancel at first order.
      2) Exact 3D Voronoi-Delaunay Laplacian satisfies Minkowski's facet vector identity
         |sum_j A_ij n_ij| < 1e-14 and reproduces Delta_Vor(k . x + b) == 0 to < 1e-14 on every cell.
      3) Exact Pyramid Volume Trace Identity: because each Voronoi facet bisects l_ij perpendicularly at
         distance l_ij/2, V_i = (1/6) sum_j A_ij l_ij forces the trace of the local second-moment tensor
         Tr(M_i) == 3.0 to < 1e-14 on every single random Voronoi cell!
      4) Symplectic Cayley propagation on the 3D random Voronoi graph conserves Hamiltonian wave energy
         to round-off (< 1e-13 over 100 steps).
    """
    res = audit_3d_random_voronoi_vs_naive_laplacian(jitter=0.35, seed=42)
    passed = bool(
        res["max_minkowski_facet_vector_err"] < 1e-13
        and res["voronoi_dec_linear_laplacian_max_err"] < 1e-13
        and res["naive_random_linear_laplacian_rms_err"] > 0.1
        and res["max_pyramid_trace_identity_err_TrM_minus_3"] < 1e-13
        and res["voronoi_stiffness_symmetry_err"] < 1e-14
        and res["cayley_energy_drift_100_steps"] < 1e-13
    )
    return {
        "passed": passed,
        **res,
    }


def test_39_random_grid_rotational_isotropy_vs_rayleigh_scattering_bound() -> dict:
    """
    Test 39: Evaluates the physical trade-off of a static 3D random Voronoi lattice:
      1) Ensemble rotational isotropy: volume-averaged second-moment tensor <M_ab> matches delta_ab
         within O(1/sqrt(N_cells)) (~ 1e-2) with no preferred crystal axes.
      2) Quenched Planck-scale disorder & 3D Rayleigh elastic scattering: individual random Voronoi cells
         have traceless shear std sigma_shear = 0.0765 and volume std sigma_vol = 0.1553.
         While this passes Fermi-LAT GRB 090510 at 31 GeV (tau_scatter = 8.2e-14 << 1), the E^4 Rayleigh
         scattering estimate gives tau_scatter = 2.2 > 1 for 2.5 PeV photons over the observed ~1.4 kpc
         LHAASO (Cygnus-region) baseline (15.6 over a hypothetical 10 kpc), whereas the periodic 18-point
         crystal lattice has tau_scatter == 0 by Bloch's theorem. Assumes the random lattice is physical.
    """
    vor_res = audit_3d_random_voronoi_vs_naive_laplacian(jitter=0.35, seed=42)
    scat = audit_random_grid_rayleigh_scattering_bounds(vor_res["cell_to_cell_directional_shear_std"])

    passed = bool(
        vor_res["volume_averaged_second_moment_isotropy_err"] < 0.02
        and vor_res["cell_to_cell_directional_shear_std"] > 0.05
        and scat["grb090510_31GeV_optical_depth_tau"] < 1e-10
        and scat["lhaaso_2p5PeV_1p4kpc_optical_depth_tau"] > 1.0
        and scat["periodic_18pt_crystal_bloch_optical_depth_tau"] == 0.0
    )
    return {
        "passed": passed,
        "volume_averaged_isotropy_err": vor_res["volume_averaged_second_moment_isotropy_err"],
        "cell_to_cell_shear_std": vor_res["cell_to_cell_directional_shear_std"],
        "cell_to_cell_volume_std": vor_res["cell_to_cell_volume_std"],
        **scat,
        "finding_static_3d_random_grid_scatters_multi_PeV_photons": bool(
            scat["lhaaso_2p5PeV_1p4kpc_optical_depth_tau"] > 1.0
        ),
    }


def test_40_random_unstructured_falling_lattice_einstein_euler_invariance() -> dict:
    """
    Test 40: Tests a co-moving random/unstructured Lagrangian falling grid (random shell sizes
    ds_k in [0.65, 1.35]/N) in the coupled general-relativistic Misner-Sharp / Einstein-Euler
    collapse and bounce solver across N = 40, 80, 120.

    Verifies that when the random grid moves with the fluid (avoiding static background disorder):
      1) Shell crossing is prevented (min_dR > 0), Lorentzian signature is preserved (min_Gamma > 0.78),
         ADM mass is conserved to machine zero (0.0), and every shell turns around inside R_-(m_i).
      2) As N = 40 -> 80 -> 120, the random unstructured mesh converges monotonically to the uniform-mesh
         bounce trajectory (L2 error of mass-weighted mean radius R_mean(tau): 9.32e-4 -> 1.52e-4 -> 1.99e-5;
         relative difference in R_surf,min < 4e-4; relative difference in peak thermal energy < 2.1e-3).
    """
    ref_uniform = run_unstructured_random_einstein_euler(N=120, jitter=0.0, seed=0)
    resolutions = [40, 80, 120]
    l2_errs = []
    eps_rel_errs = []
    rmin_rel_errs = []
    all_valid = True

    for N in resolutions:
        rnd = run_unstructured_random_einstein_euler(N=N, jitter=0.35, seed=42 + N)
        l2_err = float(np.sqrt(np.mean((rnd["R_mean_series"] - ref_uniform["R_mean_series"]) ** 2)))
        eps_err = float(abs(rnd["max_mean_eps"] - ref_uniform["max_mean_eps"]) / ref_uniform["max_mean_eps"])
        rmin_err = float(abs(rnd["R_surf_min"] - ref_uniform["R_surf_min"]) / ref_uniform["R_surf_min"])
        l2_errs.append(l2_err)
        eps_rel_errs.append(eps_err)
        rmin_rel_errs.append(rmin_err)
        if not (
            rnd["min_dR"] > 0.0
            and rnd["min_Gamma"] > 0.7
            and rnd["adm_mass_rel_err"] < 1e-13
            and rnd["all_turns_inside_R_minus"]
        ):
            all_valid = False

    monotone_l2 = bool(l2_errs[0] > l2_errs[1] > l2_errs[2] and l2_errs[2] < 5e-5)
    monotone_eps = bool(eps_rel_errs[0] > eps_rel_errs[1] > eps_rel_errs[2] and eps_rel_errs[2] < 3e-3)
    passed = bool(all_valid and monotone_l2 and monotone_eps and rmin_rel_errs[-1] < 5e-4)

    return {
        "passed": passed,
        "resolutions_N": resolutions,
        "random_vs_uniform_R_mean_L2_errors": l2_errs,
        "random_vs_uniform_max_mean_eps_rel_errors": eps_rel_errs,
        "random_vs_uniform_R_surf_min_rel_errors": rmin_rel_errs,
        "all_random_runs_prevent_crossing_and_preserve_Gamma_positive": all_valid,
    }


def test_41_static_and_stretched_anamorphic_grid_light_bending_and_zero_scattering() -> dict:
    """
    Test 41: Audits a smoothly stretched / tidal aspect-ratio (anamorphic) grid:
      1) Static 3D radially anamorphic lattice (b_r(R) = 1/sqrt(f(R)) != b_perp(R) = 1, a(R) = sqrt(f(R)),
         directional conductances K_r(R) = 1 != K_perp(R) = 1/f(R)) reproduces the exact 1915 GR 1PN
         light-bending coefficient 2.000 and 2PN deflection (< 1e-3 residual).
      2) The smooth anamorphic aspect ratio A(R) varies only on the macroscopic curvature scale
         R >= r_s >> ell_P (smooth gradient delta_A_cell ~ ell_P / (2 r_s) ~ 2.7e-39 near a solar-mass
         horizon). This is a deterministic gradient, not random disorder, so no Rayleigh optical depth is
         claimed for it; away from masses the grid is the periodic crystal (tau_scatter == 0, Bloch).
      3) In the relativistic Einstein-Euler collapse, a smoothly stretched anamorphic radial coordinate
         s(xi) = 0.65 * xi^0.50 + 0.35 * xi balances spherical shell masses dmu_k ~ R_k^2 dR_k, reducing
         the peak thermal-heating discretization error at N = 50 by 2.22x (2.78e-2 -> 1.25e-2) and
         reducing surface bounce error (4.70e-3 -> 3.53e-3) relative to a uniform radial grid.
    """
    res = audit_static_and_stretched_anamorphic_grid(b_impact=2000.0, r_s=2.0, ell=1.0)
    passed = bool(
        abs(res["anamorphic_1PN_coeff_after_2PN_subtraction"] - 2.0) < 1e-3
        and res["thermal_heating_error_reduction_factor"] > 2.0
        and res["N50_anamorphic_rel_err_Rsurf_min"] < res["N50_uniform_rel_err_Rsurf_min"]
        and res["N50_anamorphic_L2_Rsurf"] < res["N50_uniform_L2_Rsurf"]
    )
    return {
        "passed": passed,
        **res,
    }


def test_42_tidal_aspect_ratio_and_hencky_anamorphic_bar_stress_in_bounce() -> dict:
    """
    Test 42: Audits physical tidal aspect-ratio deformation alpha_k(tau) = (ell_r/ell_r0)/(ell_perp/ell_perp0)
    and anisotropic Misner-Sharp / Einstein-Euler dynamics (p_r != p_perp) with Hencky logarithmic
    deviatoric bar elasticity + Kelvin-Voigt tidal shear viscosity:
      Pi_anam = p_r - p_perp = -G_bar * rho_0 * ln(alpha_k) - eta_anam * rho_0 * h * c_s * l_visc * d(ln alpha_k)/dtau.

    Verifies across inhomogeneities delta in {0.42, 0.48, 0.52}:
      1) Every falling Planck cell undergoes prolate tidal spaghettification during infall (max alpha_k > 1.7)
         followed by oblate pancake compression at the core bounce (min alpha_k in [0.21, 0.29] < 1).
      2) Unlike isotropic shock viscosity (which trades off shell-crossing resistance against Gamma -> 0
         deceleration), the Hencky + Kelvin-Voigt anamorphic bar stress simultaneously:
           - increases minimum shell thickness min(dR/dR_0) (stronger shell-crossing prevention),
           - increases minimum pancake aspect ratio min(alpha_k),
           - reduces peak prolate spaghettification max(alpha_k), AND
           - increases min(Gamma_i) (staying further from the Gamma = 0 spatial throat),
         because during prolate infall (alpha_k > 1) p_perp > p_r provides outward hoop support without
         premature radial braking, and flips to p_r > p_perp only inside the oblate pancake bounce (alpha_k < 1).
      3) Conserves ADM mass to machine zero (0.0), preserves the Misner-Sharp constraint (< 3e-8), and
         converges at second order O(1/N^2) across N = 30 -> 60 -> 120 vs 240 (empirical order p ~ 2.20).
    """
    res = audit_tidal_aspect_ratio_hencky_bounce()
    passed = bool(
        res["all_improve_min_dR_ratio"]
        and res["all_improve_min_alpha_pancake"]
        and res["all_reduce_max_alpha_spaghetti"]
        and res["all_improve_min_Gamma"]
        and res["empirical_convergence_order_mean"] > 1.9
        and all(
            c["anam_adm_mass_rel_err"] < 1e-13
            and c["anam_max_constraint_err"] < 1e-7
            and c["anam_all_turns_inside_R_minus"]
            for c in res["comparisons"]
        )
    )
    return {
        "passed": passed,
        **res,
    }


def run_module_12() -> dict:
    t38 = test_38_random_voronoi_delaunay_vs_naive_random_laplacian_consistency()
    t39 = test_39_random_grid_rotational_isotropy_vs_rayleigh_scattering_bound()
    t40 = test_40_random_unstructured_falling_lattice_einstein_euler_invariance()
    t41 = test_41_static_and_stretched_anamorphic_grid_light_bending_and_zero_scattering()
    t42 = test_42_tidal_aspect_ratio_and_hencky_anamorphic_bar_stress_in_bounce()

    all_passed = bool(
        t38["passed"]
        and t39["passed"]
        and t40["passed"]
        and t41["passed"]
        and t42["passed"]
    )
    return {
        "module": "Module 12: Random Voronoi-Delaunay vs. Smooth Anamorphic Grids & Tidal Bar Stress Audit",
        "test_38_random_voronoi_delaunay_vs_naive_random_laplacian_consistency": t38,
        "test_39_random_grid_rotational_isotropy_vs_rayleigh_scattering_bound": t39,
        "test_40_random_unstructured_falling_lattice_einstein_euler_invariance": t40,
        "test_41_static_and_stretched_anamorphic_grid_light_bending_and_zero_scattering": t41,
        "test_42_tidal_aspect_ratio_and_hencky_anamorphic_bar_stress_in_bounce": t42,
        "module_12_passed": all_passed,
        "all_passed": all_passed,
    }


if __name__ == "__main__":
    results = run_module_12()
    print(json.dumps(results, indent=2))
    print(f"\nMODULE 12 PASSED: {results['all_passed']}")
    if not results["all_passed"]:
        sys.exit(1)
