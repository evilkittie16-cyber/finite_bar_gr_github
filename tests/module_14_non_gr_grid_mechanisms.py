"""
Module 14: Non-GR Grid Mechanisms Audit — How Mass Affects the 18-Point Grid
Without Assuming Einstein's Field Equations Upfront (Tests 48-52).

Tests five distinct physical mechanisms on the discrete lattice:
  - Test 48: Pure 3D 18-Point Hooke's-Law Vector Spring Relaxation (Eshelby 1/r^3 Inclusion vs. 1/r Gravity)
  - Test 49: Crystalline Topological Disclination Theory (Kroner Incompatibility eta_{ij} == G^{(3)}_{ij})
  - Test 50: 3D 18-Point Lattice Poisson + Bernoulli Superfluid Condensate Flow & Bopp-Podolsky UV Audit
  - Test 51: Sakharov One-Loop Quantum Zero-Point Energy (Induced Gravity) on the 18-Point Brillouin Zone
  - Test 52: Exact 3D Simplicial Regge Calculus (Cayley-Menger, Schlafli Identity, Variational Stationarity & Jacobson Area Deficit)
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
)


def test_48_3d_18pt_hooke_eshelby_inclusion_vs_gravity() -> dict:
    """
    Test 48: Pure 3D 18-Point Hooke's-Law Vector Spring Relaxation (Eshelby 1/r^3 vs. 1/r Gravity).
    Verifies on an N=64^3 3-component vector spring truss:
      1. Cauchy-Born O(q^2) elastostatic isotropy when k_1 = k_2 (anisotropy ~ 1.0e-4 vs 1.47 for 6-pt axial),
         with Cauchy speed ratio c_L / c_T = sqrt(3) = 1.73205.
      2. A localized radial inclusion produces u_r(r) ~ 1/r^2 (slope ~ -1.99) and bar strain
         eps_{rr}(r) ~ 1/r^3 (slope ~ -3.23), failing to produce 1/r gravity.
      3. Any single-valued node displacement u_i(x) in flat 3D space has identically zero
         linearised Riemann / Einstein curvature (eta_{ij} = 0 to ~1e-19).
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
    Verifies on an N=48^3 3D lattice:
      1. Kroner's strain incompatibility tensor eta_{ij} = eps_{ikl} eps_{jmn} D_k D_m eps_{ln}
         equals the linearised 3D spatial Einstein tensor G^{(3)}_{ij}[h = 2 eps] to ~1e-16
         and obeys D_i eta_{ij} = 0 to ~1e-17.
      2. Treating mass density rho(x) as topological wedge-disclination density produces the
         true 1/r spatial strain decay (slope = -1.009), while static 3D disclinations alone
         supply only the spatial half (1.0x Newtonian) of light bending.
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
    Verifies on an N=48^3 3D lattice:
      1. The 3D 18-point Poisson Green's function is 25.9x more rotationally isotropic than the
         6-point Laplacian across exact Euclidean shells.
      2. Combining Delta_{18} Phi_N = 4 pi G rho with Bernoulli conservation 1/2 J^2 + Phi_N = 0
         yields the Painleve-Gullstrand Schwarzschild metric and 2.0x Newtonian (1.9972) 1PN light bending.
      3. Fourth-order Bopp-Podolsky lattice regularisation (1 - ell_0^2 Delta_{18}) Delta_{18} Phi = 4 pi G rho
         makes Phi_reg(0) finite, but leaves a K(r) ~ 1/r^2 tidal Kretschmann singularity (slope -2.002)
         unlike the regular de Sitter core (slope +2.000 in tidal g/r^0) of Hayward / effective-LQC.
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
    Verifies:
      1. Exact 3D Brillouin-zone integration gives dimensionless vacuum zero-point density
         E_vac = 0.9887 hbar c / ell^4 (exposing the bare cosmological-constant problem) and
         Sakharov induced coupling c_1 = 0.007009 (ell / ell_P_ind = 0.5936 * sqrt(N_fields)).
      2. Exact diagonalisation of the 3D 18-point wave operator on a unimodular (det g == 1)
         transverse-traceless metric wave h_{xy}(z) = eps cos(k z) shows strictly positive
         quadratic curvature stiffness (dE(2 eps)/dE(eps) = 4.00000, stiffness = +0.1638 > 0).
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
    Test 52: Exact 3D Simplicial Regge Calculus (Cayley-Menger, Schlafli Identity & Variational Stationarity).
    Verifies:
      1. On a 3D Kuhn tetrahedron built from the 18-point axial + face-diagonal bars + body diagonal,
         Cayley-Menger dihedral angles give theta_body = pi/3 (6 * pi/3 = 2*pi flat-space closure to 3.55e-15)
         and satisfy the exact Schlafli identity sum_e l_e (d theta_e / d l_m) = 0 to 3.28e-9.
      2. Numerical finite-difference stationarity dS_Regge/da = 0, dS_Regge/dA = 0 of the discrete
         Regge-Hilbert action functional (with no analytical Einstein equations hardcoded) converges
         at O(1/N^2) (p_A = 1.995, p_a = 1.998, error 3.25e-6 at N=200) to the Schwarzschild bar
         and clock profiles and satisfies Jacobson's entanglement area-deficit law (4.84e-6).
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


def run_module_14() -> dict:
    t48 = test_48_3d_18pt_hooke_eshelby_inclusion_vs_gravity()
    t49 = test_49_kroner_crystalline_disclination_incompatibility_3d()
    t50 = test_50_18pt_poisson_bernoulli_condensate_and_biharmonic_uv()
    t51 = test_51_sakharov_one_loop_zero_point_induced_gravity_18pt()
    t52 = test_52_simplicial_regge_schlafli_and_variational_stationarity()

    all_passed = bool(
        t48["passed"]
        and t49["passed"]
        and t50["passed"]
        and t51["passed"]
        and t52["passed"]
    )
    return {
        "module": "Module 14: Non-GR Grid Mechanisms Audit (Hooke/Eshelby, Kroner Disclinations, Poisson+Bernoulli, Sakharov One-Loop, Regge Variational)",
        "test_48_3d_18pt_hooke_eshelby_inclusion_vs_gravity": t48,
        "test_49_kroner_crystalline_disclination_incompatibility_3d": t49,
        "test_50_18pt_poisson_bernoulli_condensate_and_biharmonic_uv": t50,
        "test_51_sakharov_one_loop_zero_point_induced_gravity_18pt": t51,
        "test_52_simplicial_regge_schlafli_and_variational_stationarity": t52,
        "module_14_passed": all_passed,
    }


if __name__ == "__main__":
    out = run_module_14()
    print(json.dumps(out, indent=2))
