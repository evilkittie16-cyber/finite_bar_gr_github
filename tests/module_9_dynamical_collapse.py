"""
Module 9: Dynamical Collapse on a Falling Lattice (the grid carries the geometry)
=================================================================================
Tests:
  Test 26: GR collapse on the falling lattice reproduces the exact LTB / Oppenheimer-Snyder solution
  Test 27: Can the cap be enforced by pressure? (causal, NEC-respecting stiff fluid vs vacuum conversion vs LQC)
  Test 28: A LOCAL cell-energy cap cannot prevent the singularity: empty cells reach K -> infinity
  Test 29: A CURVATURE (mean-density) cap makes every shell, empty or not, bounce at R_core(m); then shells cross

All units G = c = ell = 1 (rho_max = 3/(8 pi), R_core(m) = (2m)^(1/3)). See collapse_falling_lattice.py.
Pass flags check numerical correctness against exact solutions; physical conclusions are reported values.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collapse_falling_lattice import run_collapse_study, homogeneous_collapse, r_core, RHO_MAX

_CACHE = {}


def _study(profile):
    if profile not in _CACHE:
        _CACHE[profile] = run_collapse_study(profile=profile)
    return _CACHE[profile]


def test_26_gr_collapse_on_falling_lattice():
    """
    Each lattice node is a free-falling shell; cell proper volumes are (4 pi/3)(R_{k+1}^3 - R_k^3) on the
    exactly flat Painleve-Gullstrand slices, and the river velocity J = -sqrt(2m/R) is determined by the
    enclosed mass. Numerically integrated node trajectories must match the exact marginally bound LTB
    solution R^(3/2) = R0^(3/2) - (3/2) sqrt(2m) tau. Reported: the first cell reaches rho_max only after a
    trapped surface exists, and the density then diverges at the exact singularity time.
    """
    out = {}
    passed = True
    for profile in ("homogeneous", "parabolic"):
        g = _study(profile)["gr"]
        passed &= g["node_ode_vs_exact_LTB_max_rel_error"] < 1e-5 and not g["shell_crossing"]
        out[profile] = {
            "node_vs_exact_LTB_max_rel_error": g["node_ode_vs_exact_LTB_max_rel_error"],
            "time_first_trapped_surface": g["time_first_trapped_surface"],
            "time_first_cell_reaches_rho_max": g["time_first_cell_reaches_rho_max"],
            "cap_reached_inside_trapped_region": bool(g["time_first_cell_reaches_rho_max"] > g["time_first_trapped_surface"]),
            "central_singularity_time_exact": g["central_singularity_time_exact"],
            "max_cell_density_over_rho_max_sampled": g["max_matter_cell_density_over_rho_max"],
        }
    # Lemma: a uniform region at rho_max of radius R has 2m/R = (8 pi/3) rho_max R^2 = (R/ell)^2.
    R = np.array([0.5, 1.0, 2.0, 10.0])
    compact = 2.0 * (4.0 / 3.0) * np.pi * RHO_MAX * R**3 / R
    lemma_err = float(np.max(np.abs(compact - R**2)))
    passed &= lemma_err < 1e-12
    out["lemma_saturated_region_compactness_equals_(R/ell)^2_err"] = lemma_err
    out["passed"] = bool(passed)
    return out


def test_27_how_can_the_cap_be_enforced_homogeneous():
    """
    Homogeneous collapsing region (H < 0) with three ways of enforcing rho <= rho_max:
      stiff_pressure : causal (c_s = c) pressure above 0.5 rho_max. NEC holds. Expected: rho diverges, because
                       d rho/dtau = -3H(rho + p) >= 0 and H^2 = 8 pi rho/3 never vanishes.
      vacuum         : conversion to vacuum energy (p = -rho) at the cap. rho stays at rho_max; a flat region
                       contracts exponentially (contracting de Sitter patch), a closed region bounces at a = ell-scale.
      lqc            : effective repulsion H^2 = (8 pi/3) rho (1 - rho/rho_max): bounce at rho_max.
    Pass = these analytic expectations are reproduced numerically.
    """
    stiff = homogeneous_collapse("stiff_pressure")
    vac_flat = homogeneous_collapse("vacuum")
    vac_closed = homogeneous_collapse("vacuum", k_curv=0.005)
    lqc = homogeneous_collapse("lqc")
    # closed vacuum phase bounces where 1/ell^2 = k/a^2  ->  a = sqrt(k) * ell
    passed = (
        (not stiff["finite"]) and stiff["max_rho_over_rho_max"] > 1e5
        and vac_flat["finite"] and abs(vac_flat["max_rho_over_rho_max"] - 1.0) < 1e-12 and not vac_flat["bounced"]
        and vac_closed["bounced"] and abs(vac_closed["a_min"] - np.sqrt(0.005)) < 1e-3
        and lqc["bounced"] and lqc["max_rho_over_rho_max"] <= 1.0 + 1e-12
    )
    return {
        "passed": bool(passed),
        "stiff_causal_pressure": stiff,
        "vacuum_conversion_flat": vac_flat,
        "vacuum_conversion_closed": vac_closed,
        "lqc_effective_bounce": lqc,
        "finding_pressure_cannot_enforce_cap": bool(not stiff["finite"]),
    }


def test_28_local_cell_energy_cap_cannot_prevent_singularity():
    """
    An empty lattice cell (energy density 0) just outside the collapsing matter keeps its enclosed
    Misner-Sharp mass M (dm/dtau = -4 pi R^2 p U = 0 where p = 0) and follows Schwarzschild free fall,
    whatever the matter inside does (as long as the matter does not overtake it from inside). It reaches
    R -> 0 in finite proper time, where K = 48 M^2/R^6 -> infinity; K exceeds the de Sitter bound 24/ell^4
    for R < 2^(-1/6) R_core(M). A rule acting only on cell energy density never fires there.
    """
    out = {}
    passed = True
    for profile in ("homogeneous", "parabolic"):
        lc = _study(profile)["local_cap"]
        g = _study(profile)["gr"]
        passed &= (not lc["singularity_avoided"]) and abs(lc["empty_node_reaches_it_at_tau"] - g["empty_node_singularity_time_exact"]) < 1e-3
        out[profile] = {
            "empty_node_reaches_R_1e-3_ell_at_tau": lc["empty_node_reaches_it_at_tau"],
            "exact_Schwarzschild_free_fall_time": g["empty_node_singularity_time_exact"],
            "kretschmann_over_deSitter_bound_at_R_1e-3_ell": g["empty_node_kretschmann_over_deSitter_bound_at_R_1e-3_ell"],
            "rule_ever_fires_on_empty_node": lc["rule_ever_fires_on_empty_node"],
        }
    out["passed"] = bool(passed)
    out["finding_local_energy_cap_insufficient"] = True
    return out


def test_29_curvature_cap_bounce_and_shell_crossing():
    """
    Cap on the mean enclosed density rho_bar = 3m/(4 pi R^3) (a curvature invariant in vacuum), implemented
    with the effective-LQC form (dR/dtau)^2 = (2m/R)(1 - rho_bar/rho_max). Checks: numerical node trajectories
    match the exact solution R^3 = R_b^3 + (9/2) m (tau - tau_b)^2; every node, including empty ones, bounces at
    exactly R_core(m) = (2m)^(1/3). Reported: shells cross shortly after the first bounce (decoupled-shell
    description ends; a shock forms), the local cell density can exceed rho_max before crossing for
    non-uniform profiles, and the crossing location relative to the trapped region.
    """
    out = {}
    passed = True
    for profile in ("homogeneous", "parabolic"):
        cc = _study(profile)["curvature_cap_V2"]
        passed &= cc["node_ode_vs_exact_max_rel_error"] < 1e-5 and cc["bounce_radius_equals_R_core_of_enclosed_mass_rel_error"] < 1e-6
        passed &= abs(cc["empty_node_bounce_radius_over_ell"] - cc["R_core_of_M_over_ell"]) / cc["R_core_of_M_over_ell"] < 1e-6
        out[profile] = {k: cc[k] for k in (
            "bounce_radius_equals_R_core_of_enclosed_mass_rel_error", "empty_node_bounce_radius_over_ell",
            "R_core_of_M_over_ell", "max_mean_density_over_rho_max_before_crossing",
            "max_matter_cell_density_over_rho_max_before_crossing", "trapped_region_forms",
            "first_bounce_time", "first_shell_crossing_time", "crossing_radius_over_ell",
            "crossing_radius_over_rs", "crossing_inside_trapped_region")}
    out["passed"] = bool(passed)
    return out


def run_module_9():
    res = {
        "test_26_gr_collapse_on_falling_lattice": test_26_gr_collapse_on_falling_lattice(),
        "test_27_how_can_the_cap_be_enforced_homogeneous": test_27_how_can_the_cap_be_enforced_homogeneous(),
        "test_28_local_cell_energy_cap_cannot_prevent_singularity": test_28_local_cell_energy_cap_cannot_prevent_singularity(),
        "test_29_curvature_cap_bounce_and_shell_crossing": test_29_curvature_cap_bounce_and_shell_crossing(),
    }
    res["module_9_passed"] = all(v["passed"] for v in res.values())
    return res


if __name__ == "__main__":
    out = run_module_9()
    print(json.dumps(out, indent=2))
    print(f"\nMODULE 9 PASSED: {out['module_9_passed']}")
