"""
Module 5: Non-Spherical 3D Flux & Spinning Kerr-Doran Ergosphere Tests (Benchmarks 9 & 10)
==========================================================================================
Tests:
  Test 14: 3D Asymmetric Discrete Gauss-Poynting Flux Identity Test (Benchmark 9)
  Test 15: 3D Spinning Kerr-Doran Whirlpool & Lense-Thirring Frame-Dragging Test (Benchmark 10)
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verify_core_finite_bar_gr import (
    test_9_discrete_gauss_poynting_flux_identity_3d_non_spherical,
    test_10_3d_spinning_kerr_doran_whirlpool_and_river_ergosphere,
)


def test_14_3d_asymmetric_discrete_gauss_poynting_flux():
    """
    Test 14:
    - Evaluates the discrete Gauss-Poynting conservation law on the 3D 27-point lattice
      for an off-center, non-spherical wave packet across a control surface R <= 3.2*ell:
        d/dt sum_{i in Omega} E_i^local = - sum_{i in Omega, j not in Omega} F_{ij}
    - Confirms link antisymmetry F_{ij} + F_{ji} = 0 (< 1e-15) and exact surface-flux
      balance (< 1e-13) without spherical symmetry assumptions.
    """
    return test_9_discrete_gauss_poynting_flux_identity_3d_non_spherical()


def test_15_3d_spinning_kerr_doran_whirlpool_and_frame_dragging():
    """
    Test 15:
    - Evaluates the 3D 27-point skew-symmetric advection matrix M_{ij} for a spinning
      regular Kerr-Doran river + whirlpool velocity field J = J_r r_hat + J_phi phi_hat.
    - Confirms an equatorial supersonic ergosphere (|J| > c while |J_r| < c), regular
      origin speed |J(0)| = 0, exact skew-symmetry M^T = -M (< 1e-15), Cayley energy
      conservation (< 1e-13), and Lense-Thirring azimuthal frame-dragging (<y> != 0).
    """
    return test_10_3d_spinning_kerr_doran_whirlpool_and_river_ergosphere()


def run_module_5():
    res = {
        "test_14_3d_asymmetric_discrete_gauss_poynting_flux": test_14_3d_asymmetric_discrete_gauss_poynting_flux(),
        "test_15_3d_spinning_kerr_doran_whirlpool_and_frame_dragging": test_15_3d_spinning_kerr_doran_whirlpool_and_frame_dragging(),
    }
    res["module_5_passed"] = all(v["passed"] for v in res.values())
    return res


if __name__ == "__main__":
    out = run_module_5()
    print(json.dumps(out, indent=2))
    print(f"\nMODULE 5 PASSED: {out['module_5_passed']}")
