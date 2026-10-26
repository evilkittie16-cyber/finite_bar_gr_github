"""
Module 6: Linear Quantum Evolution & Hayward Remnant Thermodynamics Tests (Sections 4.5 & 5.2)
(The linear lattice evolution is unitary; the saturation damping step is NOT. The remnant thermodynamics
 is that of Hayward's 2006 metric and requires the inner horizon to exist.)
========================================================================================
Tests:
  Test 16: Heisenberg Commutator, Ehrenfest Trajectory, & Horizon Entanglement Test (Benchmark 6)
  Test 17: Hawking Surface Gravity, Specific Heat Flip, & First-Law Remnant Test (Benchmark 8)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verify_core_finite_bar_gr import (
    test_6_quantum_route_preserves_gr,
    test_8_surface_gravity_hawking_thermodynamics_and_zero_temp_remnant,
)


def test_16_heisenberg_unitarity_ehrenfest_and_entanglement():
    """
    Test 16:
    - Verifies canonical quantum commutator preservation S(t) Omega S(t)^T = Omega
      across 200 horizon-crossing steps (drift < 1e-13),
    - Ehrenfest agreement is exact by construction for linear dynamics and is NOT tested,
    - Evaluates the classical 1PN index coefficient for two masses (no superposition is formed;
      agreement with 2.0 is by construction), and finite non-zero Gaussian ground-state entanglement
      entropy S_ent > 0 across the horizon cut.
    """
    return test_6_quantum_route_preserves_gr()


def test_17_hawking_surface_gravity_specific_heat_and_remnant():
    """
    Test 17:
    - Verifies the exact analytical surface gravity identity
        kappa(r_+) = (c^2 / (2 r_+)) * (1 - 3*ell^2 / r_+^2)
      against finite differences of f(r),
    - Confirms peak Hawking temperature at r_+ = 3.0*ell (r_s = 27/8 * ell = 3.375*ell)
      where d(kappa)/dr_+ = 0, positive specific heat C_M > 0 for sqrt(3)*ell < r_+ < 3*ell,
      zero-temperature extremal remnant kappa(sqrt(3)*ell) = 0, and First Law residual
      |c^2 dM - T_H dS_reg| < 1e-14.
    """
    return test_8_surface_gravity_hawking_thermodynamics_and_zero_temp_remnant()


def run_module_6():
    res = {
        "test_16_heisenberg_unitarity_ehrenfest_and_entanglement": test_16_heisenberg_unitarity_ehrenfest_and_entanglement(),
        "test_17_hawking_surface_gravity_specific_heat_and_remnant": test_17_hawking_surface_gravity_specific_heat_and_remnant(),
    }
    res["module_6_passed"] = all(v["passed"] for v in res.values())
    return res


if __name__ == "__main__":
    out = run_module_6()
    print(json.dumps(out, indent=2))
    print(f"\nMODULE 6 PASSED: {out['module_6_passed']}")
