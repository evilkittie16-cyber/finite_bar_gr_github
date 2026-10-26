"""
viscosity_sweep_audit.py
========================
Sensitivity of the Module 11 Lagrangian Misner-Sharp / Einstein-Euler bounce observables to the
fixed physical shock-dissipation length l_visc (referee item C2). Because l_visc is held fixed in
units of ell, N -> infinity converges to the *viscous* PDE; this script measures how the headline
observables (core heating, min Gamma, min shell thickness) depend on l_visc itself.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from einstein_euler_core_bounce import run_lagrangian_einstein_euler


def run_sweep(N: int = 120, l_visc_values=(0.03, 0.06, 0.12)) -> dict:
    rows = []
    for lv in l_visc_values:
        res = run_lagrangian_einstein_euler(N=N, l_visc=lv)
        rows.append({
            "l_visc": lv,
            "max_mean_eps": res["max_mean_eps"],
            "heating_factor_vs_eps_floor": res["max_mean_eps"] / 5e-4,
            "min_Gamma": res["min_Gamma"],
            "min_dR": res["min_dR"],
            "R_surf_min": res["R_surf_min"],
            "all_turns_inside_R_minus": res["all_turns_inside_R_minus"],
        })
    keys = ["heating_factor_vs_eps_floor", "min_Gamma", "min_dR", "R_surf_min"]
    spread = {}
    for k in keys:
        vals = [r[k] for r in rows]
        spread[k + "_rel_spread_over_sweep"] = (max(vals) - min(vals)) / abs(sum(vals) / len(vals))
    return {"N": N, "rows": rows, **spread}


if __name__ == "__main__":
    out = run_sweep()
    print(json.dumps(out, indent=2))
