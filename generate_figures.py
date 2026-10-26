"""
generate_figures.py
===================
Generates the 6-panel summary figure (figures/finite_bar_gr_summary.png). All panels are computed from the
current code; nothing is typed in by hand.

  Panel (a): Hayward river speed |J(r)|/c with both horizons r_- and r_+ and the core scale R_core
  Panel (b): Hayward core stresses: SEC violated, NEC satisfied
  Panel (c): 3D shell faceting on R = 3 ell for the 6-pt, 18-pt and 27-pt stencils (from the simulator)
  Panel (d): Inner-horizon blueshift vs lattice spacing, and the reservoir deposit a saturation cap creates
  Panel (e): Hayward surface gravity / Hawking temperature (requires the inner horizon)
  Panel (f): 3D Killing-energy budget of a capped run: what is conserved and what is not
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate_3d_18pt_freezing_core import run_3d_simulation
from pg_wave_1d import run_inner_horizon_packet, hayward_horizons


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "figure.facecolor": "white"})
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 13.5), dpi=150)
    ell = 1.0

    # (a) Hayward river profile and horizons
    ax = axes[0, 0]
    r = np.linspace(0.0, 12.0, 600)
    r_s_bh = 8.0
    R_core_bh = (r_s_bh * ell**2) ** (1.0 / 3.0)
    J_bh = np.sqrt((r_s_bh * r**2) / (r**3 + r_s_bh * ell**2))
    r_s_crit = 1.5 * np.sqrt(3.0) * ell
    J_crit = np.sqrt((r_s_crit * r**2) / (r**3 + r_s_crit * ell**2))
    r_m, r_p, _, _ = hayward_horizons(r_s_bh, ell)
    ax.plot(r, np.sqrt(r_s_bh / np.maximum(r, 0.25)), "--", color="#888888", lw=1.4, label=r"Schwarzschild $\sqrt{r_s/r}$")
    ax.plot(r, J_bh, "-", color="#1f77b4", lw=2.3, label=r"Hayward ($r_s = 8\ell$)")
    ax.plot(r, J_crit, "-.", color="#2ca02c", lw=2.0, label=r"Extremal remnant ($r_s = \frac{3\sqrt{3}}{2}\ell$)")
    ax.axhline(1.0, color="#d62728", ls=":", lw=1.4, label=r"$|J| = c$")
    ax.axvline(r_m, color="#9467bd", lw=1.3, label=rf"inner horizon $r_- = {r_m:.2f}\ell$ (still present)")
    ax.axvline(r_p, color="#d62728", lw=1.3, label=rf"outer horizon $r_+ = {r_p:.2f}\ell$")
    ax.axvline(R_core_bh, color="#1f77b4", ls="--", lw=1.0, label=rf"core scale $R_{{core}} = {R_core_bh:.1f}\ell$")
    ax.axvspan(r_m, r_p, color="#d62728", alpha=0.07, label=r"trapped region ($c + J < 0$)")
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(0.0, 2.2)
    ax.set_xlabel(r"$r/\ell$")
    ax.set_ylabel(r"$|J(r)|/c$")
    ax.set_title("(a) Hayward river profile: both horizons remain")
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.92)
    ax.grid(True, alpha=0.25)

    # (b) stresses
    ax = axes[0, 1]
    x = np.linspace(0.0, 2.5, 500)
    rho = 1.0 / ((x**3 + 1.0) ** 2)
    p_r = -rho
    p_t = (2.0 * x**3 - 1.0) / ((x**3 + 1.0) ** 3)
    ax.plot(x, p_r, "-", color="#1f77b4", lw=2.2, label=r"$p_r/(\rho_{\max}c^2)$")
    ax.plot(x, p_t, "-", color="#ff7f0e", lw=2.2, label=r"$p_\theta/(\rho_{\max}c^2)$")
    ax.plot(x, rho + p_r + 2 * p_t, "--", color="#9467bd", lw=1.8, label=r"SEC: $\rho c^2 + p_r + 2p_\theta$ (violated)")
    ax.plot(x, rho + p_t, ":", color="#2ca02c", lw=2.0, label=r"NEC: $\rho c^2 + p_\theta \geq 0$ (satisfied)")
    ax.axhline(0.0, color="black", lw=0.9, alpha=0.6)
    ax.axvline(1.0, color="#888888", ls="--", lw=1.0)
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(-2.1, 0.6)
    ax.set_xlabel(r"$r / R_{core}$")
    ax.set_ylabel(r"stress $[\rho_{\max} c^2]$")
    ax.set_title("(b) Hayward core: SEC violated, NEC satisfied")
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)
    ax.grid(True, alpha=0.25)

    # (c) faceting from the simulator
    ax = axes[1, 0]
    runs = {m: run_3d_simulation(cap_density=True, stencil_mode=m) for m in ("6pt", "18pt", "27pt")}
    vals = [100.0 * runs[m]["shell_R3_axial_vs_diagonal_rel_anisotropy"] for m in ("6pt", "18pt", "27pt")]
    bars = ax.bar(["6-point", "18-neighbor", "27-point"], vals, color=["#d62728", "#1f77b4", "#2ca02c"],
                  width=0.52, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v * 1.03 + 0.05, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.0, max(vals) * 1.2)
    ax.set_ylabel(r"shell anisotropy $\sigma_\Omega(\Phi)/\langle|\Phi|\rangle$ [%]")
    ax.set_title(r"(c) 3D shell faceting on $R = 3\ell$ (18-pt best; 27-pt has larger 6th-order spread)")
    ax.grid(True, axis="y", alpha=0.25)

    # (d) inner-horizon blueshift vs resolution and reservoir deposit
    ax = axes[1, 1]
    hs = np.array([0.04, 0.02, 0.01])
    unc = [run_inner_horizon_packet(h, T=40.0) for h in hs]
    cap = [run_inner_horizon_packet(h, T=40.0, cap_factor=2.0) for h in hs]
    peaks = [u["max_comoving_energy_near_r_minus_over_E0"] for u in unc]
    res = [c["reservoir_comoving_energy_over_E0"] for c in cap]
    ax.loglog(1.0 / hs, peaks, "o-", color="#d62728", lw=2.0, label=r"uncapped: peak comoving energy near $r_-$ / $E_0$")
    ax.loglog(1.0 / hs, res, "s-", color="#1f77b4", lw=2.0, label=r"capped: energy dumped into reservoir / $E_0$")
    ax.loglog(1.0 / hs, peaks[0] * (0.04 / hs), ":", color="#d62728", lw=1.0, label=r"$\propto 1/h$")
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label=r"initial packet energy $E_0$")
    ax.set_xlabel(r"resolution $\ell/h$")
    ax.set_ylabel(r"energy / $E_0$")
    ax.set_title("(d) Inner-horizon blueshift: the cap moves it into the reservoir")
    ax.legend(loc="upper left", fontsize=7.8, framealpha=0.92)
    ax.grid(True, which="both", alpha=0.25)

    # (e) Hawking temperature
    ax = axes[2, 0]
    r_pl = np.linspace(np.sqrt(3.0), 12.0, 500)
    kappa_reg = (1.0 / (2.0 * r_pl)) * (1.0 - 3.0 / (r_pl**2))
    r_cl = np.linspace(0.8, 12.0, 500)
    ax.plot(r_cl, 1.0 / (2.0 * r_cl), "--", color="#888888", lw=1.5, label=r"Schwarzschild $c^2/2r_+$")
    ax.plot(r_pl, kappa_reg, "-", color="#d62728", lw=2.3, label=r"Hayward $\frac{c^2}{2r_+}(1 - 3\ell^2/r_+^2)$")
    ax.axvline(3.0, color="#1f77b4", ls=":", lw=1.4, label=r"peak at $r_+ = 3\ell$")
    ax.axvline(np.sqrt(3.0), color="#2ca02c", ls="-.", lw=1.5, label=r"$T_H = 0$ at $r_+ = r_- = \sqrt{3}\ell$")
    ax.set_xlim(0.8, 12.0)
    ax.set_ylim(0.0, 0.25)
    ax.set_xlabel(r"$r_+/\ell$")
    ax.set_ylabel(r"$\kappa(r_+)/(c^2/\ell) \propto T_H$")
    ax.set_title("(e) Hayward temperature: the cold remnant needs the inner horizon")
    ax.legend(loc="upper right", fontsize=7.9, framealpha=0.92)
    ax.grid(True, alpha=0.25)

    # (f) 3D energy budget
    ax = axes[2, 1]
    run = run_3d_simulation(cap_density=True, stencil_mode="18pt", record_history=True)
    hist = run["history"]
    t = np.array(hist["t"])
    H0 = hist["H_killing"][0]
    ax.plot(t, np.array(hist["H_killing"]) / H0, "-", color="black", lw=2.2, label=r"field Killing energy $H(t)/H(0)$")
    ax.plot(t, 1.0 + np.array(hist["bg_change"]) / H0, "--", color="#ff7f0e", lw=1.8,
            label=r"$1 +$ change from ad-hoc self-gravity updates")
    ax.plot(t, 1.0 - np.array(hist["removed_by_damping"]) / H0, "-.", color="#1f77b4", lw=1.8,
            label=r"$1 -$ removed by saturation damping")
    ax.plot(t, (H0 + np.array(hist["bg_change"]) - np.array(hist["removed_by_damping"])) / H0, ":", color="#2ca02c",
            lw=2.4, label=rf"budget $H(0) + \Delta_{{bg}} - \Delta_{{damp}}$ (closes to {run['killing_budget_closure_residual']:.0e})")
    ax.set_xlabel(r"$t/\Delta t_{\min}$")
    ax.set_ylabel("relative Killing energy")
    ax.set_title("(f) 3D capped run: energy budget (not conserved; terms shown)")
    ax.legend(loc="lower left", fontsize=7.8, framealpha=0.92)
    ax.grid(True, alpha=0.25)

    plt.tight_layout(pad=1.7)
    out_file = os.path.join(out_dir, "finite_bar_gr_summary.png")
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved 6-panel summary figure to: {out_file}")


if __name__ == "__main__":
    main()
