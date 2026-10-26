"""
pg_wave_1d.py
=============
Clean 1D (planar) Painleve-Gullstrand test-field solver on a Hayward river profile, used to study
what happens to a wave packet at the inner (Cauchy) horizon r_- and what a local saturation cap does.

Model (test field on a FIXED background, no backreaction):
    dPhi/dt = Pi - (D_J Phi),     dPi/dt = -(L Phi) - (D_J Pi)
with D_J ~ J d/dr skew-symmetric and L ~ -c^2 d^2/dr^2, so the characteristic speeds are J(r) +/- c.
J(r) = -c sqrt(r_s r^2 / (r^3 + r_s ell^2)) (Hayward).

Numerical design choices (all chosen to remove artifacts found in earlier versions):
  * The outer boundary sits far OUTSIDE the outer horizon r_+, inside an absorbing sponge.
  * J is tapered smoothly to zero inside the sponge, so the boundary closure is a plain Neumann wave
    boundary. (With J != 0 at a truncated boundary, the skew-symmetric closure admits boundary-localized
    growing modes whose growth rate scales like 1/h; that numerical artifact contaminated the earlier
    Test 13 setup whose outer edge was inside the trapped region.)
  * Time stepping: implicit Cayley / Crank-Nicolson midpoint (sparse LU), which conserves the quadratic
    Killing energy exactly in the absence of the sponge and the cap.

Diagnostics:
  * comoving (free-fall) energy E = 1/2 Pi^2 + 1/2 c^2 (dPhi/dr)^2 near r_- and in total,
  * with a cap: the comoving energy removed from the field and deposited into a static reservoir
    (this reservoir is what the 3D simulator adds to the enclosed Misner-Sharp mass).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def hayward_horizons(r_s, ell=1.0):
    roots = np.sort(np.real(np.roots([1.0, -r_s, 0.0, r_s * ell**2])))
    r_minus, r_plus = float(roots[1]), float(roots[2])
    kappa_minus = abs(0.5 / r_minus * (1.0 - 3.0 * ell**2 / r_minus**2))
    kappa_plus = abs(0.5 / r_plus * (1.0 - 3.0 * ell**2 / r_plus**2))
    return r_minus, r_plus, kappa_minus, kappa_plus


def build_operators(h, r_s=12.0, ell=1.0, c=1.0, R_out=30.0, taper_center=23.0, taper_width=1.5):
    r = np.arange(0.5 * h, R_out, h)
    n = r.size
    J = -c * np.sqrt(r_s * r**2 / (r**3 + r_s * ell**2))
    J = J * 0.5 * (1.0 - np.tanh((r - taper_center) / taper_width))
    J_mid = 0.5 * (J[:-1] + J[1:])
    D = sp.diags([J_mid / (2.0 * h), -J_mid / (2.0 * h)], [1, -1], shape=(n, n), format="csr")
    main = np.full(n, 2.0)
    main[0] = 1.0
    main[-1] = 1.0
    L = sp.diags([main, -np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1], shape=(n, n), format="csr") * (c**2 / h**2)
    I = sp.identity(n, format="csr")
    F = sp.bmat([[-D, I], [-L, -D]], format="csc")
    return r, J, D, L, F


def comoving_density(phi, pi, h, c=1.0):
    grad = np.zeros_like(phi)
    grad[1:-1] = (phi[2:] - phi[:-2]) / (2.0 * h)
    return 0.5 * pi**2 + 0.5 * (c**2) * grad**2


def killing_energy(phi, pi, D, L):
    return float(0.5 * pi @ pi - pi @ (D @ phi) + 0.5 * phi @ (L @ phi))


def run_inner_horizon_packet(h, r_s=12.0, ell=1.0, c=1.0, T=40.0, cap_factor=None, use_sponge=True,
                             r0=4.0, width=0.6, sponge_start=22.0, sponge_scale=8.0):
    """Evolve a packet launched between r_- and r_+ moving outward relative to the river.

    cap_factor: if given, the comoving energy density is capped at cap_factor * (initial peak density);
                the removed comoving energy is accumulated in a static reservoir.
    Returns a dict of diagnostics.
    """
    r, J, D, L, F = build_operators(h, r_s=r_s, ell=ell, c=c)
    n = r.size
    dt = 0.5 * h
    lu = spla.splu(sp.identity(2 * n, format="csc") - 0.5 * dt * F)
    B = sp.identity(2 * n, format="csc") + 0.5 * dt * F
    if use_sponge:
        sponge = np.exp(-dt * 2.0 * np.clip((r - sponge_start) / sponge_scale, 0.0, None) ** 2)
    else:
        sponge = np.ones(n)

    phi = np.exp(-(((r - r0) / width) ** 2))
    pi = -c * np.gradient(phi, h)
    e_init = comoving_density(phi, pi, h, c)
    E0 = float(np.sum(e_init) * h)
    peak0 = float(e_init.max())
    cap = None if cap_factor is None else cap_factor * peak0
    H0 = killing_energy(phi, pi, D, L)

    r_minus, r_plus, kappa_minus, _ = hayward_horizons(r_s, ell)
    near = np.abs(r - r_minus) < 0.5
    reservoir = 0.0
    reservoir_profile = np.zeros(n)
    max_near = float(np.sum(e_init[near]) * h)
    max_total = E0
    max_density = peak0
    max_killing_drift = 0.0
    t = 0.0
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        z = lu.solve(B @ np.concatenate([phi, pi]))
        phi, pi = z[:n] * sponge, z[n:] * sponge
        if cap is not None:
            e = comoving_density(phi, pi, h, c)
            over = e > cap
            if np.any(over):
                e_pot = e - 0.5 * pi**2
                new_pi = np.where(over, np.sign(pi) * np.sqrt(np.maximum(2.0 * (cap - e_pot), 0.0)), pi)
                removed = 0.5 * pi**2 - 0.5 * new_pi**2
                reservoir += float(np.sum(removed) * h)
                reservoir_profile += removed * h
                pi = new_pi
        t += dt
        e = comoving_density(phi, pi, h, c)
        max_near = max(max_near, float(np.sum(e[near]) * h))
        max_total = max(max_total, float(np.sum(e) * h))
        max_density = max(max_density, float(e.max()))
        if not use_sponge and cap is None:
            max_killing_drift = max(max_killing_drift, abs(killing_energy(phi, pi, D, L) - H0) / abs(H0))

    dep_near = float(np.sum(reservoir_profile[np.abs(r - r_minus) < 1.0]))
    return {
        "h": float(h),
        "r_minus": r_minus,
        "r_plus": r_plus,
        "kappa_minus": kappa_minus,
        "initial_comoving_energy_E0": E0,
        "max_comoving_energy_near_r_minus_over_E0": max_near / E0,
        "max_total_comoving_energy_over_E0": max_total / E0,
        "max_density_over_initial_peak": max_density / peak0,
        "reservoir_comoving_energy_over_E0": reservoir / E0,
        "reservoir_fraction_deposited_within_1ell_of_r_minus": (dep_near / reservoir) if reservoir > 0 else 0.0,
        "max_relative_killing_drift_no_sponge_no_cap": max_killing_drift,
        "all_finite": bool(np.all(np.isfinite(phi)) and np.all(np.isfinite(pi))),
    }


def boundary_mode_growth_rate(h, r_s=12.0, ell=1.0, c=1.0, T=60.0, taper=True, seed=0):
    """Power iteration on the Cayley map: asymptotic growth rate (per unit time) of the fastest mode.
    With taper=False the flow J is not tapered at the outer boundary (reproduces the artifact)."""
    if taper:
        r, J, D, L, F = build_operators(h, r_s=r_s, ell=ell, c=c)
    else:
        r, J, D, L, F = build_operators(h, r_s=r_s, ell=ell, c=c, taper_center=1e9)
    n = r.size
    dt = 0.5 * h
    lu = spla.splu(sp.identity(2 * n, format="csc") - 0.5 * dt * F)
    B = sp.identity(2 * n, format="csc") + 0.5 * dt * F
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(2 * n)
    z /= np.linalg.norm(z)
    logs = []
    for _ in range(int(T / dt)):
        z = lu.solve(B @ z)
        nz = np.linalg.norm(z)
        z /= nz
        logs.append(np.log(nz))
    rate = float(np.mean(logs[-int(10.0 / dt):]) / dt)
    mode = np.abs(z[:n])
    mode /= mode.max()
    support = r[mode > 0.2]
    return rate, float(support.min()), float(support.max())
