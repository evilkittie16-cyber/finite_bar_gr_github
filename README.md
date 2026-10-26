# Finite-Bar Discrete General Relativity: Lattice Waves on a Planck-Density (Hayward) Black Hole

[![Verify Finite-Bar Discrete GR Benchmarks](https://github.com/evilkittie16-cyber/finite_bar_gr_github/actions/workflows/verify.yml/badge.svg)](https://github.com/evilkittie16-cyber/finite_bar_gr_github/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A manuscript and reproducible code for a **scalar test field on a Planck-spaced cubic lattice**, propagating on a fixed Painlevé–Gullstrand background whose density is capped at $\rho_{\max} = 3c^2/(8\pi G\ell^2)$, i.e. Hayward's (2006) regular black hole with core scale $R_{\text{core}} = (r_s\ell^2)^{1/3}$.

> [!IMPORTANT]
> **Status (after independent review).** The code is now numerically sound: the black-hole sign is correct, the lattice evolution conserves energy and traps waves inside the horizon, and every energy flow in the 3D runs is accounted for. Several headline claims of earlier versions are **not supported** by the code and have been withdrawn: the inner horizon is *not* excised, reflected waves are *not* pinned on a static core surface, local saturation does *not* by itself resolve mass inflation, there is no spin-2 sector, and the model only partly determines what happens in the black-hole interior (a cap on cell energy fails; a cap on curvature gives a bounce). The master runner prints a claim-by-claim **claims audit**.

![Summary figure](figures/finite_bar_gr_summary.png)

---

## What the model is (and is not)

- **Is:** a scalar field on a prescribed spherically symmetric metric, discretised with isotropic 18-/27-point stencils and a skew-symmetric (summation-by-parts) Killing Hamiltonian, integrated with the symplectic Cayley map; plus a phenomenological rule that caps the free-fall energy per voxel and moves any excess into a static "core reservoir".
- **Also:** a spherically symmetric collapse code in which **the grid carries the geometry** (`collapse_falling_lattice.py`). Its nodes are free-falling shells, cell proper volumes evolve by Einstein's equations, and the river velocity is determined by the enclosed mass. This is where the saturation rule is tested inside the gravitational dynamics.
- **Is not:** a general 3D discretisation of Einstein's equations. In the wave sector the geometry is prescribed (the 3D code's "self-gravity" update is not GR), there is no gravitational-wave (spin-2) sector, and the rotating flow is not the Kerr metric.

## Main results

| Result | Where |
| :--- | :--- |
| GR recovered outside the horizon (1PN bending coefficient 2, deflection $4GM/c^2b$, photon sphere, $\ell=2$ ringdown) — by construction, since the metric is an input | Module 1, Module 7 Test 20 |
| Isotropic stencils: tree-level GRB 090510 delay $1.85\times10^{-19}$ s from the spatial lattice (about $5.6\times10^{-19}$ s with Planck time steps); at sixth order the 18-point stencil is *more* isotropic than the 27-point one (directional ratios $1:\tfrac32:\tfrac{11}{9}$ vs $1:\tfrac32:\tfrac53$); 3D shell faceting 3.78% / 0.345% / 0.545% for 6 / 18 / 27 points | Module 3 |
| Energy-conserving, symplectic black-hole evolution (Killing drift ~$10^{-14}$); packets inside $r_+$ move inward | Module 4 Test 10 |
| The Hayward core violates the SEC but satisfies the NEC; Penrose's theorem is evaded through the inner Cauchy horizon, which remains in the geometry | Module 2, Module 4 Test 9 |
| Inside $r_+$ every outgoing signal moves inward and fixed-$r$ surfaces are spacelike, so no static "frozen surface" can exist there | Module 4 Test 12 |
| Inner-horizon blueshift of a test packet grows like $1/h$ (peak 6.1 → 12.4 → 25.5 $E_0$ for $h = 0.04, 0.02, 0.01$) | Module 4 Test 13 |
| A local saturation cap bounds the field but converts the excess into reservoir mass of 4.8 → 6.9 → 7.9 $E_0$ (growing with resolution), 95–98% of it within $\ell$ of $r_-$: it relabels rather than removes the blueshifted energy | Module 4 Test 13 |
| In 3D, the pile-up above the cap appears only with the ad-hoc self-gravity update (peak/cap: 0.956 fixed background vs 1.647 with updates); the energy budget closes to $10^{-15}$ | `simulate_3d_18pt_freezing_core.py`, Module 4 Test 11 |
| A saturated core can hold at most $\pi r_s/\ell_P$ nats, $1.8\times10^{38}$ times less than $S_{\text{BH}}$ for one solar mass | Module 8 Test 25 |
| **Dynamical collapse on the falling lattice:** GR reproduced exactly (error $3\times10^{-6}$); $\rho_{\max}$ is reached only inside a trapped region (a region at $\rho_{\max}$ has $2m/R = (R/\ell)^2$) | Module 9 Test 26 |
| Causal pressure cannot enforce the cap (stiff fluid passes $10^6\rho_{\max}$); only conversion to vacuum energy ($p=-\rho$) or modified dynamics can | Module 9 Test 27 |
| A cap on each cell's energy cannot remove the singularity: empty cells (zero energy density) still fall to $R \to 0$ with $K \to \infty$ | Module 9 Test 28 |
| A cap on curvature (mean enclosed density) makes every shell, empty ones included, bounce at exactly $R_{\text{core}}(m) = (r_s\ell^2)^{1/3}$; shells then cross and a shock forms | Module 9 Test 29 |

## Claims audit

| Claim | Verdict |
| :--- | :--- |
| Matches GR outside the horizon | Supported (by construction) |
| Energy-conserving, symplectic black-hole wave evolution | Supported |
| Tree-level dispersion passes Fermi-LAT | Supported (loop-level Lorentz violation open) |
| 27-point stencil removes the sixth-order body-diagonal error | Not supported |
| Bounded curvature, $R_{\text{core}} = (r_s\ell^2)^{1/3}$ | Supported (Hayward metric) |
| Inner Cauchy horizon is excised | Not supported |
| Reflected waves pinned on a static core surface | Not supported |
| Local saturation resolves mass inflation | Not supported |
| Singularity theorems evaded by SEC violation | Not supported |
| Cold $T_H = 0$ remnant | Supported (conditional: needs the inner horizon) |
| Quantum unitarity / information preserved | Not supported |
| Spin-2 sector | Not supported |
| No post-merger echoes | Supported (generic to any horizon) |
| A cap on each cell's energy prevents the singularity | Not supported |
| A curvature (mean-density) cap gives a regular bounce at $R_{\text{core}}$ | Supported (with LQC-type dynamics) |
| The lattice rule selects a transition (V1), a bounce (V2) or an inner horizon (V3) | Partly answered: it must act on curvature; repulsion gives V2, vacuum conversion gives V1; none gives V3 |

Next open steps: follow the V2 bounce through shell crossing (needs shock capturing), and decide physically whether the cap acts by repulsion (V2) or phase change (V1). See Sections 4.5–4.6 of the manuscript.

---

## Repository contents

| File | Description |
| :--- | :--- |
| [`proof_corrected.md`](proof_corrected.md) | Manuscript: postulates, Hayward core, stencils, lattice Hamiltonian, saturation rule and its effect, singularity-theorem / entropy constraints, the three possible completions, verification suite, claims audit, limitations |
| [`verify_core_finite_bar_gr.py`](verify_core_finite_bar_gr.py) | Master runner: 10 core benchmarks + 9 modules (29 tests) + claims audit; writes [`verify_core_finite_bar_gr_results.json`](verify_core_finite_bar_gr_results.json) |
| [`tests/`](tests/) | Modules 1–9 (each runnable standalone) |
| [`simulate_3d_18pt_freezing_core.py`](simulate_3d_18pt_freezing_core.py) | 3D $25^3$ Cayley lattice with local saturation damping and a closed Killing-energy budget; writes [`simulate_3d_18pt_freezing_core_results.json`](simulate_3d_18pt_freezing_core_results.json) |
| [`pg_wave_1d.py`](pg_wave_1d.py) | Clean 1D inner-horizon solver (outer boundary outside $r_+$, absorbing sponge, tapered flow) used by Test 13 |
| [`collapse_falling_lattice.py`](collapse_falling_lattice.py) | Spherically symmetric collapse on a falling lattice (grid carries the geometry): GR, local cap, pressure, vacuum conversion, curvature cap (Module 9) |
| [`generate_figures.py`](generate_figures.py) | Regenerates [`figures/finite_bar_gr_summary.png`](figures/finite_bar_gr_summary.png) from the current code |
| [`requirements.txt`](requirements.txt) | `numpy`, `matplotlib`, `scipy` |

## Quickstart

```bash
git clone https://github.com/evilkittie16-cyber/finite_bar_gr_github.git
cd finite_bar_gr_github
pip install -r requirements.txt

python verify_core_finite_bar_gr.py      # all numerical checks + claims audit
python simulate_3d_18pt_freezing_core.py # 3D runs and energy budget
python generate_figures.py               # summary figure
```

Expected final lines of the master runner:

```text
ALL 9 MODULES (29 TESTS) NUMERICAL CHECKS PASSED: True
ALL 10 CORE BENCHMARK NUMERICAL CHECKS PASSED: True

CLAIMS AUDIT (what the code does and does not support):
  [SUPPORTED] Matches GR outside the horizon ...
  ...
  [NOT SUPPORTED] Local saturation resolves inner-horizon mass inflation
  ...
  [PARTLY ANSWERED] The lattice saturation rule selects V1 (transition), V2 (bounce) or V3 (inner horizon)
```

"Numerical checks passed" means the code does what it says (conservation, stability, identities). It is not a statement that the physics claims hold; the claims audit is.

## Modules

| Module | Tests | Checks |
| :--- | :--- | :--- |
| 1 Exterior GR | 1–3 | Light-bending coefficients in two gauges; dormant cells; ray-traced deflection and photon sphere |
| 2 Core | 4–5 | Einstein-tensor stresses, SEC combination, $p_\theta$ profile; $K(0) = 24/\ell^4$ |
| 3 Stencils | 6–8 | Moments and sixth-order directional ratios; 3D shell faceting; GRB 090510 delay |
| 4 Horizon & saturation | 9–13 | Horizon ordering; Cayley conservation and inward trapping; 3D energy budget; trapped region; inner-horizon blueshift and reservoir deposit vs resolution |
| 5 Non-spherical & rotating flow | 14–15 | Discrete Killing-flux identity; conservation on an ad-hoc rotating flow (not Kerr) |
| 6 Quantum & thermodynamics | 16–17 | Linear commutator preservation; Hayward temperature, peak at $r_+ = 3\ell$, $T_H(\sqrt3\ell) = 0$ |
| 7 Strong field | 18–21 | Proper time; tidal terms; junction jumps (spacelike layer); WKB ringdown; zero echo inside $r_+$ |
| 8 Extensions | 22–25 | Off-centre 3D packet; two-component scalar toy (not spin-2); regular SdS and Brillouin zone; analytic V1/V2 comparison and entropy bound |
| 9 Dynamical collapse | 26–29 | Falling-lattice GR collapse vs exact LTB; pressure vs vacuum conversion vs LQC; local cap fails in empty cells; curvature cap bounces every shell at $R_{\text{core}}(m)$ |

## Known limitations
Dynamical geometry only in the spherically symmetric dust collapse code (no pressure, stops at shell crossing); scalar test field in the wave sector; the 3D runs resolve $r_s$ with ~3 cells, do not resolve $r_-$, and last $1.65\,t_P$ on a periodic box; ad-hoc rotating flow; the fundamental-lattice Lorentz-violation loop problem is not addressed; the saturation step is non-unitary; tidal forces near the core are bounded but Planckian.
