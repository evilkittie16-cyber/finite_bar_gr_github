# Finite-Bar Discrete General Relativity: A Nonsingular Black Hole Model

[![Verify Finite-Bar Discrete GR Benchmarks](https://github.com/evilkittie16-cyber/finite_bar_gr_github/actions/workflows/verify.yml/badge.svg)](https://github.com/evilkittie16-cyber/finite_bar_gr_github/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Self-contained manuscript, **7-Module / 21-Test independent verification suite** (plus the 10-benchmark master runner), and 3D $25 \times 25 \times 25$ (`15,625-cell`) isotropic lattice simulator for the **Painlevé–Gullstrand / Misner–Sharp reduction and canonical scalar wave sector** of Finite-Bar Discrete General Relativity.

![Finite-Bar Discrete GR Summary](figures/finite_bar_gr_summary.png)

---

## Repository Contents

| File | Description |
| :--- | :--- |
| [`proof_corrected.md`](proof_corrected.md) | Complete mathematical manuscript covering the Planck-scale UV lattice postulates ($\ell = \ell_P, \Delta t_{\min} = t_P, \rho_{\max} = \frac{3c^2}{8\pi G\ell^2}$), 2PN/3PN weak-field isotropic & Painlevé–Gullstrand gauge match, de Sitter / Hayward core unification ($R_{\text{core}}(M) = (r_s\ell^2)^{1/3}, K(0) = \frac{24}{\ell^4}$), 18-neighbor & 27-point isotropic 3D stencils, skew-symmetric summation-by-parts (SBP) river Hamiltonian ($M_{ij} = -M_{ji}$), Cayley symplectic evolution, Neumann link-decoupled Stefan core accretion, discrete Gauss–Poynting flux conservation, Hawking specific-heat reversal ($C_M > 0$ below $r_+ = 3\ell$) & zero-temperature remnant ($T_H(\sqrt{3}\ell) = 0$), and 3D Kerr–Doran spinning whirlpool+river ergosphere propagation. |
| [`tests/`](tests/) | **7 independent verification modules (`21 tests`)** covering every analytical and numerical claim in Sections 1–6 (each runnable standalone or via `verify_core_finite_bar_gr.py`). |
| [`verify_core_finite_bar_gr.py`](verify_core_finite_bar_gr.py) | Master verification runner executing all 10 core benchmarks and all 7 modules (`21 tests`) and exporting [`verify_core_finite_bar_gr_results.json`](verify_core_finite_bar_gr_results.json). |
| [`verify_core_finite_bar_gr_results.json`](verify_core_finite_bar_gr_results.json) | Pre-computed JSON output from `verify_core_finite_bar_gr.py` (all 10 benchmarks + 7 modules / 21 tests). |
| [`simulate_3d_18pt_freezing_core.py`](simulate_3d_18pt_freezing_core.py) | Shortcut-free 3D $25 \times 25 \times 25$ (`15,625-cell`) lattice wave simulator comparing uncapped inner-horizon pile-up against local Stefan cell freezing across 6-point cubic, 18-neighbor isotropic, and 27-point isotropic stencils. |
| [`simulate_3d_18pt_freezing_core_results.json`](simulate_3d_18pt_freezing_core_results.json) | Pre-computed JSON simulation output from `simulate_3d_18pt_freezing_core.py`. |
| [`generate_figures.py`](generate_figures.py) | Reproducible figure generation script that produces the 6-panel summary plot [`figures/finite_bar_gr_summary.png`](figures/finite_bar_gr_summary.png). |
| [`CITATION.cff`](CITATION.cff) & [`LICENSE`](LICENSE) | Standard academic citation metadata and MIT open-source license. |
| [`requirements.txt`](requirements.txt) | Python dependencies (`numpy>=1.20.0`, `matplotlib>=3.5.0`). |

---

## Quickstart & Reproducing All Verification Suites

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/evilkittie16-cyber/finite_bar_gr_github.git
cd finite_bar_gr_github
pip install -r requirements.txt
```

### 2. Run Individual Verification Modules (Modules 1–7, Tests 1–21)
Each theoretical pillar can be verified independently:
```bash
python tests/module_1_exterior_gr.py                    # Module 1 (Tests 1-3): Weak-Field & Exterior GR
python tests/module_2_core_regularization.py            # Module 2 (Tests 4-5): Core Regularization, SEC, & Kretschmann Bound
python tests/module_3_stencil_isotropy.py               # Module 3 (Tests 6-8): 18/27-Pt Isotropy & Fermi-LAT Dispersion
python tests/module_4_horizon_and_freezing.py           # Module 4 (Tests 9-13): Horizon Crossing, Cayley, & Stefan Freezing
python tests/module_5_nonspherical_and_spin.py          # Module 5 (Tests 14-15): 3D Gauss-Poynting Flux & Spinning Kerr-Doran
python tests/module_6_quantum_and_thermodynamics.py      # Module 6 (Tests 16-17): Quantum Unitarity & Remnant Thermodynamics
python tests/module_7_strong_field_and_observables.py    # Module 7 (Tests 18-21): Raindrop Geodesics, Israel Junctions, QNMs, & Echoes
```

### 3. Run the Master Verification Runner (All 10 Benchmarks + All 7 Modules / 21 Tests)
```bash
python verify_core_finite_bar_gr.py
```
Expected console confirmation:
```text
ALL 7 MODULES (21 TESTS) PASSED: True
ALL CORE FINITE-BAR GR TESTS PASSED: True
```

### 4. Run the 3D $25 \times 25 \times 25$ Isotropic Lattice Freezing-Core Simulator
```bash
python simulate_3d_18pt_freezing_core.py
```
Expected console confirmation:
```text
ALL 3D 18-PT & 27-PT LATTICE SIMULATOR CHECKS PASSED: True
```

### 5. Regenerate the 6-Panel Summary Figure
```bash
python generate_figures.py
```

---

## The 7-Module, 21-Test Verification Architecture

### Module 1: Weak-Field & Exterior General Relativity Tests ([`tests/module_1_exterior_gr.py`](tests/module_1_exterior_gr.py))
*Goal: Prove the discrete lattice reproduces standard 1915 General Relativity outside the black hole ($r \gg R_{\text{core}}$).*

| Test | What is Verified in Code | Result |
| :--- | :--- | :--- |
| **Test 1: Dual-Gauge Refractive Index & Light-Bending** | Evaluates $n_{\text{iso}}(r) = \psi^2/w$ and $n_{\text{PG}}(r) = (1 - J^2/c^2)^{-1}$ at $r = 1000\,GM/c^2$, verifying the $\mathbf{2.0000}$ 1PN GR factor and analytical 2PN coefficients ($\frac{7}{4}$ and $4$). | **PASSED** (`conformal err = 3.33e-16`; `1PN after 2PN = 2.000001` iso, `2.000008` PG) |
| **Test 2: Dormant Empty-Space Efficiency** | Initializes a 3D lattice with a localized wave packet and verifies $\dot{\Phi}_i = 0, \dot{\Pi}_i = 0, E_i^{\text{local}} = 0$ in all unreached cells. | **PASSED** (`0.0` exact floating-point activity in dormant region) |
| **Test 3: Geodesic Ray-Tracing & EHT Photon Sphere** | Integrates null rays $(du/d\phi)^2 + f(1/u)u^2 = 1/b^2$ via 200-node Gauss–Legendre quadrature at $b = 500\,r_s$ and checks $r_{\text{ph}} = 1.5\,r_s$. | **PASSED** (`4GM/(c^2 b)` coeff `4.000043` after 2PN, `4.00000016` after 3PN; $r_{\text{ph}}$ `err = 0.0`) |

### Module 2: Core Regularization, SEC Violation, & Hayward Unification ([`tests/module_2_core_regularization.py`](tests/module_2_core_regularization.py))
*Goal: Prove the finite saturation ceiling $\rho_{\max}$ eliminates the $r \to 0$ singularity, bypasses Buchdahl/Penrose, and matches Hayward's smooth profile.*

| Test | What is Verified in Code | Result |
| :--- | :--- | :--- |
| **Test 4: Einstein Tensor & SEC Violation** | Computes $p_r(r)$ and $p_\theta(r)$ from $G^\mu_\nu$, verifying $p_r = -\rho c^2$, SEC violation $\rho c^2 + p_r + 2p_\theta = -2\rho_{\max}c^2 < 0$ at $r \to 0$, $p_\theta(2^{-1/3}R_{\text{core}}) = 0$, and $p_\theta(R_{\text{core}}) = +\frac{1}{8}\rho_{\max}c^2$. | **PASSED** (`de Sitter tension err = 1.78e-15`; `p_theta(R_core) = +0.125` exact) |
| **Test 5: Kretschmann Bound $K(0) = \frac{24}{\ell^4}$ & $C^2$ Cusp Removal** | Evaluates $K(r) = (f'')^2 + \frac{4(f')^2}{r^2} + \frac{4(f-1)^2}{r^4}$ down to $r = 0$ and checks discrete 27-point $\nabla_{\text{iso}}\rho = 0, \Delta_{\text{iso}}\rho = 0$ inside the uniform frozen core. | **PASSED** ($K(0)\ell^4 = 24.000000000000007$; discrete interior $\nabla\rho = 0.0, \Delta\rho = 0.0$) |

### Module 3: 18-Point & 27-Point Stencil Isotropy & Lorentz Dispersion ([`tests/module_3_stencil_isotropy.py`](tests/module_3_stencil_isotropy.py))
*Goal: Prove the discrete cubic grid behaves like an isotropic continuum sphere and easily passes gamma-ray burst bounds.*

| Test | What is Verified in Code | Result |
| :--- | :--- | :--- |
| **Test 6: Taylor Moment & $\mathcal{O}(\ell^6)$ Body-Diagonal Cancellation** | Computes 2nd, 4th, and 6th-order stencil moments and discrete Fourier symbols $\tilde{\Delta}(\mathbf{k})$ for the 6-pt, 18-pt ($\frac{1}{3},\frac{1}{6},0$), and 27-pt ($\frac{7}{15},\frac{1}{10},\frac{1}{30}$) stencils. | **PASSED** ($m_2 = 2.0$, $m_4$ sphere ratio $= 2.0$, advection norm $= 1.0$, 27-pt $360W_{\text{corner}} - 12 = 0.0$) |
| **Test 7: 3D Euclidean Shell Angular Faceting ($R = 3.0\ell$)** | Propagates a 3D spherical wave on the $25^3$ lattice and measures $\sigma_\Omega(\Phi)/\langle\lvert\Phi\rvert\rangle$ across the 30 exact integer lattice points on $X^2 + Y^2 + Z^2 = 9\ell^2$. | **PASSED** (`6.68%` on 6-pt $\to$ `0.55%` on 18-pt [`12.2x`] $\to$ `0.40%` on 27-pt [`16.5x`]) |
| **Test 8: Fermi-LAT GRB 090510 Dispersion** | Evaluates group velocity $v_g(k) = \partial\omega/\partial k$ for 31 GeV photons ($\lambda = 4 \times 10^{-17}\text{ m}$) over $T = 2.3 \times 10^{17}\text{ s}$ (`7.3 Gly`). | **PASSED** (linear dispersion $= 0.0$; quadratic delay $\Delta t = 1.85 \times 10^{-19}\text{ s} \ll 0.8\text{ s}$) |

### Module 4: Horizon Crossing, Cayley Symplectic Evolution, & Stefan Core Freezing ([`tests/module_4_horizon_and_freezing.py`](tests/module_4_horizon_and_freezing.py))
*Goal: Prove waves cross the horizon smoothly, conserve energy to machine precision, and freeze onto the core without inner-horizon mass inflation.*

| Test | What is Verified in Code | Result |
| :--- | :--- | :--- |
| **Test 9: Supersonic River & Inner Horizon Excision** | Evaluates $\lvert J(r)\rvert$ across all three zones and solves $r^3 - r_s r^2 + r_s\ell^2 = 0$ across a mass sweep to confirm $r_- < R_{\text{core}}(M) < r_+$. | **PASSED** ($\lvert J(r_s)\rvert = c$, $\lvert J(R_{\text{core}})\rvert = 3c > c$, $r_- = 1.019\ell < R_{\text{core}} = 3.0\ell$) |
| **Test 10: Skew-Symmetric & Cayley Symplectic Conservation** | Verifies $M + M^T = 0$, $S(\Delta t)^T \Omega S(\Delta t) = \Omega$, and Hamiltonian conservation across 200 supersonic horizon-crossing steps. | **PASSED** ($M + M^T$ `err = 0.0`; symplectic `err = 6.66e-16`; energy `drift = 1.33e-14`) |
| **Test 11: Dynamic 3D Stefan Cell-Freezing ($25^3$ Grid)** | Runs the 3D $25\times 25\times 25$ lattice with local Stefan link-decoupling ($K_{ij}\to 0, M_{ij}\to 0$ at $H_i \ge E_{\text{voxel},\max}$). | **PASSED** ($N_{\text{cube}}: 27 \to 57$ cells; interior activity `0.0`; total energy `drift = 1.40e-8`) |
| **Test 12: Vaidya Horizon Vise & Bounced-Wave Trapping** | Evaluates $v_{\text{bounce}}(R_{\text{core}}) = c(1 - \sqrt{r_s(M+\Delta M)/R_{\text{core}}(M)})$ and 5-wave shell accretion onto the core. | **PASSED** ($v_{\text{bounce}} = -0.140c < 0$ trapped; interior quadratic metric `err = 3.33e-16`) |
| **Test 13: Unfrozen Mass Inflation vs. Frozen Core Stability** | Back-to-back comparison of Run A (`cap_density=False` + inner-horizon blueshift $\exp(2\lvert\kappa_-\rvert t)$ at $r_-$) vs. Run B (`cap_density=True` Stefan core freezing at $R_{\text{core}} > r_-$). | **PASSED** (Unfrozen 3D peak $= 2.16\,\rho_{\max}$ and Cauchy blueshift $= 6.59 \times 10^{10}$; Frozen active peak $= 0.993\,\rho_{\max} \le 1.0$) |

### Module 5: Non-Spherical 3D Flux & Spinning Kerr–Doran Ergosphere ([`tests/module_5_nonspherical_and_spin.py`](tests/module_5_nonspherical_and_spin.py))
*Goal: Prove the 3D lattice handles off-center, non-spherical waves and spinning black holes with exact energy conservation.*

| Test | What is Verified in Code | Result |
| :--- | :--- | :--- |
| **Test 14: 3D Asymmetric Discrete Gauss–Poynting Flux** | Verifies $F_{ij} + F_{ji} = 0$ and $\frac{d}{dt}\sum_{i\in\Omega} E_i^{\text{local}} = -\sum_{\partial\Omega} F_{ij}$ for an off-center, non-spherical 3D wave packet across $R \le 3.2\ell$. | **PASSED** (link antisymmetry `1.39e-17`; Gauss–Poynting flux balance `residual = 1.78e-15`) |
| **Test 15: 3D Spinning Kerr–Doran Whirlpool & Frame-Dragging** | Evolves a 3D wave packet on the 27-point lattice in the spinning flow $\mathbf{J} = J_r\hat{\mathbf{r}} + J_\phi\hat{\boldsymbol{\phi}}$ ($a_{\text{spin}} = 0.85\,r_s$). | **PASSED** (`16` supersonic ergosphere cells; $\lvert\mathbf{J}(0)\rvert = 0$; $M^T = -M$ `err = 0.0`; Cayley `drift = 6.45e-15`; Lense–Thirring shift $\Delta\langle y\rangle = -0.206\ell$) |

### Module 6: Quantum Operator Unitarity & Remnant Thermodynamics ([`tests/module_6_quantum_and_thermodynamics.py`](tests/module_6_quantum_and_thermodynamics.py))
*Goal: Prove the model is quantum-mechanically unitary and resolves the final stage of Hawking evaporation into a stable zero-temperature remnant.*

| Test | What is Verified in Code | Result |
| :--- | :--- | :--- |
| **Test 16: Heisenberg Commutator, Ehrenfest, & Entanglement** | Verifies $[\hat{\Phi}_i(t), \hat{\Pi}_j(t)] = i\hbar\delta_{ij}$ over 200 horizon steps, Ehrenfest equivalence, branch superposition $2.0000$ bending, and Gaussian entanglement entropy $S_{\text{ent}} > 0$. | **PASSED** (`commutator drift = 4.12e-14`; Ehrenfest diff `0.0`; $S_{\text{ent}} = 1.110$) |
| **Test 17: Hawking Surface Gravity, Specific Heat Flip, & First Law** | Verifies $\kappa(r_+) = \frac{c^2}{2r_+}(1 - 3\ell^2/r_+^2)$, peak $T_H$ at $r_+ = 3.0\ell$ ($C_M > 0$ below $3\ell$), $T_H(\sqrt{3}\ell) = 0$, and $c^2 dM = T_H dS_{\text{reg}}$. | **PASSED** ($\kappa$ identity `err = 2.78e-17`; $\kappa(\sqrt{3}\ell) = 0$; First Law `residual = 2.22e-16`) |

### Module 7: Strong-Field Geodesics, Junction Conditions, Ringdown QNMs, & Wave Echoes ([`tests/module_7_strong_field_and_observables.py`](tests/module_7_strong_field_and_observables.py))
*Goal: Prove infalling observers experience bounded tidal compression rather than spaghettification, verify exact Israel–Darmois junction continuity and Tolman virial balance, confirm LIGO ringdown quasinormal modes match 1915 GR, and demonstrate zero post-merger echoes alongside ergosphere Penrose negative Killing energy.*

| Test | What is Verified in Code | Result |
| :--- | :--- | :--- |
| **Test 18: Painlevé–Gullstrand Raindrop Geodesic & Bounded Tidal Reversal** | Integrates $\frac{dr}{d\tau} = J(r)$ from $5\,r_s$ across $r_+$ to $R_{\text{core}}$ and evaluates radial geodesic deviation $\mathcal{A}_{\text{tidal}}/c^2 = -\frac{1}{2}f''(r)$. | **PASSED** ($g_{\mu\nu}u^\mu u^\nu = -c^2$ `err = 0.0`; finite $\tau_{\text{core}} = 200.89$; tidal stretching vanishes at $r_{\text{tidal}} = \left(\frac{7+3\sqrt{5}}{2}\right)^{1/3}R_{\text{core}} \approx 1.89955\,R_{\text{core}}$, flips to compression $-\frac{5}{8\ell^2}$ at $R_{\text{core}}$, and is bounded by $\frac{1}{\ell^2}$ at $r = 0$) |
| **Test 19: Israel–Darmois Phase-Boundary Junction & Tolman Virial Integral** | Verifies $[f]^+_- = 0, [J]^+_- = 0$ ($\sigma_{\text{shell}} = 0$) and $[f']^+_- = \frac{8\pi G\rho_{\max}}{c^2}R_{\text{core}}$ at $r = R_{\text{core}}(M) = (r_s\ell^2)^{1/3}$, plus the 3D Tolman virial integral $\int_0^\infty 4\pi r^2(p_\theta - p_r)\,dr = \frac{3}{2}M c^2$. | **PASSED** ($[f]^+_- = 0.0$, $[J]^+_- = 4.44\times 10^{-16}$, $[f']^+_-$ `err = 0.0`, Tolman virial integral `rel err = 5.80e-15`) |
| **Test 20: Regge–Wheeler $\ell = 2$ Barrier & WKB Quasinormal Mode Ringdown** | Evaluates the $\ell = 2$ Regge–Wheeler potential $V_{\text{RW}}(r) = f(r)\left(\frac{6}{r^2} - \frac{6GM(r)}{c^2 r^3}\right)$ and Schutz–Will WKB fundamental ringdown frequency $(\omega_R, \omega_I)$ against 1915 Schwarzschild GR. | **PASSED** ($r_{\text{peak}} = 1.6403882\,r_s$; relative QNM shift $\Delta\omega_R/\omega_R = 1.93\times 10^{-9}$, $\Delta\omega_I/\omega_I = 4.64\times 10^{-9}$ at $r_s/\ell = 10^4$) |
| **Test 21: Time-Domain Wave-Echo Absence & Ergosphere Penrose Superradiance** | Evolves a radial wave packet reflecting off an outside-horizon shell ($R_{\text{shell}} = 1.15\,r_s > r_s$) vs. our inside-horizon frozen core ($R_{\text{core}} = 0.65\,r_s < r_s$), and constructs a negative-Killing-energy Penrose mode on the 3D 27-pt spinning lattice. | **PASSED** (Outside-horizon shell echo $= 0.0707 > 0$ vs. inside-horizon frozen core echo $= \mathbf{0.0}$; ergosphere Penrose mode has $E_{\text{comoving}} = +0.00645 > 0$ and $H_{\text{Killing}} = -0.00624 < 0$) |
