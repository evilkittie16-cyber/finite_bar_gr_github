# Finite-Bar Discrete General Relativity: A Nonsingular Black Hole Model

[![Verify Finite-Bar Discrete GR Benchmarks](https://github.com/evilkittie16-cyber/finite_bar_gr_github/actions/workflows/verify.yml/badge.svg)](https://github.com/evilkittie16-cyber/finite_bar_gr_github/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Self-contained manuscript, analytical verification suite, and 3D $25 \times 25 \times 25$ (`15,625-cell`) isotropic lattice simulator for the **spherically symmetric Painlevé–Gullstrand / Misner–Sharp reduction and scalar wave sector** of Finite-Bar Discrete General Relativity.

![Finite-Bar Discrete GR Summary](figures/finite_bar_gr_summary.png)

---

## Repository Contents

| File | Description |
| :--- | :--- |
| [`proof_corrected.md`](proof_corrected.md) | Complete mathematical manuscript covering the Planck-scale UV lattice postulates ($\ell = \ell_P, \Delta t_{\min} = t_P, \rho_{\max} = \frac{3c^2}{8\pi G\ell^2}$), 2PN weak-field isotropic & Painlevé–Gullstrand gauge match, de Sitter / Hayward core unification ($R_{\text{core}}(M) = (r_s\ell^2)^{1/3}, K(0) = \frac{24}{\ell^4}$), 18-neighbor & 27-point isotropic 3D stencils, skew-symmetric summation-by-parts (SBP) river Hamiltonian ($M_{ij} = -M_{ji}$), Cayley symplectic evolution, and Neumann link-decoupled Stefan core accretion. |
| [`verify_core_finite_bar_gr.py`](verify_core_finite_bar_gr.py) | Self-contained 7-benchmark verification suite testing all analytical and numerical identities in Sections 1–5 of the manuscript. |
| [`verify_core_finite_bar_gr_results.json`](verify_core_finite_bar_gr_results.json) | Pre-computed JSON benchmark output from `verify_core_finite_bar_gr.py`. |
| [`simulate_3d_18pt_freezing_core.py`](simulate_3d_18pt_freezing_core.py) | Shortcut-free 3D $25 \times 25 \times 25$ (`15,625-cell`) lattice wave simulator comparing uncapped inner-horizon pile-up against local Stefan cell freezing across 6-point cubic, 18-neighbor isotropic, and 27-point isotropic stencils. |
| [`simulate_3d_18pt_freezing_core_results.json`](simulate_3d_18pt_freezing_core_results.json) | Pre-computed JSON simulation output from `simulate_3d_18pt_freezing_core.py`. |
| [`generate_figures.py`](generate_figures.py) | Reproducible figure generation script that produces the 4-panel summary plot [`figures/finite_bar_gr_summary.png`](figures/finite_bar_gr_summary.png). |
| [`CITATION.cff`](CITATION.cff) & [`LICENSE`](LICENSE) | Standard academic citation metadata and MIT open-source license. |
| [`requirements.txt`](requirements.txt) | Python dependencies (`numpy>=1.20.0`, `matplotlib>=3.5.0`). |

---

## Quickstart & Reproducing All Benchmarks

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/evilkittie16-cyber/finite_bar_gr_github.git
cd finite_bar_gr_github
pip install -r requirements.txt
```

### 2. Run the 7 Core Mathematical & Physical Verification Benchmarks
```bash
python verify_core_finite_bar_gr.py
```
Expected console confirmation:
```text
ALL CORE FINITE-BAR GR TESTS PASSED: True
```

### 3. Run the 3D $25 \times 25 \times 25$ Isotropic Lattice Freezing-Core Simulator
```bash
python simulate_3d_18pt_freezing_core.py
```
Expected console confirmation:
```text
ALL 3D 18-PT & 27-PT LATTICE SIMULATOR CHECKS PASSED: True
```

### 4. Regenerate the 4-Panel Summary Figure
```bash
python generate_figures.py
```

---

## Summary of Verification Benchmarks

| Benchmark | Physical / Mathematical Check | Result |
| :--- | :--- | :--- |
| **1. Outside GR Match in Both Gauges ($r \gg R_{\text{core}}$)** | Weak-field isotropic ($\psi^2/w$) and Painlevé–Gullstrand ($(1 - J^2/c^2)^{-1}$) light-bending & 2PN expansions ($\frac{7}{4}$ and $4$) | **PASSED** (`2.0000` leading-order GR factor; `3PN residual = 8.0e-6` at $r = 1000\,GM/c^2$) |
| **2. Event Horizon Crossing & Inner Cauchy Horizon Excision** | Unified river flow $J(r) = -c\sqrt{\frac{r_s r^2}{r^3 + R_{\text{core}}(M)^3}}$ across $r_s$ and supersonic core boundary $\lvert J(R_{\text{core}})\rvert > c$ | **PASSED** (excises inner Cauchy horizon $r_- \approx \ell$ at $R_{\text{core}}(M) = (r_s\ell^2)^{1/3} \gg \ell$) |
| **3. 18-Pt & 27-Pt Stencil Isotropy & $16.5\times$ Faceting Reduction** | Elimination of $\mathcal{O}(\ell^2)$ and $\mathcal{O}(\ell^4)$ spherical anisotropy plus $\mathcal{O}(\ell^6)$ body-diagonal ($k_x^2 k_y^2 k_z^2$) error | **PASSED** ($R=3\ell$ shell angular anisotropy cut from `6.68%` on 6-pt to `0.55%` on 18-pt and `0.40%` on 27-pt) |
| **4. Symplectic & Co-Moving Energy Conservation ($\frac{dH}{dt} = 0$)** | Skew-symmetric $+M / -M^T$ Cayley propagation (Benchmark 4a) and 3D $25^3$ dynamic cell freezing (Benchmark 4b) | **PASSED** (`drift = 9.17e-15` fixed background; `drift = 1.29e-8` on 18-pt and `1.40e-8` on 27-pt during 3D cell freezing) |
| **5. Surface Freezing & Horizon Trapping ($R_{\text{core}} \propto M^{1/3}$)** | Bounced-wave trapping $v_{\text{bounce}}(R_{\text{core}}) = c(1 - \sqrt{r_s/R_{\text{core}}}) < 0$ and uniform-sphere growth $M(r) = M\frac{r^3}{R_{\text{core}}^3}$ | **PASSED** (`v_bounce = -0.140c < 0` above threshold; `3.33e-16` match to $\frac{r^3}{r} = r^2$ core law) |
| **6. Quantum Operator Evolution & Branch Superposition** | Unitary Heisenberg commutator $[\hat{\Phi}_i, \hat{\Pi}_j] = i\hbar\,\delta_{ij}$, Ehrenfest trajectory equivalence, and horizon entanglement | **PASSED** (`commutator drift = 4.12e-14`; Ehrenfest quantum vs. classical GR difference `0.0`) |
| **7. Observational, SEC/Buchdahl, & Remnant Shut-Off Tests** | Fermi-LAT GRB 090510 group-velocity bound, SEC bypass ($p_r = -\rho_{\max}c^2$), smooth Hayward $p_\theta(r)$, and remnant shut-off $\max_r\lvert J(r)\rvert = c$ | **PASSED** ($\Delta t_{\text{iso}} = 1.85 \times 10^{-19}\text{ s} \ll 0.8\text{ s}$; remnant shut-off at $M_{\text{crit}}^{\text{sharp}} = 0.5\,m_P$ and $M_{\text{crit}}^{\text{smooth}} = \frac{3\sqrt{3}}{4}m_P \approx 1.299\,m_P$) |
