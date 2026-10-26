# Finite-Bar Discrete General Relativity: Lattice Wave Dynamics on a Planck-Density (Hayward) Black Hole — What Works, What Does Not, and What Is Still Open

## Abstract
We study a scalar test field on a cubic lattice of spacing $\ell = \ell_P = \sqrt{\hbar G/c^3}$ that propagates on a fixed, spherically symmetric Painlevé–Gullstrand (PG) background whose density is capped at $\rho_{\max} = \frac{3c^2}{8\pi G\ell^2}$ (one cubic voxel holds at most $E_{\text{voxel},\max} = \frac{3}{8\pi}E_P$). The background is Hayward's (2006) regular black hole $\frac{2GM(r)}{c^2 r} = \frac{r_s r^2}{r^3 + r_s\ell^2}$ with core scale $R_{\text{core}} = (r_s\ell^2)^{1/3}$, which is also, up to a factor $2^{-1/6}$, the radius at which the Schwarzschild curvature reaches the de Sitter limiting value $24/\ell^4$. Our results are: (i) outside the horizon the model reproduces GR (1PN light bending, photon sphere, $\ell = 2$ ringdown) — by construction, since the metric is an input; (ii) the 18-neighbor and 27-point isotropic stencils remove $\mathcal{O}(\ell^2)$ and $\mathcal{O}(\ell^4)$ anisotropy, giving a tree-level GRB 090510 delay of $1.85\times10^{-19}$ s from the spatial lattice (about $5.6\times10^{-19}$ s if time is also discretised at $t_P$); at sixth order the 18-point stencil is the *more* isotropic of the two; (iii) a skew-symmetric (summation-by-parts) Killing Hamiltonian with the correct black-hole sign, integrated with the Cayley midpoint map, conserves energy to round-off and traps packets inside $r_+$; (iv) the Hayward core violates the Strong but satisfies the Null Energy Condition, so it evades Penrose's theorem through its inner Cauchy horizon, not through SEC violation; (v) in clean test-field simulations the blueshift at the inner horizon $r_-$ grows like $1/h$ with lattice spacing $h$; a local saturation cap keeps the field below $\rho_{\max}$ but converts the excess into reservoir mass of $4.8$–$7.9\times$ the infalling packet's energy (still growing as $h \to 0$), deposited within $\ell$ of $r_-$, so **the cap does not by itself resolve mass inflation**; (vi) in a dynamical collapse on a falling lattice (the grid carries the geometry), a cap on each cell's energy cannot prevent the singularity, because causal pressure cannot enforce it and empty cells still reach infinite curvature, whereas a cap on curvature (mean enclosed density) makes every shell bounce at exactly $R_{\text{core}}(m)$ before the shells cross; (vii) a saturated core can store at most $\pi r_s/\ell_P$ nats, short of $S_{\text{BH}}$ by a factor $r_s/\ell_P \approx 1.8\times10^{38}$ for a solar mass. We list three self-consistent completions (a spacelike limiting-curvature transition, a bounce, or a retained inner horizon), state which original claims are not supported, and identify what remains open: following the bounce through shell crossing, and a physical argument for whether the cap acts by repulsion or by phase change.

> [!IMPORTANT]
> **Scope and status.** The wave sector (Sections 1–4.5) is a *test scalar field on a prescribed background* plus a phenomenological saturation rule. Einstein's equations are solved only in the spherically symmetric, pressureless collapse code of Section 4.6. There is no spin-2 sector, and which completion of the black-hole interior the lattice rule selects is only partly answered (Section 4.6). Every claim below is tied to a script in this repository; `python verify_core_finite_bar_gr.py` prints a **claims audit** that marks each headline claim as SUPPORTED, NOT SUPPORTED, PARTLY ANSWERED, or NOT TESTED. Earlier versions of this manuscript claimed that frozen lattice cells excise the inner horizon, pin reflected waves on a static core surface, and resolve mass inflation; those claims are withdrawn here for the reasons given in Section 4.

---

## 0. Status at a Glance (read this first)

**What works (and is verified by the code):**
1. **Lattice wave numerics.** On a fixed background, the skew-symmetric Killing Hamiltonian with the black-hole sign, integrated with the Cayley map, conserves energy to round-off ($\sim10^{-14}$), is symplectic, and carries waves inside $r_+$ inward (Module 4).
2. **Isotropic stencils.** The 18- and 27-point stencils are isotropic through $\mathcal{O}(\ell^4)$. The tree-level dispersion delay for GRB 090510 is $\sim10^{-19}$ s, far below the 0.8 s bound (Module 3).
3. **GR on a falling lattice.** For spherically symmetric pressureless collapse, a lattice whose cells are free-falling shells reproduces the exact Einstein solution (error $3\times10^{-6}$). Here the grid really is the space: cell volumes evolve and the river velocity is set by the enclosed mass (Module 9).
4. **A curvature cap regularises collapse.** Capping the mean enclosed density (a curvature measure) with LQC-type dynamics makes every shell bounce at exactly $R_{\text{core}}(m) = (r_s\ell^2)^{1/3}$ (Module 9).
5. **Correct closed-form results:** $\rho_{\max}$ and $R_{\text{core}}$ (given the postulates), the Hayward stresses, $K(0) = 24/\ell^4$, Hayward's surface gravity and cold-remnant thermodynamics, the core entropy bound.

**What does not work, and why:**
1. **GR outside the horizon is not derived from the grid.** The exterior metric is an input, and the light-bending, photon-sphere and ringdown checks are continuum formulas, not lattice simulations. The lattice only shows that it does not spoil GR at observable scales.
2. **The inner horizon is not excised.** $r_-$ lies inside $R_{\text{core}}$ but remains part of the Hayward geometry; stopping a test field there does not change the spacetime.
3. **No static "frozen core" can exist inside the horizon.** Surfaces of constant $r$ inside $r_+$ are spacelike, so a static surface there would need faster-than-light worldlines. Every outgoing signal there moves inward anyway, so "pinning" reflected waves is not a mechanism.
4. **The saturation cap does not resolve mass inflation.** It keeps the field below $\rho_{\max}$, but moves the blueshifted energy into core mass at $r_-$ (4.8–7.9 times the infalling energy, still growing with resolution) instead of removing it.
5. **A cap on each cell's energy does not prevent the singularity.** In dynamical collapse $\rho_{\max}$ is reached only inside a trapped region. A causal pressure cannot hold the density down, and empty cells, which have zero energy density, still fall to infinite curvature. The rule as originally stated (postulate 3 below) is therefore insufficient; it has to act on curvature.
6. **SEC violation is not what avoids the singularity theorems.** Penrose's theorem uses the NEC, which the Hayward core satisfies.
7. **Quantum unitarity is not achieved.** The linear lattice is unitary, but the saturation step is not, and a saturated core can hold $\sim10^{38}$ times too little information for a solar-mass black hole.
8. **There is no spin-2 / gravitational-wave sector,** and the rotating flow is not the Kerr metric.
9. **The 3D "self-gravity" simulation is not GR.** Its pile-up comes from an ad-hoc update that changes the energy by about 5%.
10. **There are no distinctive testable predictions yet.**

**What is open:**
1. Following the curvature-cap bounce through shell crossing. This needs shock capturing and pressure, and it would determine if and when matter re-emerges, which is where predictions could come from.
2. Deciding on physical grounds whether the cap acts by repulsion (bounce, V2) or by conversion into vacuum energy (V1). The postulates do not decide this.
3. Lorentz invariance of a fundamental lattice beyond tree level.

---

## 1. Curvature Breakdown in Continuous GR & The Planck-Scale UV Lattice Postulates

### 1.1 Where Continuous General Relativity Breaks ($r \to 0$)
In standard continuous General Relativity, spacetime is modeled as a smooth manifold divisible down to zero-volume points ($r \to 0, \ell = 0$).
1. **Where Standard GR Works ($r \ge r_s$):** Outside a black hole down to the event horizon ($r_s = \frac{2GM}{c^2}$), Einstein's field equations match all current observations (planetary orbits, light deflection, gravitational redshift, EHT horizon shadows, and LIGO–Virgo–KAGRA merger waveforms).
2. **Where Standard GR Breaks ($r \to 0$):** At the center of a classical Schwarzschild black hole, curvature and energy-density invariants diverge—specifically, the mass density $\rho = \frac{M}{V} \to \infty$ and the Kretschmann curvature scalar $K \equiv R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma} = \frac{12\,r_s^2}{r^6} \to \infty$ blow up as $r \to 0$, while continuous quantum field theory near a horizon suffers from trans-Planckian ultraviolet ($k \to \infty$) mode divergence (**Unruh 1995; Corley & Jacobson 1996**).

### 1.2 The Planck-Scale UV Lattice Cutoff Postulates ($\ell > 0, \Delta t_{\min} > 0, \rho_{\max} < \infty$)
We introduce three **ultraviolet (UV) lattice cutoff postulates** motivated by Planck-scale physics. They are postulates, not derivations. The intent was to bound curvature and energy density. Section 4.6 shows that a bound on energy density alone (postulate 3) does **not** bound curvature, because empty regions can still reach infinite curvature; the cap has to act on curvature.
1. **Minimum Spatial Cell Scale ($\ell > 0$):** Space is discretized into finite 3D cubic voxels of minimum non-zero width $\ell = \ell_P = \sqrt{\frac{\hbar G}{c^3}} > 0$, bounding the maximum lattice wavenumber along each axis at the Brillouin–Nyquist cutoff $k_{\max} = \frac{\pi}{\ell}$ ($\lambda_{\min} = 2\ell$).
2. **Minimum Reaction Tick ($\Delta t_{\min} > 0$):** Local causal state updates require a shortest non-zero reaction tick $\Delta t_{\min} = t_P = \sqrt{\frac{\hbar G}{c^5}} > 0$, fixing the maximum causal propagation speed across the lattice at:

$$c = \frac{\ell}{\Delta t_{\min}} = \frac{1\text{ cell}}{1\text{ tick}} = 1 \quad (\text{in natural grid units})$$

3. **Maximum Voxel Energy-Density Ceiling ($\rho_{\max} < \infty$), Core Voxel Count ($N_{\text{cube}} \propto M/m_P$), & Bekenstein Entropy Bound:** Each cubic lattice voxel of geometric volume $\Delta V_{\text{voxel}} = \ell^3$ carries a maximum energy capacity:

$$E_{\text{voxel},\max} = \rho_{\max}c^2\ell^3 = \frac{3}{8\pi}E_P \approx 0.119366\,E_P$$

   so that a ball of radius $\ell$ ($\Delta V_{\text{sphere}} = \frac{4}{3}\pi\ell^3$) holds at most $E_{\max} = \frac{4\pi}{3}E_{\text{voxel},\max} = \frac{1}{2}E_P = \frac{\hbar}{2\Delta t_{\min}}$. The factor $\frac12$ is a choice. It is equivalent to requiring that a Planck-radius ball at $\rho_{\max}$ sits exactly at its own Schwarzschild radius ($2GE_{\max}/c^4 = \ell$). Other $\mathcal{O}(1)$ choices would change $\rho_{\max}$ and $R_{\text{core}}$ by $\mathcal{O}(1)$ factors. This fixes the saturation density ceiling used throughout:

$$\ell = \sqrt{\frac{\hbar G}{c^3}}, \qquad \Delta t_{\min} = \frac{\ell}{c} = \sqrt{\frac{\hbar G}{c^5}}, \qquad \rho_{\max} = \frac{E_{\text{voxel},\max}/c^2}{\ell^3} = \frac{E_{\max}/c^2}{\frac{4}{3}\pi\ell^3} = \frac{3c^2}{8\pi G\ell^2} = \frac{3c^5}{8\pi\hbar G^2} < \infty$$

   When a total mass $M$ saturates a spherical core of radius $R_{\text{core}}(M) = \left(\frac{3M}{4\pi\rho_{\max}}\right)^{1/3} = (r_s\ell^2)^{1/3}$, the number of saturated spherical influence units $N_{\text{sphere}}$ and the number of saturated cubic lattice voxels $N_{\text{cube}}$ are:

$$N_{\text{sphere}} = \left(\frac{R_{\text{core}}}{\ell}\right)^3 = \frac{r_s}{\ell_P} = \frac{2M}{m_P}, \qquad N_{\text{cube}} = \frac{\frac{4}{3}\pi R_{\text{core}}^3}{\ell^3} = \frac{4\pi}{3}\frac{r_s}{\ell_P} = \frac{8\pi}{3}\frac{M}{m_P} \approx 7.65 \times 10^{38}\left(\frac{M}{M_\odot}\right)$$

   satisfying the exact mass identity $N_{\text{cube}}\,E_{\text{voxel},\max} = N_{\text{sphere}}\,E_{\max} = M c^2$.
   - **Important Consequence for Black Hole Entropy (Bekenstein Core Bound vs. Bekenstein–Hawking Area Entropy):** Because $N_{\text{cube}} \propto \frac{r_s}{\ell_P} \propto M^1$ scales linearly with mass (a 1D scaling in $r_s/\ell_P$) and each Planck-sized cell of energy $E_{\text{voxel},\max} = \frac{3}{8\pi}E_P$ has a maximum Bekenstein entropy $\frac{S_{\text{cell}}}{k_B} \le \frac{2\pi E_{\text{voxel},\max}\ell_P}{\hbar c} = \frac{3}{4}$, the total thermodynamic entropy that can be stored inside the saturated core is bounded by:

$$\frac{S_{\text{core}}}{k_B} \le N_{\text{cube}}\left(\frac{3}{4}\right) = \pi\frac{r_s}{\ell_P} \approx 5.74 \times 10^{38}\left(\frac{M}{M_\odot}\right) \ll \frac{S_{\text{BH}}}{k_B} = \frac{\pi r_s^2}{\ell_P^2} \approx 1.05 \times 10^{77}\left(\frac{M}{M_\odot}\right)^2$$

   falling short of the Bekenstein–Hawking entropy $S_{\text{BH}}$ by the exact factor $\frac{S_{\text{BH}}}{S_{\text{core}}} = \frac{r_s}{\ell_P} \approx 1.83 \times 10^{38}\left(\frac{M}{M_\odot}\right)$ ($38$ orders of magnitude for a solar-mass black hole, verified in Module 8, Test 25). Therefore, **neither a static saturated core nor a Planck-mass remnant can store the Bekenstein–Hawking information of an astrophysical black hole**; preserving global quantum unitarity requires either an effective-LQG bounce (Version 2 in Section 4.5) or information escape in the Hawking radiation prior to the remnant stage.

### 1.3 Event-Driven Cells, Local Flatness, and the 1915 GR Light-Bending Factor in Both Gauges
1. **No Wasted Processing in Empty Space ($H_i = 0$):** Where no wave gradient or momentum is present ($\Phi_i - \Phi_j = 0, \Pi_i = 0$), the lattice equations give $\dot{\Phi}_i = 0, \dot{\Pi}_i = 0$, and $H_i = 0$. Dormant cells remain static until a wave enters them.
2. **Local light speed vs. coordinate speed:** With cell inertia $a_i$ and link stiffness $K_i$, the coordinate wave speed is $\sqrt{K_i/a_i}$; choosing $a_i, K_i$ from the metric (below) makes this equal to the coordinate speed of light, so locally measured light speed is $c$. This is a property of how the weights are chosen from the input metric, not something the lattice produces by itself.
3. **The 1915 GR Factor of $2$ in Weak-Field Isotropic Coordinates vs. Painlevé–Gullstrand River Coordinates:**
   - **Weak-Field Isotropic Gauge ($r \gg r_s$):** In isotropic coordinates $ds^2 = -w^2 c^2 dt^2 + \psi^4(dx^2 + dy^2 + dz^2)$ with $\psi = 1 + \frac{GM}{2c^2\tilde{r}}$ and $w = \frac{1 - GM/(2c^2\tilde{r})}{\psi}$, the scalar wave density weight is $\sqrt{-g}\lvert g^{00}\rvert \propto \frac{\psi^6}{w}$ and spatial stiffness is $\sqrt{-g}g^{ij} \propto \psi^2 w$ (or up to conformal rescaling by $\psi$, $a_i = \frac{\psi_i^7}{w_i}$ and $K_i = \psi_i^3 w_i$). Since all three spatial directions share the same conformal factor $\psi^4$, the optical metric is isotropic with coordinate wave speed $c_{\text{coord}}(r) = \sqrt{\frac{K_i}{a_i}} = \frac{w}{\psi^2}$ and scalar refractive index:

$$n_{\text{iso}}(r) = \frac{\psi^2}{w} = 1 + \mathbf{2}\,\frac{GM}{c^2 r} + \frac{7}{4}\left(\frac{GM}{c^2 r}\right)^2 + \mathcal{O}\!\left(\frac{r_s^3}{r^3}\right)$$

   - **Horizon-Penetrating Painlevé–Gullstrand (River) Gauge & Anisotropic Optical Metric:** In the Painlevé–Gullstrand metric $ds^2 = -c^2 dt^2 + (dr - J(r)\,dt)^2 + r^2 d\Omega^2$ with $f(r) = 1 - \frac{J(r)^2}{c^2} = 1 - \frac{2GM(r)}{c^2 r}$, eliminating the cross-term $J(r)\,dr\,dt$ via the Schwarzschild time transformation $dt_S = dt - \frac{J(r)}{c^2 - J(r)^2}\,dr$ yields the **anisotropic Fermat optical metric**:

$$dl_{\text{opt}}^2 = c^2 dt_S^2 = \frac{dr^2}{f(r)^2} + \frac{r^2\,d\Omega^2}{f(r)} \implies n_r(r) = \frac{1}{f(r)}, \quad n_\perp(r) = \frac{1}{\sqrt{f(r)}}$$

     For radial light rays, the two-way harmonic round-trip index equals the radial optical index $n_{\text{PG}}(r) = \frac{1}{2}\left(\frac{c}{c + J(r)} + \frac{c}{c - J(r)}\right) = \frac{1}{f(r)}$:

$$n_{\text{PG}}(r) = \frac{1}{1 - \frac{2GM(r)}{c^2 r}} = 1 + \mathbf{2}\,\frac{GM}{c^2 r} + \mathbf{4}\left(\frac{GM}{c^2 r}\right)^2 + \mathcal{O}\!\left(\frac{r_s^3}{r^3}\right)$$

     while non-radial light rays grazing at impact parameter $b$ obey the exact Schwarzschild/Hayward null geodesic equation $\left(\frac{du}{d\phi}\right)^2 = \frac{1}{b^2} - u^2 f(1/u)$ ($u = 1/r$), which integrates to the exact 1915 Einstein deflection angle $\Delta\phi = \frac{4GM}{c^2 b} + \frac{15\pi}{4}\left(\frac{GM}{c^2 b}\right)^2 + \mathcal{O}((r_s/b)^3)$ (checked by 200-node Gauss–Legendre quadrature of the continuum geodesic equation in Module 1, Test 3).

   > [!NOTE]
   > Everything in this subsection is a property of the **input metric**, evaluated with continuum formulas. No test sends a wave packet across the lattice past a mass and measures its bending. The lattice's own contribution is the dispersion correction of Section 3, which is negligible at observable wavelengths.

---

## 2. Part I: 3D Finite-Bar Integration, Energy Conditions, & Unified Hayward Core

Instead of evaluating gravity at a zero-volume point $r = 0$, we integrate over **finite 3D cells (bars) of minimum width $\ell > 0$** capped at saturation density $\rho_{\max}$. This section describes **postulated static core profiles**. Section 4.6 shows that collapsing matter does not actually settle into such a static core: inside the horizon no static configuration is possible, and the dynamics instead lead to a bounce or a phase transition, depending on how the cap is enforced.

### 2.1 Exact 3D Spherical Core Integral, Energy Conditions, & Singularity Theorems
In 3D spherical coordinates, the volume of a thin shell of radius $r$ is $4\pi r^2\,dr$. If $N_{\text{sphere}} \ge 1$ central spherical units ($N_{\text{cube}} = \frac{4\pi}{3}N_{\text{sphere}}$ cubic voxels) are assumed to saturate at $\rho_{\max}$ with vacuum-like stress $p_r = -\rho_{\max}c^2$, they form a de Sitter core of radius:

$$R_{\text{core}}(M) = N_{\text{sphere}}^{1/3}\ell = \left(\frac{3 N_{\text{cube}}}{4\pi}\right)^{1/3}\ell = \left(\frac{3M}{4\pi\rho_{\max}}\right)^{1/3} = \left(r_s\,\ell^2\right)^{1/3} \ge \ell \quad \left(\text{where } \rho_{\max} = \frac{3c^2}{8\pi G\ell^2}\right)$$

Integrating $\frac{1}{r}$ across the spherical core of radius $R_{\text{core}}$ cancels $r$ in the denominator against $r^2$ in the volume element ($\int_0^{R_{\text{core}}} \frac{1}{r}\,4\pi r^2\,dr = 2\pi R_{\text{core}}^2 < \infty$). For a sharp uniform sphere of density $\rho_{\max}$ inside $r < R_{\text{core}}$ and vacuum outside $r \ge R_{\text{core}}$, the enclosed mass is $M(r) = M\frac{r^3}{R_{\text{core}}^3}$ inside $r < R_{\text{core}}$, giving:

$$\frac{2G\,M_{\text{sharp}}(r)}{c^2\,r} = \begin{cases} \dfrac{r_s}{r} & \text{for } r \ge R_{\text{core}}(M) \quad (\text{exact Schwarzschild GR outside the core}) \\[10pt] \dfrac{r_s}{R_{\text{core}}(M)^3}\,r^2 = \dfrac{r^2}{\ell^2} & \text{for } r < R_{\text{core}}(M) \quad (\text{regular de Sitter core—bounded curvature at } r = 0) \end{cases}$$

- **Relation to Buchdahl's (1959) Theorem, Hawking–Penrose (1970) SEC Violation, and Penrose's (1965) NEC Theorem:**
  1. **Buchdahl (1959) vs. Anisotropic de Sitter Interior:** Buchdahl's (1959) $\frac{9}{8}r_s$ bound applies to **static exterior** fluid spheres ($R > r_s$) with **isotropic positive pressure** ($p_r = p_\theta > 0$) and $\frac{d\rho}{dr} \le 0$. For a core inside the horizon ($R_{\text{core}} < r_s$), staticity already fails in the supersonic trapping zone ($r_- < r < r_+$); moreover, in the Painlevé–Gullstrand metric ($ds^2 = -c^2 dt^2 + (dr - J(r)\,dt)^2 + r^2 d\Omega^2$ with $\frac{J(r)^2}{c^2} = \frac{2G\,M(r)}{c^2 r}$ and $-g_{tt}g_{rr} = 1$, **Sakharov 1966; Dymnikova 1992; Hayward 2006**), Einstein's equations $G^\mu_\nu = \frac{8\pi G}{c^4}T^\mu_\nu$ for $f(r) = 1 - \frac{r_s}{R_{\text{core}}^3}r^2$ give negative radial vacuum tension:

$$G^t_t = G^r_r = \frac{r f'(r) + f(r) - 1}{r^2} = -\frac{3r_s}{R_{\text{core}}^3} = -\frac{8\pi G}{c^2}\rho_{\max} \implies p_r = -\rho_{\max}c^2$$

     coupled with anisotropic tangential stress $p_\theta(r) = p_r(r) + \frac{r}{2}\frac{dp_r}{dr} \neq p_r(r)$ across the core transition layer, replacing the divergent central pressure of Schwarzschild's (1916) constant-density interior fluid.
  2. **Hawking–Penrose (1970) SEC Violation vs. Penrose (1965) NEC & The Three Singularity-Avoidance Routes:** Inside the core, $p_r = -\rho_{\max}c^2$ **violates the Strong Energy Condition (SEC)**:

$$\rho_{\max}c^2 + p_r + 2p_\theta = -2\rho_{\max}c^2 < 0$$

     which reverses timelike geodesic focusing (geodesics defocus: the radial tidal term at the centre is $A_{\text{tidal}}(0)/c^2 = -\frac{1}{2}f''(0) = +\frac{1}{\ell^2}$, i.e. repulsive stretching, finite but Planckian, $\sim c^2/\ell_P^2 \sim 10^{86}\text{ s}^{-2}$ for $\ell = \ell_P$) and bypasses the **Hawking–Penrose (1970)** timelike convergence condition. However, **Penrose's (1965)** null singularity theorem assumes only the **Null Energy Condition (NEC)** ($T_{\mu\nu}k^\mu k^\nu \ge 0$), a non-compact Cauchy surface, and global hyperbolicity. For the static Hayward profile, radial null vectors satisfy $\rho c^2 + p_r = 0$ (marginal NEC) and tangential null vectors satisfy $\rho c^2 + p_\theta = \frac{3\rho_{\max}c^2 R_{\text{core}}^6 r^3}{(r^3 + R_{\text{core}}^3)^3} \ge 0$, so the classical NEC is not violated by the static background alone! Consequently, in the presence of a trapped region ($r < r_+$), any self-consistent nonsingular completion must evade Penrose (1965) by one of three named mechanisms (compared in Section 4.5; Section 4.6 shows that the rule must act on curvature, after which repulsive enforcement gives a V2 bounce and conversion to vacuum energy gives V1, so the route is only partly determined):
     - **Route V1 (Limiting-Curvature Phase Transition, Frolov–Markov–Mukhanov 1989):** Collapse hits $\rho = \rho_{\max}$ across a **spacelike** transition hypersurface at $R_{\text{core}} = (r_s\ell^2)^{1/3}$ and enters a de Sitter / baby-universe phase (evading Penrose via topology change, with no inner horizon $r_-$). For this to remove the singularity in collapse, the transition must also occur in the empty region around the matter, i.e. it must be triggered by curvature, not matter density (Section 4.6);
     - **Route V2 (Effective-LQG Saturation Bounce, Kelly–Santacruz–Wilson-Ewing 2020):** Quantum-gravity holonomy corrections modify the effective dynamical equations at $\rho \to \rho_{\max}$ (violating the effective dynamical NEC $\rho_{\text{eff}}c^2 + p_{\text{eff}} < 0$ during the bounce), so the collapsing matter bounces at $R_{\text{bounce}} = R_{\text{core}} = (r_s\ell^2)^{1/3}$. Shells then cross and a shock forms (Section 4.6), leading to an outgoing shock / white-hole phase whose timing is not computed here;
     - **Route V3 (Inner Cauchy Horizon & Extremal Remnant, Hayward 2006 / Carballo-Rubio et al. 2022):** The static regular metric retains its inner Cauchy horizon at $r_- \approx \ell$ (evading Penrose 1965 by breaking global hyperbolicity at $r_-$), allowing $r_+$ and $r_-$ to merge smoothly into a cold $T_H = 0$ remnant at $r_+ = r_- = \sqrt{3}\ell$. Local saturation damping bounds the *field's* energy density at $r_-$, but Section 4.4 shows that it moves the blueshifted energy into core mass at $r_-$ rather than removing it.

### 2.2 Reconciling the Sharp Piecewise Core with the Smooth Hayward Profile ($R_{\text{core}}(M) = (r_s\ell^2)^{1/3}$)
In spherical symmetry with $-g_{tt}g_{rr} = 1$, conservation of the stress-energy tensor ($\nabla_\mu T^\mu_\nu = 0$) requires the tangential pressure $p_\theta = p_\phi$ to satisfy:

$$p_\theta(r) = p_r(r) + \frac{r}{2}\frac{dp_r}{dr}$$

1. **Sharp Piecewise Core (Section 2.1):** In the sharp piecewise profile $M_{\text{sharp}}(r)$, $p_r$ jumps from $-\rho_{\max}c^2$ inside $r < R_{\text{core}}$ to $0$ outside $r > R_{\text{core}}$, so $\frac{dp_r}{dr} = +\rho_{\max}c^2\,\delta(r - R_{\text{core}})$, corresponding to an **Israel (1966) thin-shell surface tension layer** regularized over one lattice spacing $\Delta r \sim \ell$ at $r = R_{\text{core}}(M)$. Note that in the sharp profile, $f_{\text{sharp}}(r) = 1 - r^2/\ell^2$ inside $r < R_{\text{core}}$ has its root at $r = \ell$ with $f_{\text{sharp}}'(\ell) = -2/\ell \neq 0$, so the sharp profile by itself does **not** have a smooth double-root zero-temperature horizon merger unless smoothed over scale $\ell$.
2. **Smooth Continuum Regularization (Hayward 2006 Analytic Single-Chart Profile):** For smooth continuum PDE analyses and horizon thermodynamics without distributional $\delta(r - R_{\text{core}})$ junction terms, we use **Hayward's (2006) smooth enclosed-mass profile** sharing the **exact same mass-dependent core radius** $R_{\text{core}}(M) = (r_s\ell^2)^{1/3} = \left(\frac{3M}{4\pi\rho_{\max}}\right)^{1/3}$:

$$\frac{2G\,M_{\text{smooth}}(r)}{c^2\,r} = \frac{r_s\,r^2}{r^3 + R_{\text{core}}(M)^3} = \frac{r_s\,r^2}{r^3 + r_s\,\ell^2}$$

   This single unified profile satisfies all three physical requirements simultaneously:
   - **Outside ($r \gg R_{\text{core}}(M)$):** $\frac{2GM_{\text{smooth}}(r)}{c^2 r} = \frac{r_s}{r}\left[1 - \mathcal{O}\!\left(\frac{R_{\text{core}}^3}{r^3}\right)\right]$, matching standard continuous GR.
   - **Inside ($r \ll R_{\text{core}}(M)$):** $\frac{2GM_{\text{smooth}}(r)}{c^2 r} \to \frac{r_s}{R_{\text{core}}(M)^3}\,r^2 = \frac{r^2}{\ell^2}$, matching the exact de Sitter core of Section 2.1 with Kretschmann scalar bounded by $K(0) = \frac{24}{\ell^4} < \infty$.
   - **Smooth Tangential Pressure ($p_\theta$ Finite Everywhere):** Because $\rho(r) = \rho_{\max}\frac{R_{\text{core}}^6}{(r^3 + R_{\text{core}}^3)^2}$ is smooth for all $r \ge 0$ and $p_r(r) = -\rho(r)c^2$, the tangential pressure $p_\theta(r) = p_r(r) + \frac{r}{2}\frac{dp_r}{dr} = -\rho(r)c^2 - \frac{r}{2}c^2\frac{d\rho}{dr}$ has the exact closed form:

$$\frac{p_\theta(r)}{\rho_{\max}c^2} = \frac{R_{\text{core}}^6\left(2r^3 - R_{\text{core}}^3\right)}{\left(r^3 + R_{\text{core}}^3\right)^3}$$

     which is smooth and finite everywhere without a delta-function shell—transitioning from isotropic vacuum tension $p_\theta(0) = -\rho_{\max}c^2$ at $r = 0$, through zero at $r = 2^{-1/3}R_{\text{core}} \approx 0.794\,R_{\text{core}}$, to positive tangential stress $p_\theta(R_{\text{core}}) = +\frac{1}{8}\rho_{\max}c^2 > 0$ at $r = R_{\text{core}}(M)$.
   - **Smoothness at the origin:** Hayward's odd radial power $r^3 = (x^2 + y^2 + z^2)^{3/2}$ in $\rho(r) = \rho_{\max}(1 - 2r^3/R_{\text{core}}^3 + \mathcal{O}(r^6))$ makes the metric only finitely differentiable at the origin ($\rho'''(0) \neq 0$). Sampling it on a lattice does not change the continuum metric used elsewhere in this document, so this mild non-smoothness remains a property of the Hayward profile.

---

## 3. Part II: The 18-Point and 27-Point Isotropic 3D Grid Stencils

Following the isotropic finite-difference stencil derivations of **Shinozaki & Oono (1993)**, **Spotz & Carey (1996)**, and **Patra & Karttunen (2006)**, a $3\times 3\times 3$ cube around cell $i$ contains **$27$ total points** ($1\text{ center cell} + 26\text{ neighbor links}$):
- **6 face neighbors** (Simple Cubic, $\lvert\mathbf{d}\rvert = \ell$): $(\pm\ell, 0, 0)$ permutations with weight $W_{\text{face}}$,
- **12 edge-diagonal neighbors** (Face-Centered Cubic / FCC spheres, $\lvert\mathbf{d}\rvert = \sqrt{2}\,\ell$): $(\pm\ell, \pm\ell, 0)$ permutations with weight $W_{\text{edge}}$,
- **8 corner-diagonal neighbors** (Body-Centered Cubic / BCC corners, $\lvert\mathbf{d}\rvert = \sqrt{3}\,\ell$): $(\pm\ell, \pm\ell, \pm\ell)$ with weight $W_{\text{corner}}$.

Requiring exact $\mathcal{O}(\ell^2)$ and $\mathcal{O}(\ell^4)$ spherical isotropy ($2W_{\text{face}} + 8W_{\text{edge}} + 8W_{\text{corner}} = 2$ and $4W_{\text{edge}} + 8W_{\text{corner}} = \frac{2}{3}$) yields the one-parameter isotropic stencil family:

$$W_{\text{face}} = \frac{1}{3} + 4W_{\text{corner}}, \qquad W_{\text{edge}} = \frac{1}{6} - 2W_{\text{corner}}$$

which includes both the **18-neighbor (19-point) stencil** ($W_{\text{corner}} = 0$) and the **full 27-point (26-neighbor) isotropic stencil** ($W_{\text{corner}} = \frac{1}{30}$; cf. Patra & Karttunen 2006):

$$\text{\textbf{18-Neighbor (19-Pt):}}\quad W_{\text{face}} = \frac{1}{3},\quad W_{\text{edge}} = \frac{1}{6},\quad W_{\text{corner}} = 0 \qquad\Big|\qquad \text{\textbf{Full 27-Point:}}\quad W_{\text{face}} = \frac{7}{15},\quad W_{\text{edge}} = \frac{1}{10},\quad W_{\text{corner}} = \frac{1}{30}$$

For any smooth wave field $\Phi$, the discrete Laplacian on both stencils is:

$$\left(\Delta_{\text{iso}}\Phi\right)_i = \frac{1}{\ell^2}\sum_{\mathbf{d}\in\mathcal{N}} W_d\left(\Phi_{i+\mathbf{d}} - \Phi_i\right) = \nabla^2\Phi(\mathbf{x}_i) + \frac{\ell^2}{12}\left(\nabla^2\right)^2\Phi(\mathbf{x}_i) + \mathcal{O}(\ell^4)$$

1. **Exact Fourth-Order Spherical Isotropy ($\mathcal{O}(\ell^4)$):** The **6 face neighbors** give $(k_x^4 + k_y^4 + k_z^4)$ with cross-term coefficient $\mathbf{0}$, while the **12 edge** and **8 corner** neighbors satisfy $6\frac{4W_{\text{edge}} + 8W_{\text{corner}}}{2W_{\text{face}} + 8W_{\text{edge}} + 8W_{\text{corner}}} = \mathbf{2}$, giving the exact round 3D sphere at fourth order:

$$\left(k_x^4 + k_y^4 + k_z^4\right) + \mathbf{2}\left(k_x^2 k_y^2 + k_y^2 k_z^2 + k_z^2 k_x^2\right) = \left(k_x^2 + k_y^2 + k_z^2\right)^2 = \lvert\mathbf{k}\rvert^4$$

2. **Exact Sixth-Order Directional Analysis ($\mathcal{O}(\ell^6)$):** At sixth order, the normalized Taylor polynomial of any stencil in the $\mathcal{O}(\ell^4)$-isotropic family is:

$$\mathcal{P}_6(\mathbf{k}) = \left(k_x^6 + k_y^6 + k_z^6\right) + 5\left[k_x^4(k_y^2 + k_z^2) + k_y^4(k_z^2 + k_x^2) + k_z^4(k_x^2 + k_y^2)\right] + 360\,W_{\text{corner}}\,k_x^2 k_y^2 k_z^2$$

   which can be written equivalently as:

$$\mathcal{P}_6(\mathbf{k}) = \lvert\mathbf{k}\rvert^6 + 2\,\lvert\mathbf{k}\rvert^2\left(k_x^2 k_y^2 + k_y^2 k_z^2 + k_z^2 k_x^2\right) + \left(360\,W_{\text{corner}} - 12\right)k_x^2 k_y^2 k_z^2$$

   Evaluating $\mathcal{P}_6(\hat{\mathbf{n}})$ on unit vectors $\hat{\mathbf{n}}$ along the lattice axis $[1,0,0]$, face-diagonal $[1,1,0]/\sqrt{2}$, and body-diagonal $[1,1,1]/\sqrt{3}$ gives the exact directional ratio:

$$\mathcal{P}_6([1,0,0]) : \mathcal{P}_6\!\left(\frac{[1,1,0]}{\sqrt{2}}\right) : \mathcal{P}_6\!\left(\frac{[1,1,1]}{\sqrt{3}}\right) = 1 : \frac{3}{2} : \left(\frac{11}{9} + \frac{40}{3}W_{\text{corner}}\right)$$

   - For the **18-neighbor stencil** ($W_{\text{corner}} = 0$): the 6th-order directional ratio is $1 : \frac{3}{2} : \frac{11}{9} \approx 1 : 1.500 : 1.222$ (directional spread $\max - \min = 0.500$).
   - For the **27-point stencil** ($W_{\text{corner}} = \frac{1}{30}$): $360W_{\text{corner}} - 12 = 0$ removes the single monomial $k_x^2 k_y^2 k_z^2$ in the $(|\mathbf{k}|^6, |\mathbf{k}|^2\sum k_\alpha^2 k_\beta^2, k_x^2 k_y^2 k_z^2)$ decomposition, but its sixth-order directional ratio is $1 : \frac{3}{2} : \frac{5}{3} \approx 1 : 1.500 : 1.667$ (spread $0.667$, larger than the 18-neighbor stencil's $0.500$). The 3D shell test agrees: faceting 0.345% (18-pt) vs 0.545% (27-pt) (Module 3, Tests 6–7). No stencil in this $3\times3\times3$ family is isotropic at sixth order; that would need a larger stencil.
- **Observational Check Against Fermi-LAT Gamma-Ray Burst GRB 090510 (Abdo et al., *Nature* 2009) & The Radiative Loop Problem (Collins et al. 2004):** Fermi-LAT observed $31\text{ GeV}$ gamma-ray photons ($\lambda \approx 4 \times 10^{-17}\text{ m}$) arriving after $7.3\text{ billion light-years}$ ($T \approx 2.3 \times 10^{17}\text{ s}$) within $\Delta t < 0.8\text{ s}$ of low-energy photons, ruling out discrete models with linear $\mathcal{O}(\ell_P/\lambda)$ group-velocity dispersion ($\Delta t_{\text{linear}} \approx 0.093\text{ s}$). Because both the 18-neighbor and 27-point stencils are central-symmetric ($\Phi_{i+\mathbf{d}} + \Phi_{i-\mathbf{d}} - 2\Phi_i$) and have identical $\mathcal{O}(k^2)$ and $\mathcal{O}(k^4)$ dispersion $\omega(k) \approx ck\left(1 - \frac{1}{24}k^2\ell^2\right)$, the tree-level packet **group velocity** is:

$$v_g = \frac{\partial\omega}{\partial k} \approx c\left(1 - \frac{1}{8}k^2\ell^2\right) = c\left[1 - \frac{\pi^2}{2}\left(\frac{\ell_P}{\lambda}\right)^2\right]$$

  giving a tree-level arrival delay of $\Delta t_{\text{iso}} \approx \mathbf{1.85 \times 10^{-19}\text{ seconds}}$ from the spatial lattice ($19$ orders of magnitude inside the $0.8\text{ s}$ Fermi-LAT bound). If time is also discretised with ticks $\Delta t = t_P$ using the Cayley rule, the time step adds a term of the same order ($\omega \to \omega(1 - \omega^2\Delta t^2/12)$), raising the delay to $\approx 5.6\times10^{-19}$ s, still 18 orders below the bound and axis-vs-diagonal anisotropy $\mathcal{O}(\ell_P/\lambda)^4 \approx 2.67 \times 10^{-74}$. *Important theoretical caveat (**Collins, Perez, Sudarsky, Urrutia & Vucetich 2004**):* while tree-level lattice propagation is quadratically suppressed by $(\ell_P/\lambda)^2 \sim 10^{-36}$, in a fundamental interacting quantum field theory on a fixed lattice, radiative loop self-energies generically generate unsuppressed $\mathcal{O}(1)$ Lorentz-violating operators unless protected by a symmetry (such as supersymmetry or emergent IR Lorentz invariance) or unless the lattice is interpreted as an effective semiclassical UV regulator for the gravitational sector rather than a hard Lorentz-breaking cutoff for Standard Model loops (see Section 6.1).

---

## 4. Part III: Waves Across the Horizon, the Saturation Rule, and the Interior

### 4.1 River profile, horizons, and the trapped region
With the Hayward mass function the inward PG velocity of space is

$$J(r) = -c\sqrt{\frac{2GM(r)}{c^2 r}} = -c\sqrt{\frac{r_s\,r^2}{r^3 + r_s\,\ell^2}}, \qquad f(r) = 1 - \frac{J(r)^2}{c^2}.$$

$f$ has two roots for $r_s > \frac{3\sqrt3}{2}\ell$: an outer horizon $r_+ \simeq r_s$ and an inner (Cauchy) horizon $r_- \simeq \ell\,(1 + \ell/2r_s)$. For every mass, $r_- < R_{\text{core}} < r_+$ (Module 4, Test 9; for $r_s = 27\ell$: $r_- = 1.019\ell$, $R_{\text{core}} = 3\ell$, $r_+ = 26.96\ell$). Three facts organise everything that follows:

1. **Both horizons are part of the geometry.** Nothing in this model removes $r_-$. Stopping the evolution of a test field inside $R_{\text{core}}$ (as an earlier version of the code did) does not change the spacetime.
2. **Inside the trapped region $r_- < r < r_+$ both characteristic speeds $J \pm c$ are negative.** Any outward-moving signal there — including one "reflected" at $R_{\text{core}}$ — moves to smaller $r$. This follows from the background mass alone (Test 12; Benchmark 5).
3. **Surfaces of constant $r$ inside $r_+$ are spacelike.** A worldline at fixed $r$ has $g_{tt} = -(c^2 - J^2) > 0$. A static "frozen surface" at $R_{\text{core}} < r_+$ is therefore not a possible physical object; a transition at $R_{\text{core}}$ can only be a spacelike (instantaneous) layer, as in the Frolov–Markov–Mukhanov construction. At $R_{\text{core}}$, $J^2/c^2 = (r_s/\ell)^{2/3}$ (sharp profile) or $\frac12(r_s/\ell)^{2/3}$ (Hayward).

### 4.2 Energy-conserving lattice Hamiltonian with the black-hole sign
For a minimally coupled scalar in PG coordinates ($\sqrt{-g} = c$, flat spatial slices), the free-fall observer has $u^\mu = (1, \mathbf{J})$ and the canonical momentum is the convective derivative $\Pi = a\,(\partial_t + \mathbf{J}\cdot\nabla)\Phi$. On the lattice:

$$H_{\text{Killing}}(\Phi,\Pi) = \sum_i \frac{\Pi_i^2}{2a_i} - \sum_{i,j}\Pi_i (D_{\mathbf J})_{ij}\Phi_j + \frac14\sum_{i,j}K_{ij}(\Phi_i - \Phi_j)^2,$$

$$(D_{\mathbf J})_{ij} = \frac{1}{4\ell}\sum_{\mathbf d\in\mathcal N} W_{\mathbf d}\,\frac{(\mathbf J_i + \mathbf J_j)\cdot\mathbf d/\ell}{2}\left(\delta_{j,i+\mathbf d} - \delta_{j,i-\mathbf d}\right) = -(D_{\mathbf J})_{ji},$$

so that $\sum_j (D_{\mathbf J})_{ij}\Phi_j = \big(\mathbf J\cdot\nabla\Phi + \tfrac12(\nabla\cdot\mathbf J)\Phi\big)_i + \mathcal O(\ell^2)$ (the symmetric split of the volume-compression term corresponds to the Liouville-rescaled field and differs from the minimally coupled one only by a potential term that vanishes in the eikonal limit). Hamilton's equations are

$$\dot\Phi_i = \frac{\Pi_i}{a_i} - (D_{\mathbf J}\Phi)_i, \qquad \dot\Pi_i = -\sum_j K_{ij}(\Phi_i - \Phi_j) - (D_{\mathbf J}\Pi)_i,$$

with generator $F = \begin{psmallmatrix} -D_{\mathbf J} & A^{-1} \\ -L_K & -D_{\mathbf J}\end{psmallmatrix}$ and continuum characteristic speeds $dr/dt = J(r) \pm c$. On a fixed background the Cayley map $S = (I - \tfrac{\Delta t}{2}F)^{-1}(I + \tfrac{\Delta t}{2}F)$ is symplectic and conserves the quadratic $H_{\text{Killing}}$ exactly for any $\Delta t$. Module 4, Test 10: $\lVert D + D^T\rVert = 0$, $\lVert S^T\Omega S - \Omega\rVert = 5.6\times10^{-16}$, Killing drift $9.0\times10^{-15}$ over 200 steps, and a packet released at $r_0 = 4$ inside $r_+ \approx 5.9$ moves inward (energy centroid $3.87 \to 2.67$). An earlier version used $\dot\Phi = \Pi/a + D\Phi$, which describes a **white hole**; that error is fixed throughout the code.

### 4.3 Comoving energy versus Killing energy
The free-fall (comoving) energy density $E_i^{\text{local}} = \frac{\Pi_i^2}{2a_i} + \frac14\sum_{\mathbf d}W_{\mathbf d}K_0(\Phi_i - \Phi_{i+\mathbf d})^2 = T_{\mu\nu}u^\mu u^\nu\,\Delta V \ge 0$ is the quantity a local saturation rule must use. It is **not** conserved on a static background: $\frac{d}{dt}\sum_i E_i^{\text{local}} = -\Pi^T A^{-1}D\Pi - \tfrac12\Phi^T[L_K, D]\Phi \neq 0$. The conserved quantity is $H_{\text{Killing}}$, which is indefinite inside $r_+$. Growth of the comoving energy at fixed Killing energy is gravitational blueshift; at the inner horizon it is the seed of mass inflation. (An earlier version of the 3D code modified the equations so that $\sum E^{\text{local}}$ was conserved, which removed blueshift by construction; that has been replaced.)

### 4.4 The local saturation rule and what it actually does
**Rule (as implemented).** After each Cayley step, any voxel whose comoving energy exceeds $E_{\text{voxel},\max}$ has its momentum scaled down to the ceiling; the removed comoving energy is added to a static reservoir $E_{\text{core}}[i]$ that counts toward the enclosed Misner–Sharp mass. The reservoir sits at fixed cells inside $r_+$; by point 3 of Section 4.1 this is only consistent if it is read as vacuum-like matter with $p_r = -\rho c^2$ (whose stress tensor singles out no velocity in the $t$–$r$ plane). The rule is non-symplectic and non-unitary.

**3D lattice ($25^3$, periodic, $r_s \approx 3.2$ cells, $R_{\text{core}} \approx 1.5$ cells, $r_- \approx 1$ cell, run length $1.65\,t_P$; `simulate_3d_18pt_freezing_core.py`).** The Killing-energy budget $H(t) = H(0) + \Delta_{\text{bg}} - \Delta_{\text{damp}}$ closes to $\sim10^{-15}$ in every run, so all energy flows are accounted for:

| Run | Peak comoving energy / ceiling | Voxels that hit the ceiling | $\Delta_{\text{bg}}/H(0)$ | $\Delta_{\text{damp}}/H(0)$ |
| :--- | :--- | :--- | :--- | :--- |
| Flat space ($J = 0$) | 0.956 | – | 0 | 0 |
| Fixed Hayward background | 0.956 | 0 | 0 | 0 |
| $J$ recomputed every step from the cells' energy (uncapped) | **1.647** | – | +0.052 | 0 |
| Same, capped (18-pt) | 1.000 | 6 ($27 \to 33$) | +0.052 | 0.064 |

On a fixed Hayward background the shell never reaches the ceiling in this run. The pile-up appears only when $J$ is recomputed each step from the cells' own energy — an ad-hoc self-gravity update that is *not* Einstein's equations and that injects about 5% Killing energy. The 3D runs are therefore a qualitative demonstration of the numerics, not evidence about black-hole interiors. (Shell faceting on $R = 3\ell$: 3.78% with 6 points, 0.345% with 18, 0.545% with 27.)

**Clean 1D inner-horizon test (`pg_wave_1d.py`, Module 4 Test 13).** Planar PG wave on the Hayward profile ($r_s = 12\ell$, $r_- = 1.047\ell$, $r_+ = 11.92\ell$, $\kappa_- = 0.83\,c^2/\ell$), outer boundary far outside $r_+$ in an absorbing sponge, flow tapered to zero there. A packet launched between $r_-$ and $r_+$ moving outward relative to the river is carried to $r_-$:

| Lattice spacing $h/\ell$ | Uncapped: peak comoving energy near $r_-$ / $E_0$ | Capped at 2× initial peak density: energy moved into reservoir / $E_0$ |
| :--- | :--- | :--- |
| 0.04 | 6.08 | 4.83 |
| 0.02 | 12.40 | 6.86 |
| 0.01 | 25.50 | 7.86 |

The uncapped blueshift scales as $h^{-1.03}$: it is limited only by the grid and diverges in the continuum limit. The cap keeps the field below the ceiling (by construction), but the energy it removes — 95–98% of it deposited within $\ell$ of $r_-$ — is several times the packet's own initial energy and keeps growing as $h \to 0$. Because the reservoir is part of the Misner–Sharp mass, **the cap relabels the blueshifted energy as mass concentrated at the inner horizon instead of removing it.** Whether that backreacts into mass inflation can only be settled with dynamical geometry, which this model does not have; the original claim that saturation resolves mass inflation is therefore **not supported**.

> [!NOTE]
> **Correction of an earlier diagnosis.** A previous version of Test 13 truncated the domain *inside* the trapped region with $J \neq 0$ at the boundary. That closure admits boundary-localised growing modes whose growth rate doubles each time $h$ is halved (2.1 → 4.2 per unit time at $h = 0.04 \to 0.02$), which produced spurious growth by factors up to $10^{20}$. These are numerical artifacts, not physics; tapering $J$ to zero in an outer sponge removes them (growth rate $< 0.01$).

### 4.5 Singularity theorems, the core entropy bound, and three possible completions
Penrose's (1965) theorem assumes the NEC, a non-compact Cauchy surface and global hyperbolicity. The Hayward core satisfies the NEC ($\rho c^2 + p_r = 0$, $\rho c^2 + p_\theta = 3\rho_{\max}c^2R_{\text{core}}^6r^3/(r^3 + R_{\text{core}}^3)^3 \ge 0$), so SEC violation is not the escape route. With trapped surfaces present, any nonsingular completion must give up one of the remaining assumptions. Combined with two further constraints — a cold remnant ($T_H \to 0$) requires an inner horizon that merges with $r_+$, and the core's Bekenstein bound $S_{\text{core}}/k_B \le \pi r_s/\ell_P$ is $r_s/\ell_P$ times too small to store $S_{\text{BH}}$ — the claims "no inner horizon", "cold remnant" and "information stored in the core" cannot all hold. Three self-consistent completions remain:

| Completion | How Penrose is evaded | What it keeps | What it loses | Existing work |
| :--- | :--- | :--- | :--- | :--- |
| **V1** Spacelike limiting-curvature transition at $\approx R_{\text{core}}$ into a de Sitter phase | Topology change (baby universe) | Nonsingular, $\rho \le \rho_{\max}$, $R_{\text{core}} \propto M^{1/3}$, no inner horizon | Cold remnant; unitarity as seen from outside; static "frozen core" | Frolov–Markov–Mukhanov 1989/90; stability: Balbinot–Poisson 1990 |
| **V2** Bounce at $\rho_{\max}$ into a white-hole / shock phase | Effective NEC violation during the bounce | Nonsingular, $\rho_{\max}$, $M^{1/3}$ scale, PG picture, possible unitarity, observable signals | Permanent black hole; frozen core; cold remnant | Haggard–Rovelli 2015; Kelly–Santacruz–Wilson-Ewing 2020; Husain et al. 2022 |
| **V3** Keep the inner horizon (ideally with $\kappa_- = 0$) | Loss of global hyperbolicity at $r_-$ | Static regular core; cold Hayward remnant | Exponential mass inflation unless $\kappa_- = 0$; even then a semiclassical instability remains | Hayward 2006; Carballo-Rubio et al. 2022; McMaken 2023 |

Module 8, Test 25 evaluates the defining equations of V1 and V2 with the bounce radius as an *input*. Section 4.6 goes further with a dynamical collapse on a falling lattice, which shows where the rule has to act and what each way of enforcing it produces.

### 4.6 Dynamical collapse on a falling lattice: where the rule has to act
To make the grid carry the geometry rather than sit on a prescribed metric, `collapse_falling_lattice.py` (Module 9) evolves gravitational collapse itself. The radial lattice nodes are free-falling shells; cell $k$ between nodes $k$ and $k+1$ carries a fixed Misner–Sharp mass $\Delta m_k$ (zero for empty cells). For matter falling from rest at infinity, Einstein's equations reduce exactly to the Lemaître–Tolman–Bondi solution in Painlevé–Gullstrand form, $ds^2 = -d\tau^2 + (dR - U\,d\tau)^2 + R^2 d\Omega^2$ with $U = -\sqrt{2m/R}$. The spatial slices are exactly flat, each node moves with the river velocity, which is now **determined by the enclosed mass** instead of being prescribed, and cell $k$ has proper volume $\frac{4\pi}{3}(R_{k+1}^3 - R_k^3)$. Units $G = c = \ell = 1$, $M = 100$ ($r_s = 200\ell$, $R_{\text{core}} = 5.85\ell$), 300 cells (240 matter, 60 empty), homogeneous and centrally concentrated profiles.

1. **GR on the lattice is exact.** Numerically integrated node trajectories match the exact solution to $3\times10^{-6}$ (Test 26). In both profiles the first cell reaches $\rho_{\max}$ only **after** a trapped surface has formed (homogeneous: $\tau = 376.8$ vs $243.8$; concentrated: $238.3$ vs $199.0$). This is general: a region at $\rho_{\max}$ of radius $R$ has $2m/R = (R/\ell)^2$, so any saturated region larger than $\ell$ is already trapped.
2. **Pressure cannot enforce the cap.** In a collapsing region obeying Einstein's equations with $\rho + p \ge 0$, $\dot\rho = -3H(\rho + p) \ge 0$ while $H < 0$, and $H^2 = 8\pi\rho/3$ never vanishes for a marginally bound region. A maximally stiff causal pressure ($c_s = c$ above $0.5\rho_{\max}$) still lets the density pass $10^6\rho_{\max}$ in finite time (Test 27). Holding $\rho$ at the cap requires $\rho + p \to 0$, i.e. conversion into vacuum energy ($p = -\rho$). That region then contracts exponentially at fixed density (flat case) or bounces at $a = \sqrt{k}\,\ell$ (closed case): this is the V1 route.
3. **A local cell-energy cap cannot remove the singularity.** An empty cell just outside the matter has zero energy density, so the rule never fires there. Its enclosed mass is fixed ($\dot m = -4\pi R^2 p\,U = 0$ where $p = 0$), so it follows Schwarzschild free fall whatever the matter inside does. It reaches $R = 10^{-3}\ell$ at exactly the Schwarzschild time ($\tau = 381.85$), where $K = 48M^2/R^6$ is $2\times10^{22}$ times the de Sitter bound (Test 28). The curvature exceeds $24/\ell^4$ as soon as $R < 2^{-1/6}R_{\text{core}}$. **The singularity simply moves into the empty cells.**
4. **A curvature cap does work.** In vacuum the mean enclosed density $\bar\rho = 3m/4\pi R^3$ is a curvature invariant ($K = 12(8\pi\bar\rho/3)^2$). Capping it with the effective-LQC form $\dot R^2 = \frac{2m}{R}(1 - \bar\rho/\rho_{\max})$, whose exact solution is $R^3 = R_b^3 + \frac92 m(\tau - \tau_b)^2$, makes **every** shell, empty ones included, bounce at exactly $R_b = (3m/4\pi\rho_{\max})^{1/3} = R_{\text{core}}(m)$ (error $10^{-8}$; Test 29). The bounce radius of the outermost shell is the manuscript's $R_{\text{core}}(M) = (r_s\ell^2)^{1/3}$. The shells then cross 2–3 time units after the first bounce: in the homogeneous case at the matter surface ($R = 11.6\ell$, inside the trapped region), in the concentrated case near the centre ($R = 1.2\ell$). There the decoupled-shell description ends and a shock forms, as found by Husain, Kelly, Santacruz and Wilson-Ewing. For non-uniform profiles the local cell density reaches $2\rho_{\max}$ just before crossing, so even this mechanism does not cap the local energy density everywhere.

**Answer to "which completion does the rule select?" (partial).** Taken literally, as a cap on each cell's energy, the rule does not prevent the singularity. A version that caps curvature (mean enclosed density) does, and then the outcome depends on how the cap acts. A repulsive (effective-LQC) enforcement gives a bounce (V2) at exactly $R_{\text{core}}(m)$, followed by shell crossing and a shock. Conversion into vacuum energy gives a de Sitter phase (V1), which for collapse also requires the transition to occur in the empty region, as in Frolov–Markov–Mukhanov. The lattice postulate itself does not choose between them. None of the mechanisms tested produces a static core with an inner horizon (V3).

### 4.7 Quantum aspects
Promoting $(\Phi_i, \Pi_i)$ to operators gives linear Heisenberg evolution that preserves $[\hat\Phi_i, \hat\Pi_j] = i\hbar\delta_{ij}$ (drift $3.8\times10^{-14}$ over 200 steps) and satisfies Ehrenfest's theorem trivially, because the system is linear. The lattice ground state has a finite entanglement entropy across a cut ($S_{\text{ent}} = 1.11$ for the 60-cell test chain). None of this extends to the saturation step, which is non-unitary, and Section 4.5 shows that the core cannot hold the black hole's information. Gravitationally induced entanglement (Bose–Marletto–Vedral) is predicted by any quantised gravitational interaction and is not a prediction specific to this model, which in any case contains no interaction between two masses.

---

## 5. Verification Suite and Claims Audit

### 5.1 How to read the tests
`python verify_core_finite_bar_gr.py` runs 10 core benchmarks and 9 modules (29 tests). A "passed" flag means a **numerical or algebraic check** succeeded (conservation, symplecticity, stability, finiteness, closed-form identities). Physics conclusions are reported as measured values and summarised in the claims audit printed at the end of the run.

| Module | Tests | What is checked |
| :--- | :--- | :--- |
| 1 Exterior GR | 1–3 | 1PN light-bending coefficient 2 in both gauges; dormant cells stay exactly static; ray-traced deflection $4GM/c^2b$ (4.00000016 after 2PN+3PN subtraction) |
| 2 Core | 4–5 | $p_r = -\rho_{\max}c^2$, SEC combination $-2\rho_{\max}c^2$, $p_\theta$ profile; $K(0)\ell^4 = 24$ |
| 3 Stencils | 6–8 | Moments; sixth-order directional ratios $1 : \frac32 : \frac{11}{9}$ (18-pt) vs $1 : \frac32 : \frac53$ (27-pt); 3D shell faceting; GRB 090510 tree-level delay |
| 4 Horizon & saturation | 9–13 | Horizon ordering $r_- < R_{\text{core}} < r_+$; Cayley conservation and inward trapping; 3D energy-budget closure; trapped region and spacelike $r = $ const surfaces; inner-horizon blueshift vs $h$ and reservoir deposit |
| 5 Non-spherical & spin | 14–15 | Exact discrete Killing-flux identity; Cayley conservation on an ad-hoc rotating flow (not Kerr) |
| 6 Quantum & thermodynamics | 16–17 | Linear commutator preservation; Hayward $\kappa(r_+)$, peak $T_H$ at $r_+ = 3\ell$, $T_H(\sqrt3\ell) = 0$, first law with $S_{\text{reg}}$ defined by $\int c^2dM/T_H$ |
| 7 Strong field | 18–21 | Raindrop proper time; tidal terms ($-\frac{5}{8\ell^2}$ at $R_{\text{core}}$, $+\frac{1}{\ell^2}$ stretching at $r = 0$); junction jumps; $\ell = 2$ WKB ringdown (shift $1.9\times10^{-9}$ at $r_s/\ell = 10^4$); zero echo from any reflector inside $r_+$ |
| 8 Extensions | 22–25 | Off-centre 3D packet with budget closure; two-component scalar toy (not spin-2); regular SdS horizons and Brillouin-zone group velocity; analytic V1/V2 comparison and entropy bound |
| 9 Dynamical collapse | 26–29 | Falling-lattice GR collapse vs exact LTB; pressure vs vacuum conversion vs LQC cap; empty cells reach $K \to \infty$ under a local cap; curvature cap bounces every shell at $R_{\text{core}}(m)$, then shell crossing |

### 5.2 Claims audit (as printed by the master runner)

| Claim | Verdict | Reason |
| :--- | :--- | :--- |
| Matches GR outside the horizon | Supported | By construction (metric is an input; corrections $\mathcal O(\ell^2/r^2)$) |
| Energy-conserving, symplectic black-hole wave evolution | Supported | Fixed background; inward trapping verified |
| Tree-level dispersion passes Fermi-LAT | Supported | Loop-induced Lorentz violation remains open |
| 27-point stencil removes the sixth-order body-diagonal error | Not supported | Only one monomial vanishes; 18-pt is more isotropic at sixth order |
| Bounded curvature, $R_{\text{core}} = (r_s\ell^2)^{1/3}$ | Supported | Properties of the Hayward metric |
| Inner Cauchy horizon is excised | Not supported | $r_-$ remains in the geometry |
| Reflected waves pinned on a static core surface | Not supported | $r = $ const is spacelike inside $r_+$ |
| Local saturation resolves mass inflation | Not supported | Cap converts blueshift into reservoir mass at $r_-$; backreaction not modelled |
| Singularity theorems evaded by SEC violation | Not supported | Penrose uses the NEC, which holds |
| Cold $T_H = 0$ remnant | Supported (conditional) | Requires the inner horizon |
| Quantum unitarity / information preserved | Not supported | Saturation is non-unitary; core entropy bound too small |
| Spin-2 sector | Not supported | Test 23 is a scalar toy |
| No post-merger echoes | Supported (generic) | True of any black hole with a horizon |
| A local cell-energy cap prevents the singularity | Not supported | Cap reached only inside a trapped region; pressure cannot enforce it; empty cells reach $K \to \infty$ |
| A curvature (mean-density) cap gives a bounce at $R_{\text{core}}$ | Supported (LQC-type dynamics) | Every shell bounces at $R_{\text{core}}(m)$; shells then cross (shock) |
| Lattice rule selects V1, V2 or V3 | Partly answered | Must act on curvature; repulsion gives V2, vacuum conversion gives V1; no mechanism tested gives V3 |

### 5.3 Predictions
At present the model makes **no distinctive, testable prediction**. The absence of echoes is generic to horizons; the Planck-suppressed dispersion is far below any foreseeable sensitivity; the cold remnant is Hayward's and requires the inner horizon; analogue (BEC / optical-lattice) experiments would test the analogue system rather than Planck-scale physics. Section 4.6 shows that a curvature cap enforced by repulsion produces a bounce (V2). If that bounce is followed through shell crossing and the matter re-emerges, it would supply observable signatures, as studied for Planck stars and black-to-white-hole transitions; this has not been computed here. If instead the cap acts by phase change (V1), Planck-mass end states would differ from Hayward remnants. These are the directions in which predictions could arise.

---

## 6. Limitations and References

### 6.1 Known limitations
1. **Limited dynamical geometry.** In the wave sector the metric is prescribed, and the 3D "self-gravity" update is not Einstein's equations. The falling-lattice collapse code (Section 4.6) does solve Einstein's equations, but only for spherically symmetric, marginally bound, pressureless matter, and it stops at shell crossing (no shock capturing).
2. **Test field only.** A single scalar field; the two-component "spin-2" test is a toy with an ad-hoc coupling. Genuine gravitational perturbations need Regge–Wheeler/Zerilli or Teukolsky equations on the regular background.
3. **Resolution.** The 3D runs resolve $r_s$ with ~3 cells and do not resolve $r_-$; they last $1.65\,t_P$ on a periodic box.
4. **Rotation.** The rotating flow is ad hoc. Kerr cannot be written in the flat-slice river form used here; it does not even admit axisymmetric conformally flat slices (Garat & Price 2000).
5. **Lorentz invariance.** A fundamental cubic lattice faces the Collins–Perez–Sudarsky–Urrutia–Vucetich loop problem; only tree-level dispersion is controlled here.
6. **Saturation rule.** Non-symplectic, non-unitary, and its physical content (what the reservoir is) is only consistent if the reservoir is vacuum-like matter.
7. **Planck-scale tidal forces.** Tidal terms are bounded but of order $c^2/\ell_P^2 \sim 10^{86}\,\text{s}^{-2}$; "bounded" does not mean survivable.
8. **1D inner-horizon test.** Test 13 uses a planar (1+1) wave equation on the Hayward river profile, without the $r^2$ factors and angular-momentum potential of a true spherical wave. It shows the mechanism, not precise numbers.
9. **Choices in the postulates.** The factor $\frac12$ in $E_{\max}$, the identification $\ell = \ell_P$, and the per-cell use of the Bekenstein bound are choices or heuristics, not derived results.
10. **Collapse code scope.** Module 9 treats only marginally bound dust (falling from rest at infinity), stops at shell crossing, and applies the LQC-type equation shell by shell (the decoupled-shell approximation).

### 6.2 Foundational Literature & References
1. **3D Spherical Shell Integration, Buchdahl Bypass, & Regular Black Hole Cores ($\frac{r^3}{r} = r^2$, $\tilde{r} = \sqrt{r^2 + R_{\text{core}}^2}$):**
   - K. Schwarzschild, *"Über das Gravitationsfeld einer Kugel aus inkompressibler Flüssigkeit"* (Interior Sphere Solution), *Sitzungsber. Preuss. Akad. Wiss.* **1916**, 424 (1916); G. D. Birkhoff, *Relativity and Modern Physics*, Harvard Univ. Press (1923); H. A. Buchdahl, *"General relativistic fluid spheres"*, *Physical Review* **116**, 1027 (1959).
   - A. D. Sakharov, *"Initial stage of an expanding universe and appearance of a nonuniform distribution of matter"* ($p = -\rho c^2$ superdense core equation of state), *Soviet Physics JETP* **22**, 241 (1966).
   - J. M. Bardeen, *"Non-singular general-relativistic gravitational collapse"*, in *Proceedings of the International Conference GR5*, Tbilisi, USSR, p. 174 (1968).
   - S. A. Hayward, *"Formation and evaporation of nonsingular black holes"*, *Physical Review Letters* **96**, 031103 (2006), `https://arxiv.org/abs/gr-qc/0506126`.
   - A. Simpson and M. Visser, *"Black-bounce to traversable wormhole"*, *Journal of Cosmology and Astroparticle Physics (JCAP)* **1902**, 042 (2019), `https://arxiv.org/abs/1811.07948`.
2. **Horizon Expansion, Inner-Horizon "Mass Inflation", & Frozen-Core Surface Accretion ($R_{\text{core}} \propto M^{1/3}$):**
   - P. C. Vaidya, *"The gravitational field of a radiating star"* (Horizon expansion as wave energy enters), *Proceedings of the Indian Academy of Sciences A* **33**, 264 (1951).
   - E. Poisson and W. Israel, *"Internal structure of black holes"*, *Physical Review D* **41**, 1796 (1990).
   - R. Carballo-Rubio, F. Di Filippo, S. Liberati, C. Pacilio, and M. Visser, *"On the viability of regular black holes"*, *Journal of Cosmology and Astroparticle Physics (JCAP)* **1807**, 023 (2018), `https://arxiv.org/abs/1805.02675`; and R. Carballo-Rubio, F. Di Filippo, S. Liberati, and M. Visser, *"Mass Inflation without Cauchy Horizons"*, *Physical Review Letters* **133**, 181402 (2024), `https://arxiv.org/abs/2402.14913`.
   - I. Dymnikova, *"Vacuum nonsingular black hole"*, *General Relativity and Gravitation* **24**, 235 (1992).
   - P. O. Mazur and E. Mottola, *"Gravitational vacuum condensate stars"* (Gravastars), *Proceedings of the National Academy of Sciences (PNAS)* **101**, 9545 (2004), `https://arxiv.org/abs/gr-qc/0109035`.
   - C. Rovelli and F. Vidotto, *"Planck stars"* ($R_{\text{core}} \propto M^{1/3}$ saturated core), *International Journal of Modern Physics D* **23**, 1442026 (2014), `https://arxiv.org/abs/1401.6562`.
   - J. Stefan, *"Über die Theorie der Eisbildung, insbesondere über die Eisbildung im Polarmeere"* (Moving phase-boundary surface freezing law), *Sitzungsber. Kaiserl. Akad. Wiss. Wien* **98**, 965 (1889); W. Israel, *"Singular hypersurfaces and thin shells in general relativity"*, *Nuovo Cimento B* **44**, 1 (1966).
3. **18-Point & 27-Point Isotropic 3D Grid Stencils & Fermi-LAT Observational Lorentz-Invariance Bound:**
   - A. Shinozaki and Y. Oono, *"Spinodal decomposition in 3-space"* (3D isotropic Laplacian stencil), *Physical Review E* **48**, 2622 (1993).
   - W. F. Spotz and G. F. Carey, *"A high-order compact formulation for the 3D Poisson equation"* (27-point $\frac{7}{15}, \frac{1}{10}, \frac{1}{30}$ stencil), *Numerical Methods for Partial Differential Equations* **12**, 235 (1996).
   - M. Patra and M. Karttunen, *"Stencils with isotropic discretization error for differential operators"*, *Numerical Methods for Partial Differential Equations* **22**, 936 (2006).
   - A. A. Abdo et al. (Fermi-LAT & GBM Collaborations), *"A limit on the variation of the speed of light arising from quantum gravity effects"* (GRB 090510), *Nature* **462**, 331 (2009), `https://arxiv.org/abs/0908.1832`; J. Collins, A. Perez, D. Sudarsky, L. Urrutia, and H. Vucetich, *"Lorentz invariance and quantum gravity: an additional fine-tuning problem?"*, *Physical Review Letters* **93**, 191301 (2004), `https://arxiv.org/abs/gr-qc/0403053`.
4. **The "River Model" of Black Holes, Kerr–Doran Whirlpool Flow, Horizon-Crossing Lattice Waves, & Laboratory Sonic Black Holes:**
   - A. J. S. Hamilton and J. P. Lisle, *"The river model of black holes"*, *American Journal of Physics* **76**, 519 (2008), `https://arxiv.org/abs/gr-qc/0411060`; C. Doran, *"A new form of the Kerr solution"*, *Physical Review D* **61**, 067503 (2000), `https://arxiv.org/abs/gr-qc/9910099`.
   - W. G. Unruh, *"Sonic analogue of black holes and the effects of high frequencies on black hole evaporation"*, *Physical Review D* **51**, 2827 (1995), `https://arxiv.org/abs/hep-th/9409082`.
   - S. Corley and T. Jacobson, *"Hawking spectrum and high frequency dispersion"*, *Physical Review D* **54**, 1568 (1996), `https://arxiv.org/abs/hep-th/9601073`.
   - J. Steinhauer, *"Observation of quantum Hawking radiation and its entanglement in an analogue black hole"*, *Nature Physics* **12**, 959 (2016), `https://arxiv.org/abs/1510.00621`; J. R. Muñoz de Nova, K. Golubkov, V. I. Kolobov, and J. Steinhauer, *"Observation of thermal Hawking radiation and its temperature in an analogue black hole"*, *Nature* **569**, 688 (2019), `https://arxiv.org/abs/1809.00913`.
5. **Quantum Operator Evolution, Superposition of Spacetime Branches, & Horizon Lattice Entanglement:**
   - P. Ehrenfest, *"Bemerkung über die angenäherte Gültigkeit der klassischen Mechanik innerhalb der Quantenmechanik"* (Ehrenfest's Theorem), *Zeitschrift für Physik* **45**, 455 (1927).
   - R. P. Feynman, *"The role of gravitation in physics"*, in *Proceedings of the 1957 Chapel Hill Conference* (1957); S. Bose et al., *"Spin entanglement witness for quantum gravity"*, *Physical Review Letters* **119**, 240401 (2017), `https://arxiv.org/abs/1707.06050`; C. Marletto and V. Vedral, *"Gravitationally induced entanglement between two massive particles"*, *Physical Review Letters* **119**, 240402 (2017), `https://arxiv.org/abs/1707.06036`.
   - M. Christodoulou and C. Rovelli, *"On the possibility of laboratory evidence for quantum superposition of geometries"*, *Physics Letters B* **792**, 64 (2019), `https://arxiv.org/abs/1808.05842`.
   - L. Bombelli, R. K. Koul, J. Lee, and R. D. Sorkin, *"Quantum source of entropy for black holes"*, *Physical Review D* **34**, 373 (1986); M. Srednicki, *"Entropy and area"*, *Physical Review Letters* **71**, 666 (1993), `https://arxiv.org/abs/hep-th/9303048`.
6. **Noether's Theorems, Transpose Energy Conservation ($\frac{dH}{dt} = 0$), & Discrete Spacetime Context:**
   - M. Bañados and I. Reyes, *"A short review on Noether's theorems, gauge symmetries and boundary terms"*, *International Journal of Modern Physics D* **25**, 1630021 (2016), `https://arxiv.org/pdf/1601.03616`.
   - M. Svärd and J. Nordström, *"Review of summation-by-parts schemes for initial–boundary-value problems"* (Transpose/Skew-Symmetric Energy Conservation on Grids), *Journal of Computational Physics* **268**, 17 (2014).
   - L. Bombelli, J. Lee, D. Meyer, and R. D. Sorkin, *"Space-time as a causal set"*, *Physical Review Letters* **59**, 521 (1987); R. Loll, *"Discrete approaches to quantum gravity in four dimensions"*, *Living Reviews in Relativity* **1**, 13 (1998), `https://arxiv.org/abs/gr-qc/9805049`.
7. **Limiting curvature, bounces, inner-horizon (in)stability, singularity theorems, and entropy bounds (added):**
   - R. Penrose, *"Gravitational collapse and space-time singularities"*, *Physical Review Letters* **14**, 57 (1965); S. W. Hawking and R. Penrose, *"The singularities of gravitational collapse and cosmology"*, *Proc. R. Soc. Lond. A* **314**, 529 (1970).
   - V. P. Frolov, M. A. Markov, and V. F. Mukhanov, *"Through a black hole into a new universe?"*, *Physics Letters B* **216**, 272 (1989); *Physical Review D* **41**, 383 (1990); R. Balbinot and E. Poisson, *"Stability of the Schwarzschild–de Sitter model"*, *Physical Review D* **41**, 395 (1990).
   - H. M. Haggard and C. Rovelli, *"Quantum-gravity effects outside the horizon spark black to white hole tunneling"*, *Physical Review D* **92**, 104020 (2015), `https://arxiv.org/abs/1407.0989`.
   - J. G. Kelly, R. Santacruz, and E. Wilson-Ewing, *"Black hole collapse and bounce in effective loop quantum gravity"*, *Classical and Quantum Gravity* **38**, 04LT01 (2021), `https://arxiv.org/abs/2006.09325`; V. Husain, J. G. Kelly, R. Santacruz, and E. Wilson-Ewing, *"Quantum gravity of dust collapse: shock waves from black holes"*, *Physical Review Letters* **128**, 121301 (2022), `https://arxiv.org/abs/2109.08667`.
   - R. Carballo-Rubio, F. Di Filippo, S. Liberati, C. Pacilio, and M. Visser, *"Regular black holes without mass inflation instability"*, *JHEP* **09** (2022) 118, `https://arxiv.org/abs/2205.13556`; T. McMaken, *"Semiclassical instability of inner-extremal regular black holes"*, *Physical Review D* **107**, 125023 (2023).
   - J. D. Bekenstein, *"Universal upper bound on the entropy-to-energy ratio for bounded systems"*, *Physical Review D* **23**, 287 (1981).
   - A. Garat and R. H. Price, *"Nonexistence of conformally flat slices of the Kerr spacetime"*, *Physical Review D* **61**, 124011 (2000).
   - G. Lemaître (1933); R. C. Tolman (1934); H. Bondi (1947) — the spherically symmetric dust solution used in Section 4.6.
