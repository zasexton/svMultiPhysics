# Rigorous audit of free-surface boundary conditions, wall wetting, and unfitted level-set methods

- **Audit date:** 2026-07-20
- **Repository:** `svMultiPhysics`
- **Branch:** `issue-449-modern-mesh-core`
- **Reviewed HEAD:** `7aaadf5ad1dc1b56e25323ca994df03635ad4c5b`
- **Mode:** read-only source, input, prior-result, and literature review; no solver source was modified, no build or new physical simulation was run, and no commit was made.

## Executive verdict

The free-surface implementation is **not yet a fully robust or quantitatively qualified method** for incompressible Navier--Stokes flow with an unfitted level set, particularly for moving contact lines and splashing.

The narrow supported sharp-P1 path has several important pieces implemented with the correct continuum signs:

- the liquid-outward interface normal is used consistently;
- the bulk pressure virtual-work sign is correct;
- the constant-surface-tension `SurfaceStress` weak form has the correct external-pressure and surface-divergence signs;
- the surface conormal is not double counted in the supported dynamic-contact path;
- the through-liquid contact-angle convention is correct;
- wall-normal impermeability and side-wall tangential frames are oriented correctly; and
- the Ren--E line-friction and Navier wall-slip terms are dissipative with the intended signs.

No new sign reversal, normal reversal, or duplicate capillary-conormal defect was found in the supported
`UnfittedLevelSet + CutVolume + LinearCorner + SurfaceStress` configuration. The observed side-wall errors are therefore not explained by a simple rotation or wall-normal bug.

There are, however, multiple high-severity method and infrastructure problems:

1. The option called free-surface velocity extension adds a dry-domain Laplacian of the **same physical velocity unknown** directly to momentum. It is not passive postprocessing and not a one-way extension. It changes wet-supported cut-cell rows, adds an artificial dry-side interface flux, couples disconnected liquid components through the fictitious dry domain, and makes exterior wall data feed back into liquid momentum.
2. The separate algebraic velocity-extension map used by level-set transport has no conditioning, coefficient, row-norm, partition-of-unity, or amplification bounds. Existing D38 evidence shows amplification by orders of magnitude.
3. Generic LinearCorner curved interfaces are not exactly balanced by the current pressure/surface-force pair. A current sessile Q1/Q1 artifact retains a 1.3328% residual after minimization over the entire admissible pressure space. The existing pressure-representability diagnostic deliberately applies no physical-distance gate.
4. Prescribed contact angle is imposed through an unscaled codimension-two penalty on the level-set residual. It has no mesh, time, trace, or dimensional scaling rule and is not paired with a contact-angle-compatible redistancing wall condition.
5. The dynamic contact model is mathematically plausible but fails its present quantitative evidence: three of four completed advancing/receding side-wall Ren--E speed gates fail, including one wrong-sign result. The wet-wall friction is also integrated with a regularized wall indicator instead of a sharp wetted-wall cut.
6. Level-set transport is continuous Galerkin SUPG/discontinuity-capturing transport, not locally conservative interface transport. The optional limiter is explicitly nonconservative and not FCT/AFC; production rejects a converged candidate if the limiter would alter it. Global volume correction can conceal transport loss while moving every interface and contact line.
7. General cut-position-independent mixed stability has not been established. Pressure-only ghost penalty, empirical scaling, aggregation, and sliver pruning do not by themselves prove uniform inf-sup stability or conditioning for the actual equal-order VMS/PSPG spaces.
8. The present physics is a one-phase liquid with prescribed external gas pressure. It cannot physically validate air cushioning, gas inertia/viscosity/compressibility, entrained or trapped air, aerodynamic sheet breakup, or gas-pressure-dependent dry-wall splash thresholds.
9. Surface geometry is refreshed and then frozen during a nonlinear solve. There is no complete shape derivative or demonstrated fully discrete capillary/wetting energy law.
10. Fitted-ALE tangential free-surface controls are parsed and validated but never consumed by production assembly or mesh motion.
11. Generic exterior traction, Robin, outflow, and weak-Nitsche forms use whole background boundary faces rather than the sharply wetted part of a cut face. Boundary work on the dry portion can enter shared cut-cell rows and the same-field dry extension.

The most defensible present characterization is:

> The code contains a narrowly supported, sign-consistent sharp-P1 free-surface/contact weak form, but the overall unfitted method is not well balanced, conservatively transported, uniformly cut stable, or quantitatively validated for moving wetting and splashing. Two distinct velocity-extension mechanisms each have unresolved feedback or stability defects.

## Review scope and evidence rules

This audit covers:

- incompressible Navier--Stokes volume residuals and stabilization;
- fitted and unfitted free-surface traction paths;
- `SurfaceStress`, legacy curvature traction, and external pressure;
- generated interface and contact-line quadrature;
- dynamic and prescribed contact-angle implementations;
- side-wall impermeability, wet-wall Navier slip, and contact-line friction;
- level-set transport, algebraic velocity extension, dry-domain PDE extension, limiting, reinitialization, and volume correction;
- curvature projection, cut-cell stabilization, aggregation, pruning, and pressure support;
- nonlinear geometry refresh and generalized-alpha state coupling;
- D18/D38, sessile-cap, moving-contact, capillary-wave, and wall-advection evidence already present in the repository; and
- finite-element, CutFEM, moving-contact-line, free-surface, dam-break, sloshing, and splash literature.

Finding labels used below are:

- **Confirmed defect:** source or existing numerical evidence directly demonstrates behavior inconsistent with the advertised or necessary contract.
- **Method deficiency:** the implementation performs what it was coded to perform, but that formulation lacks consistency, stability, conservation, or physical completeness needed for the target problem.
- **Qualification gap:** no failure is proved, but the required theorem, diagnostic, or converged validation evidence is absent.
- **Model limitation:** behavior lies outside the governing physics, even if the numerics were exact.

This report does not treat a passed unit test as proof of a continuum property. It also does not infer that a source-level risk is the unique cause of a particular failed run unless an ablation establishes that causal link.

### Exact source snapshot

The working tree was already heavily modified before this review. Consequently, the findings apply to the files as read, not merely to the named Git commit. Relevant working-tree SHA-256 values are:

| Component | SHA-256 |
|---|---|
| `IncompressibleNavierStokesVMSModule.cpp` | `71d418eec790fdcdd65de664f3fce78aa7d8f756556fc3420cbce7247c537850` |
| `NavierStokesRegister.cpp` | `f926918fcd1a2d3ce9c1507d6aa46fa618208151c98ce3b0ab681e1e8854e5a3` |
| `ApplicationDriver.cpp` | `716be9218e81c3b7c04f6bc8c2b8ce13a862f9f88cb02b7a65ee9cabf9721552` |
| `LevelSetTransport.cpp` | `4babe16f4862540fad78ac6880dc0324a8b1bc7d7311cc39a4c2613cfa91fc56` |
| `LevelSetVelocityExtensionConstraint.cpp` | `82d358a0eb467efef12f96528da7a179ea53be17a6ef1a5628e06cf268dad08a` |
| `LevelSetReinitialization.cpp` | `4fc96c9c45011e438391c600c9fd30c6b8c76073977c8aa697866e0f8a38030c` |
| `LevelSetVolume.cpp` | `ce0b086bb3ec5b4cc73d55d30400ee8b54b2a6dd0657e657ef480641830e66b1` |
| `LevelSetImplicitCutQuadratureBackend.cpp` | `66826216b22b292d9a920bd14d74fb07a28d4fd01d016c227f2a788fc3ab8c5f` |

The detailed earlier implementation and qualification history remains in
[free_surface_level_set_review_20260713.md](free_surface_level_set_review_20260713.md). This report re-audits the current mechanisms and uses those recorded artifacts only where explicitly identified.

## Mathematical contract

Let \(\Omega_l(t)\) be the liquid domain, \(\Gamma(t)\) its liquid--gas boundary, and \(n\) the unit normal pointing out of the liquid. With

\[
\sigma_l = -pI + 2\mu D(u), \qquad
D(u)=\tfrac12(\nabla u+\nabla u^T),
\]

constant surface tension \(\gamma\), prescribed external pressure \(p_{ext}\), and the convention that a convex liquid surface has positive curvature \(\kappa\), the intended traction is

\[
\sigma_l n = (-p_{ext}-\gamma\kappa)n.
\]

After integration by parts, a curvature-free surface-divergence form contributes

\[
+\int_\Gamma p_{ext}\,n\cdot v\,dS
+\gamma\int_\Gamma (I-n\otimes n):\nabla v\,dS
\]

to the residual when the volume pressure term is

\[
-\int_{\Omega_l}p\,\nabla\cdot v\,dV.
\]

For a wall with unit normal \(n_w\), and a contact angle measured through the liquid, equilibrium Young geometry is

\[
n\cdot n_w = -\cos\theta_e.
\]

The surface energy for a closed isothermal one-phase wetting problem can be written, up to constants, as

\[
\mathcal E =
\frac12\int_{\Omega_l}\rho |u|^2dV
+\gamma A_{lg}
-\gamma\cos\theta_e A_{sl}
+\int_{\Omega_l}\rho g z\,dV.
\]

For Navier slip length \(l_s>0\) and line friction \(\xi>0\), the expected dissipation contains

\[
2\mu\int_{\Omega_l}|D(u)|^2dV
+\frac{\mu}{l_s}\int_{\Gamma_{sl}}|u_t|^2dS
+\xi\int_C V_{CL}^2d\ell.
\]

The Ren--E-type constitutive relation used by this repository is

\[
\xi V_{CL}=\gamma(\cos\theta_e-\cos\theta_d),
\qquad \xi=1/M.
\]

A robust finite-element implementation must preserve the relationship between the surface virtual work, Young wall energy, wall slip, contact-line dissipation, and interface transport. Correct signs in each isolated term are necessary but not sufficient: discrete geometry, pressure space, time integration, maintenance operations, and cut stabilization must also be compatible.

## Prioritized findings

| ID | Severity | Classification | Finding | Immediate implication |
|---|---:|---|---|---|
| FSR-01 | High | Confirmed method defect | Dry-domain “velocity extension” is a Laplacian of the physical velocity in the same momentum system. | Artificial dry-side traction/stiffness feeds into liquid rows and can couple separate liquid bodies. |
| FSR-02 | High | Confirmed defect | Algebraic wet-extension map has no conditioning or amplification bounds; D38 maps amplify tiny velocities into very large extension values. | Level-set advection can be dominated by map artifacts even when its frozen-map Jacobian is exact. |
| FSR-03 | High | Confirmed method deficiency | Generic LinearCorner surface loads are not exactly representable by the pressure-gradient range; a current sessile case retains a 1.3328% best-space residual. | Static caps generate irreducible parasitic forcing; the current diagnostic has no acceptance threshold. |
| FSR-04 | High | Method deficiency | Prescribed contact angle is an unscaled codimension-two level-set penalty with no compatible reinitialization wall condition. | Mesh/time dependence and angle drift are expected; the penalty can conflict with momentum-side wall energy. |
| FSR-05 | High | Failed qualification | Three of four completed side-wall Ren--E speed gates fail, including one wrong sign. | The dynamic wetting law is not calibrated or mesh independent despite correct wall orientation. |
| FSR-06 | High | Method deficiency | Continuous-Galerkin level-set transport, limiter, and global shift are not locally conservative. | Thin sheets, rims, droplets, and contact-line volume can be created, lost, or globally redistributed. |
| FSR-07 | High | Qualification gap | Uniform inf-sup stability and cut-position-independent conditioning are not established for the actual equal-order cut formulation. | Tiny cuts and topology changes can produce pressure modes, ill conditioning, or feature deletion. |
| FSR-08 | High | Model limitation | Gas is only a prescribed exterior pressure, not a solved phase. | Full dry splash, air cushioning, entrainment, trapped gas, and aerodynamic breakup cannot be validated physically. |
| FSR-09 | High | Qualification gap | Refreshed-frozen geometry lacks the complete shape tangent and a demonstrated fully discrete energy law. | Nonlinear convergence and capillary energy exchange depend on outer refresh and time-step policy. |
| FSR-10 | High | Confirmed defect | Fitted-ALE tangential mesh policies are parsed but unused. | `Free`, `SmoothingOnly`, and `Prescribed` do not implement their advertised motion. |
| FSR-11 | Medium | Confirmed configuration defect | Per-free-surface Nitsche values overwrite module-global Nitsche policy. | Multiple fitted surfaces and unrelated weak velocity BCs are last-one-wins/order dependent. |
| FSR-12 | Medium | Confirmed configuration defect | Partial contact-line input can be silently ignored. | Typographical or incomplete wetting configurations can run without the requested physics. |
| FSR-13 | Medium | Generality defect | Generated physical phase volume uses one parent-cell scale instead of pointwise geometry mapping. | Distorted/nonaffine and high-order-cell volume diagnostics/correction can disagree with assembly measure. |
| FSR-14 | Medium | Validation gap | Cut-backend checks do not fully verify root accuracy, normal accuracy, volume-point containment, or phase side. | A backend defect can pass structural validation and contaminate residuals or diagnostics. |
| FSR-15 | Medium | Coupling risk | Scalar contact reconstruction ignores the supplied interface-domain object and rebuilds roots independently. | Contact and surface rules lack an enforced same-revision/same-parent geometry invariant. |
| FSR-16 | High | Resolved in qualified scope | Supported one-phase affine C0 P1/LinearCorner exterior weak and natural forms are clipped to the generated wetted-face domain; unsupported higher-order routes fail closed. | Closure is limited to the declared envelope; WP-7, Q1, higher-order support, and uniform cut conditioning remain open. |
| FSR-17 | Medium | Confirmed generality defect | Supplemental high-order curvature samples pair a value at one reference point with a separately averaged physical point. | Curvature least squares receives inconsistent `(x,phi)` data on nonaffine/isoparametric cells. |
| FSR-18 | Medium | Robustness defect | Colliding algebraic extension bands choose the smaller component ID and use cell-node clique adjacency. | Splash components can receive numbering-dependent extension partitions with broad, non-geometric graph communication. |

## FSR-01: the dry-domain PDE extension changes physical momentum

The routine `applyFreeSurfaceVelocityExtension` in
[`IncompressibleNavierStokesVMSModule.cpp`](../Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp)
adds, on the inactive/dry part of the mesh,

\[
\alpha\int_{\Omega_{dry}}\nabla u:\nabla v\,dV
\]

using the **same** velocity coefficients and test functions as the liquid momentum equation. The key implementation is at lines 4564--4620. Cut-cell weights are multiplied by

\[
\max(r_{dry},10^{-3}),
\]

while fully dry cells receive full strength \(\alpha\). The ordinary physical momentum residual is wet-volume-only at lines 6615--6626. Inactive velocity pins are disabled when this extension is enabled at lines 6000--6041.

This is not a passive or one-way extension. A continuous finite-element basis function supported by a cut cell spans both wet and dry subregions, so the dry term necessarily changes wet-supported matrix rows. In continuum terms it creates a dry harmonic field coupled to liquid velocity at the interface. Its interface transmission balance includes an artificial dry-side flux \(\alpha\partial_n u_{dry}\), which is an artificial traction on the physical problem.

### Scale analysis

The D18/D38 configurations use approximately

\[
\mu_l=0.001003\ \mathrm{Pa\,s}, \qquad \alpha=1.
\]

Therefore:

- a full dry cell has extension stiffness about \(997\mu_l\);
- the hard-coded cut floor gives \(\alpha_{eff}=10^{-3}\), approximately the liquid viscosity;
- a half-dry cut cell can receive a dry extension term hundreds of times the corresponding physical viscous coefficient; and
- a tiny wet sliver can receive nearly full-cell liquid-scale stiffness from the dry floor.

The coefficient also has the units of a viscosity-like diffusion coefficient but is configured as a dimensionless-looking value without a physical or mesh/time scaling contract.

### Consequences

1. Exterior dry-wall Dirichlet data can pull on the free-surface velocity through the dry harmonic problem.
2. Separate droplets or liquid regions can communicate through a connected fictitious dry domain.
3. Interface velocity can depend on the size and topology of the unused exterior mesh.
4. A dry-domain energy term is added to the nonlinear system, but it is not part of the physical liquid energy or a demonstrated consistent CutFEM stabilization.
5. The term can delay, accelerate, or damp side-wall run-up depending on the dry field and its boundary constraints.

This is a high-risk plausible contributor to wetting/run-up error. Source inspection alone does not prove that it dominates D18/D38: a previously documented Test02 toggle changed the front by only `+0.00130 m` at `t=0.2`. The required conclusion is narrower and firm: the option is inaccurately described if it is intended to be a no-feedback velocity extension.

### Required replacement or proof

A defensible implementation should use one of the following:

- a distinct auxiliary extension field solved after, or block-triangularly from, liquid velocity;
- a one-way normal/constant-along-normal extension in a narrow band;
- an aggregation/discrete-extension operator whose slave values cannot modify master liquid equations; or
- a mathematically consistent ghost penalty with the correct field, scaling, support, consistency result, and cut-independent stability proof.

Before replacement, isolate the effect with:

- \(\alpha\), floor, \(h\), \(\Delta t\), and \(\mu\) sweeps;
- wet-block residual and matrix differences with the option on/off;
- a manufactured wet solution while dry-domain size and boundary data vary;
- two disconnected liquid islands separated by dry cells;
- full no-slip versus normal-only dry-wall constraints; and
- explicit dry-extension energy and interface-flux telemetry.

## FSR-02: the algebraic transport extension is exact at a frozen map but the map is unbounded

This mechanism is distinct from FSR-01. The level-set transport infrastructure introduces an algebraic extension unknown \(E\) and constraints of the form

\[
E_i-\sum_jP_{ij}(\phi)u_j-\sum_kL_{ik}(\phi)E_k=0.
\]

[`LevelSetVelocityExtensionConstraint.cpp`](../Code/Source/solver/FE/LevelSet/LevelSetVelocityExtensionConstraint.cpp)
assembles the frozen-map residual and Jacobian consistently at lines 237--270. The fixed-map chain

\[
u\rightarrow E\rightarrow \phi
\]

is therefore algebraically sound for the supported P1 trace contract.

The problem is construction of \(P(\phi)\). [`ApplicationDriver.cpp`](../Code/Source/solver/Application/Core/ApplicationDriver.cpp)
uses a small dense tangential regression, accepts a solve based principally on pivot success, converts the result directly to graph weights, and applies it. There is no enforced:

- numerical rank or condition estimate;
- maximum coefficient or row-\(L^1\) norm;
- partition-of-unity/constant-reproduction check;
- negative-weight or convex-hull check;
- preview ratio \(\|E\|/\|u\|\);
- bound on map change between geometry revisions; or
- bounded nearest/normal/convex fallback.

Existing D38 artifacts show the result. Early extension norms are approximately

\[
135.759,\ 51.2615,\ 0.2225
\]

while physical velocity norms are only about `0.0404--0.0447`. Rebuilding the map at the same state changes the total residual to about `169.862`. Later extension values reach millions and the level-set residual reaches about `112433`.

On refresh, production replaces the extension constraint rows but retains the previous \(E\) coefficients as a warm start; it does not project them onto the newly built algebraic map in this branch. That explains part of the immediate post-refresh residual jump and is independently worth correcting for nonlinear robustness. Reprojecting \(E\) would not, however, cure an unstable map whose coefficients turn a `0.04` physical velocity into values of order `10^2--10^6`.

This is a confirmed numerical stability defect in a supported infrastructure path. Correct rollback and a correct frozen-map Jacobian cannot make an ill-conditioned refreshed map safe.

Required production guards are:

1. rank and condition estimation before accepting a regression;
2. coefficient and row-\(L^1\) limits;
3. exact or tolerance-based constant reproduction;
4. negative-weight/convexity policy;
5. preview amplification and map-change limits;
6. per-revision telemetry keyed to accepted time, state, and nonlinear phase; and
7. a bounded fallback that preserves wall-normal compatibility.

The multi-component policy also needs correction. When two extension bands collide, production silently selects the smaller connected-component label and records only a collision count. Existing tests intentionally encode that minimum-ID choice. Component numbering is not a physical criterion, so swapping IDs can change the extension partition. Moreover, band adjacency is built as a complete clique among all nodes of a cell rather than from mesh edges or geometric distance. This can transmit information across diagonals and create a wider/non-geometric extension graph. These behaviors are especially risky for colliding droplets, crown fingers, or nearby sheets and motivate FSR-18.

## FSR-16: resolved within the affine P1/LinearCorner envelope

Current disposition, recorded on 2026-08-26: FSR-16 is closed for the supported one-phase affine C0 P1/LinearCorner exterior-boundary route. Traction, Robin, outflow, pressure-flux, symmetric and unsymmetric Nitsche, wall-slip, coupled-outflow, and PSPG boundary work use the generated active boundary; a dry face contributes exactly zero, missing or ambiguous sharp domains are hard errors, and unsupported higher-order selection fails closed. The checksum-bound [V6 qualification record](qualification_logs/free_surface_wp3_sharp_boundary_v6_20260826_a73c77f4/record.md) and [summary](qualification_logs/free_surface_wp3_sharp_boundary_v6_20260826_a73c77f4/summary.json) are the acceptance evidence. This disposition does not close WP-7, Q1, higher-order support, or uniform cut conditioning.

The paragraphs below preserve the defect state reviewed on 2026-07-20 and the rationale for the completed work.

Cut-volume momentum is restricted to the liquid, but generic exterior-boundary operators are still assembled using ordinary full-face measures such as `.ds(marker)`. The generic boundary manager is invoked in
[`IncompressibleNavierStokesVMSModule.cpp`](../Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp), lines 6813--6832, and weak velocity Nitsche forms in
[`NavierStokesBCFactories.h`](../Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h), lines 358--419, use the uncut background face. Nitsche installation occurs around line 6864 of the NS module.

For a background boundary face that is partly liquid and partly dry, the physical boundary integral should be restricted to its wetted subset. The reviewed generic traction, Robin, outflow, and weak-Nitsche paths have no active-side factor and no generated wet `(d-1)` boundary quadrature. They can therefore apply stress, consistency, symmetry, or penalty work over the dry part. Because cut-cell basis functions are shared, and because FSR-01 solves the same velocity in the dry domain, this fictitious boundary work can feed into liquid momentum.

Strong wall constraints are a special case: applying them to the whole background wall can be interpreted as also specifying the dry extension field, although full no slip then prevents a physical sharp contact line from moving. The dynamic-contact Navier term is another special case; it uses the smoothed wet-wall indicator discussed above. Neither exception makes the generic weak/natural operators sharply correct.

Required tests are:

- a manufactured half-wet boundary-face integral with an analytic wetted measure;
- a dry-only boundary whose weak operator contributes exactly zero to wet rows;
- active-side Nitsche consistency/coercivity versus cut fraction;
- traction and Robin/outflow wet-fraction sweeps; and
- a coupled surface/wall energy balance at a moving contact line.

## FSR-03: pressure is representable, but capillary equilibrium is not well balanced

The current one-phase model must not be misdiagnosed as requiring a two-phase pressure jump enrichment in every case. Gas pressure is prescribed on the boundary and no gas-pressure field is solved. Constants belong to the liquid pressure space.

The factory creates a vector H1 velocity space and scalar H1 pressure space. Unless Taylor--Hood is explicitly selected, pressure order equals velocity order. The current sessile/free-surface generators use continuous equal-order Q1/Q1 on linear quads, with analogous P1/P1 behavior on linear tetrahedra. Constant pressure is retained on active cut support, and small-cut aggregation explicitly preserves constants. Constant pressure also nulls PSPG and pressure ghost-gradient terms.

The discrete residual is nevertheless not generally well balanced. It contains

\[
-\int_{\Omega_h}p_h\nabla\cdot v_h
+\int_{\Gamma_h}p_{ext}n_h\cdot v_h
+\gamma\int_{\Gamma_h}(I-n_h\otimes n_h):\nabla v_h
-\gamma\cos\theta_e\int_{\partial\Gamma_h}v_h\cdot m_h.
\]

Exact equilibrium requires \(\Gamma_h\) to be a stationary point of the **discrete** surface-plus-wall energy under the **discrete** volume constraint: a discrete constant-mean-curvature and Young-angle identity. Sampling a continuum circle or cap at Q1/P1 vertices and converting it to LinearCorner chords does not generally satisfy that identity.

The repository already forms the complete constrained pressure-gradient operator \(G\) and uses LSQR to minimize

\[
\|Gp+f_{surface}\|.
\]

In a current `theta90_n16` sessile artifact, the normal equations converge to relative normal residual `1.45e-11`, yet the primal relative residual remains `0.013328`, or **1.3328%**. Thus not even an arbitrary pressure in the full Q1 pressure space exactly spans the discrete surface-plus-wall load. Other recorded static caps retain roughly 1.2--4.4% best-space residuals.

This proves a method-level discrete compatibility deficiency. It is not evidence of a wrong pressure sign; the signs and wall-energy ownership are correct. A flat interface or specially constructed discrete constant-mean-curvature surface can still balance.

There is also a qualification defect. Telemetry explicitly reports that the representability-distance gate is not applied and not claimed. The runner checks finite LSQR behavior and normal-equation convergence, not whether the primal residual is physically small. A “green” diagnostic therefore does not qualify balanced force.

Required remedies should be evaluated rather than assumed:

- construct surface and wall forces as the exact variation of a discrete geometric energy with matching discrete volume;
- use a geometry/pressure pair with a provable equilibrium property;
- consider appropriate pressure enrichment for future two-phase jump problems;
- use a stabilized mean-curvature vector or parametric interface treatment with a compatible energy identity;
- gate the actual best-space residual, pressure error, force residual, and parasitic kinetic energy; and
- sweep mesh refinement, interface position, interface orientation, and contact angle.

## Free-surface traction audit

### Components found correct within the supported contract

1. **Bulk pressure sign.** The Galerkin term is `-p div(v)` in
   [`IncompressibleNavierStokesVMSModule.cpp`](../Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp), lines 6217--6228.
2. **External pressure and surface tension.** `SurfaceStress` adds \(p_{ext}n\cdot v\) and \(\gamma(I-nn):\nabla v\) at lines 5443--5494.
3. **Legacy traction consistency.** The explicit curvature path constructs the same declared traction sign at line 5504.
4. **Active-side orientation.** Generated outward normals are oriented consistently with the chosen active liquid side near line 1663.
5. **Zero-load behavior.** With zero external pressure and zero surface tension, the interface has natural zero traction and no artificial residual at lines 5399--5409.
6. **One conormal owner.** Under `SurfaceStress`, the surface-divergence term owns the capillary conormal. The contact contribution adds line friction and equilibrium Young wall energy, not a second dynamic conormal, at lines 4350--4360.

### Remaining limitations

- `SurfaceStress` rejects differentiated quadrature. Geometry is refreshed outside the solve and frozen during each nonlinear solve.
- There is no derivative of projector, normal, point, surface measure, contact geometry, or topology transition in the monolithic Jacobian.
- The outer process is therefore Picard/quasi-Newton with respect to shape even when the field Jacobian is otherwise exact.
- No complete fully discrete identity shows that capillary work equals change in surface-plus-wall energy at the generalized-alpha stage.
- Legacy raw-curvature traction is more sensitive to curvature recovery; the default `SurfaceStress` choice is preferable for constant \(\gamma\), but it does not by itself solve FSR-03.

## Side-wall wetting and contact-line audit

### Geometry and orientation are correct

The true vertical-sidewall case uses:

- wall `x=0`;
- outward liquid-to-solid wall normal `(-1,0,0)`;
- tangential contact-line motion in `y`; and
- strong zero normal-only `x` velocity.

The module rejects a dynamic moving contact line if the wall is given a tangential/full no-slip strong constraint, and it verifies mapped physical wall normals and interface/wall transversality. Rotation-equivalent bottom-wall and side-wall tests differ only at roundoff-to-small discretization scale. The side-wall frame, configured normal, contact tangent, and postprocessing orientation are therefore not the observed root problem.

### Dynamic line-law structure is sign consistent

The implemented gap uses

\[
\cos\theta_d=-n\cdot n_w,
\]

and the residual combines:

- positive line friction \(\xi V_{CL}\);
- equilibrium wall energy \(-\gamma\cos\theta_e\); and
- positive Navier wall dissipation \((\mu/l_s)u_t\cdot v_t\).

This is structurally consistent with the selected Ren--E convention and with a decreasing total energy. It is also invariant to a positive rescaling of \(\phi\) in the normal construction.

### Quantitative qualification fails

All four current advancing/receding vertical-wall cases complete 20/20 fixed steps to `t=0.02`, with zero rejected step and converged recorded solves. Only the `n=32` receding case passes the fixed direct constitutive-speed gate. Relative errors are:

| Resolution/regime | Relative speed error | Disposition |
|---|---:|---|
| `n=16`, advancing | `15.8991` | Fails; measured sign is wrong |
| `n=32`, advancing | `0.60010` | Fails |
| `n=16`, receding | `6.11871` | Fails |
| `n=32`, receding | `0.338382` | Passes the current gate only |

The same mismatch after rotating to the bottom wall proves that the main discrepancy is orientation independent.

### Likely contributors to the moving-contact mismatch

1. **Wet-wall integration is regularized.** Navier friction is multiplied by a compact C1 cubic wetted-wall indicator over a width based on \(h\) or a configured width, rather than integrated over the exact sharp wetted footprint. The construction has a dry tail and is homogeneous in \(\phi\), but it introduces width- and mesh-dependent dissipation and does not inherit the sharp-interface energy estimate without analysis.
2. **Contact observables are not stage-paired.** Operator-angle telemetry lacks accepted time, state revision, nonlinear phase, and generalized-alpha stage identity. It cannot be paired exactly with stage-based contact speed.
3. **Geometry is refreshed-frozen.** Dynamic angle, contact point, and velocity do not share a complete monolithic shape tangent.
4. **Extension feedback.** FSR-01 and FSR-02 can alter the velocity transported to the interface or the level-set geometry used to measure angle.
5. **Redistancing has no wetting wall condition.** Maintenance can change the near-wall normal and hence angle.

Required isolations are sharp-versus-smeared wetted-wall quadrature, width and slip-length refinement, stage-synchronized angle/speed telemetry, and a benchmark whose slip scale is resolved independently of mesh size.

### Prescribed contact angle is not a robust wall boundary condition

The prescribed-angle residual is essentially

\[
\int_C \beta_\theta
\left(n\cdot n_w+\cos\theta_e\right)\eta\,d\ell.
\]

It is implemented as a scalar level-set residual over a codimension-two contact rule. The penalty has no documented dependence on \(h\), \(\Delta t\), advection scale, trace inequality, level-set scaling, or spatial dimension. The same literal number therefore represents different enforcement strength in 2D and 3D and as the mesh changes.

This term is not derived as a consistent weak boundary condition for the hyperbolic level-set transport equation. It can also enforce geometry while momentum separately contains Young wall energy, without a proof that both are compatible at the discrete stage. Existing static refinement failures are consistent with this weakness, although they do not isolate it uniquely.

A robust alternative needs a derived wall condition or constrained geometric update, consistent with advection and reinitialization, plus a penalty/Nitsche scaling and convergence result.

### Unsupported wetting regimes

The current dynamic path is deliberately narrow:

- continuous scalar P1 level set;
- first-order parent geometry and `LinearCorner` contact construction;
- stationary axis-aligned planar wall;
- strong zero normal-only impermeability;
- positive literal constant \(\gamma\), mobility, and slip length;
- no complete-wetting endpoints;
- no contact-angle hysteresis, roughness law, moving wall, general curved wall, or variable-\(\gamma\)/Marangoni model.

These fail-closed restrictions are preferable to silently wrong generality, but they define a research envelope rather than a general production wetting method.

### D18/D38 are not wetting validation cases

D18/D38 set surface tension and contact-line physics off and impose full no-slip on physical side/bottom walls. On a vertical wall, no slip removes the tangential wall velocity required for a material contact line to run upward. Classical sharp-interface theory also makes a moving no-slip contact line singular.

Any apparent wall motion of \(\phi\) in those configurations can arise from numerical diffusion, discontinuity capturing, extension, redistancing, or global shift. It is not evidence of physically correct wetting. D18/D38 should be labeled bulk violent-transport/false-wet tests only.

## Level-set transport audit

### Discretization actually implemented

[`LevelSetTransport.cpp`](../Code/Source/solver/FE/LevelSet/LevelSetTransport.cpp), approximately lines 1240--1575, implements a continuous scalar H1 equation on the background mesh with either:

- advective form \(\partial_t\phi+u\cdot\nabla\phi=0\); or
- strong conservative form \(\partial_t\phi+\nabla\cdot(\phi u)=0\).

The latter differs from the advective form by \(\phi\nabla\cdot u\), which is not identically zero for a discretely divergence-free VMS/PSPG velocity. Multiplying that strong form by continuous tests does not make it locally conservative in finite-volume/DG flux balance.

SUPG uses a metric/time-based parameter. Residual discontinuity capturing adds approximately

\[
\nu_{DC}\sim h\frac{|R_\phi|}{|\nabla\phi|},
\]

capped by a combination of \(h|u|\) and \(h^2/\Delta t\). This is empirical nonlinear diffusion. It has no discrete maximum principle and can thicken or erase thin interface features.

### The limiter is a rejection gate, not a conservative transport method

The optional one-ring nodal limiter is documented as non-FCT and nonconservative. Production computes a limited candidate, but if limiting would change a converged Newton solution, it rejects the step and requests a smaller time step rather than accepting a state that was never a coupled solution. That rejection policy is logically sound, but it does not make the accepted CG/SUPG operator invariant-domain or locally conservative.

For splash problems this distinction is critical. Crown sheets, rims, satellite droplets, and thin wall films can be only a few elements thick. A method can retain global volume after correction while losing the correct local sheet mass and breakup time.

### Reinitialization

The option enum advertises Hamilton--Jacobi, fast marching, and projection methods, but the production parser supports projection only and rejects the other names. Projection repair is serial and uses displacement/topology safeguards; production accepts only a converged candidate, which is good defensive behavior.

Limitations remain:

- an arbitrary curved P1 zero set is not generally exactly redistancable in the same space;
- there is no contact-angle-aware wall condition;
- the original-crossing guard records only edges whose endpoints have strict opposite signs outside tolerance;
- zero vertices, zero edges, tangential contacts, and some contact/topology degeneracies are not represented by that crossing sample; and
- the method must be qualified on contact-angle error and contact-line displacement, not only \(|\nabla\phi|-1\).

The missing wall condition is a direct literature concern: redistancing near a contact line must extend or enforce the appropriate dynamic angle along the wall.

### Global volume correction

The supported correction adds one scalar shift to all level-set coefficients. Per-event and cumulative interface/contact displacement bounds and topology guards are valuable. The operation is nevertheless intrinsically nonlocal:

- it moves every interface and contact line;
- it redistributes local transport error between disconnected components;
- it can change a well-resolved region to compensate for an underresolved one; and
- it can make a global volume history look accurate while local mass, film thickness, and breakup are wrong.

Report both raw pre-maintenance volume and corrected post-maintenance volume. Treat the shift as a bounded fallback, not proof of conservative transport.

### Physical-volume mapping defect outside the affine envelope

Generated phase volume in [`LevelSetVolume.cpp`](../Code/Source/solver/FE/LevelSet/LevelSetVolume.cpp), lines 1218--1263, multiplies each reference-region measure by one per-cell ratio of corner-based physical parent measure to reference parent measure. Assembly instead maps generated quadrature pointwise using the Jacobian/determinant in
[`StandardAssembler.cpp`](../Code/Source/solver/FE/Assembly/StandardAssembler.cpp), lines 2891--2965.

One constant scale is exact for affine cells. It is not generally exact for distorted bilinear/trilinear or curved high-order parents whose Jacobian varies. Therefore volume diagnostics, volume targets, and correction can disagree with the measure actually used by the PDE. This is a confirmed generality defect, although rectangular affine Q1 and linear simplex cases avoid it.

There is a second measure mismatch: lifecycle phase-volume diagnostics can sum generated regions that PDE assembly later removes through the cut-rule pruning threshold. In that circumstance, the quantity labeled conserved/target volume is not exactly the volume of the domain on which the momentum and continuity equations were assembled. Volume accounting must consume the identical retained rule set, or report lifecycle and assembled measures separately.

## Interface geometry and cut quadrature audit

### Supported and experimental backends

- `LinearCorner` is the production first-order interface path.
- In 3D, its reported interface order is at most one and volume order at most two.
- Saye high-order production support is limited principally to 2D quadrilaterals; other combinations are experimental or rejected.
- `HighOrderSubcell` is experimental.
- `MomentFit` is unavailable.
- High-order contact geometry fails closed instead of silently claiming high order.

These restrictions are honest, but splash surfaces contain high curvature, very thin sheets, and frequent topology transitions. First-order faceting and topology-dependent quadrature changes can dominate capillary and breakup errors unless refinement is extremely aggressive.

### Sliver pruning and feature loss

The default tiny-fragment fraction is about `1e-8`. Pruning can be necessary for conditioning, but it introduces a topology-dependent deletion operation. A sheet, neck, satellite droplet, or contact fragment below the threshold can disappear discontinuously. Aggregation can likewise pin or eliminate a component with no well-resolved root cell.

Every splash validation should record:

- number and volume of pruned fragments by phase;
- number of components without an aggregation root;
- deleted liquid volume and surface area;
- minimum sheet/neck thickness in cells; and
- sensitivity to the prune/aggregation threshold.

The current D38 geometry also requested interface quadrature order 2 but achieved order 1 on all recorded interface cells. The backend reports this as a capability downgrade rather than a fallback, and there is no minimum-achieved-order acceptance policy. Requested order must not be treated as achieved accuracy.

### Backend validation is incomplete

The validator in
[`LevelSetImplicitCutQuadratureBackend.cpp`](../Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp), lines 5389--5524, checks finiteness, positive weights, weight sums, markers, side labels, and some metadata. It does not fully establish geometric correctness:

- root residual is enforced only when the backend marks a fragment `root_polished`;
- normal validation requires only nonnegative alignment with the evaluated gradient, so a nearly orthogonal normal is not rejected;
- volume quadrature points are not independently checked for parent-cell containment; and
- their level-set sign is not evaluated to verify the declared negative/positive phase.

This does not prove that current backend points are wrong. It proves that the common validation layer cannot detect several consequential backend errors.

### Contact reconstruction revision risk

The scalar-field overload of
[`GeneratedInterfaceBoundaryIntersectionDomain.cpp`](../Code/Source/solver/FE/Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.cpp), lines 902--1003, explicitly ignores the supplied `interface_domain` object and reconstructs boundary roots from scalar nodal values. There is no enforced invariant that contact points use the same parent fragments, geometry revision, active-side decision, and topology as the surface quadrature used for capillary work.

Independent reconstruction may be intentional, but the consistency contract is unqualified. Add same-revision and same-parent correspondence checks, plus tests at vertices, edges, nearly tangent intersections, and MPI ownership boundaries.

### High-order curvature sampling is geometrically inconsistent

The application curvature workflow constructs a supplemental reference point as the mean of reference nodes and independently constructs its purported physical location as the mean of physical nodes. It evaluates \(\phi\) at the former and stores the latter with that value. On a nonaffine or curved mapping,

\[
x\!\left(\frac{1}{N}\sum_i\xi_i\right)
\ne
\frac{1}{N}\sum_i x(\xi_i),
\]

so the least-squares sample is an inconsistent `(physical position, level-set value)` pair. The relevant implementation is in
[`ApplicationDriver.cpp`](../Code/Source/solver/Application/Core/ApplicationDriver.cpp), lines 6291--6337 and 6511--6589. Cut-rule curvature samples map their points correctly in
[`LevelSetCurvatureSamples.cpp`](../Code/Source/solver/Application/Core/LevelSetCurvatureSamples.cpp), lines 250--292.

The high-order inclusion screen also uses nodal coefficient extrema plus only one interior sample. Lagrange shape functions can be negative inside an element, so this is not a certificate of the polynomial range or interface absence.

The curvature projection itself is nodal recovery followed optionally by volumetric graph smoothing, not an interface finite-element \(L^2\) projection. Its `MassStiffness` mode estimates measures from primary corners and builds heuristic graph weights rather than using the actual cut-surface metric. These limitations are mostly outside the supported affine-P1 `SurfaceStress` path, which avoids explicit curvature, but they make high-order legacy-curvature and diagnostic claims unqualified and establish FSR-17.

## Cut-cell mixed stability and conditioning

The current method combines equal-order VMS/PSPG stabilization with pressure ghost penalty or small-cut aggregation. The pressure ghost term uses empirical scaling of the form

\[
0.01\frac{h^3}{\mu+\rho h^2/\Delta t}
\]

for first derivative jumps, with an \(h^5\) second-derivative option. Velocity ghost penalty is retired in the reviewed path; aggregation may constrain both velocity and pressure. Cut-metadata scaling is optional and not the universal default.

This is not automatically equivalent to CutFEM analyses that prove cut-independent coercivity, inf-sup stability, or condition-number bounds for a particular velocity/pressure pair with both velocity and pressure stabilization. Nor is it automatically equivalent to AgFEM theory for a different aggregate space and pressure choice.

Existing rank evidence is mixed: one fixed `n=2` coupling is rank `6/8` without stabilization, while larger or stabilized samples can be full rank. That is useful diagnostic evidence, not a uniform theorem.

Small-cut aggregation preserves constants through a scaled partition check, but otherwise accepts finite positive or negative extrapolation weights without a uniform row-\(L^1\), path-length, or conditioning bound. A rootless disconnected active island may be pinned/deleted to keep the algebraic system solvable. That is a feature-deletion policy, not conservation of a small physical droplet, and must be exposed in splash telemetry.

Required qualification is a systematic family over:

- cut fraction from roughly `1e-8` through `0.5`;
- subcell position and orientation;
- mesh refinement and polynomial order;
- viscosity, density, and time-step limits;
- serial and multiple MPI partitions;
- pressure-only ghost penalty, aggregation, and any combined mode; and
- connected and subresolution disconnected liquid components.

Record smallest relevant singular values or inf-sup surrogates, condition estimates, Krylov iterations, divergence norm, pressure error, velocity error, and constraint amplification.

## Nonlinear and temporal coupling

The interface geometry and extension maps are refreshed around nonlinear solves. `SurfaceStress` does not accept differentiated cut quadrature. As a result:

- field derivatives at fixed geometry can be exact;
- the overall coupled update is not a monolithic Newton method in shape;
- map refresh can cause a residual jump unrelated to the frozen-map tangent;
- contact-angle observations can refer to a different state/stage than velocity observations; and
- an energy decrease cannot be inferred solely from positive individual dissipation terms.

The minimum required telemetry key is

`(accepted_step, accepted_time, dt, state_revision, geometry_revision, map_revision, nonlinear_iteration, nonlinear_phase, generalized_alpha_stage)`.

At each accepted step, record kinetic, gravitational, liquid--gas surface, and Young wetted-wall energies, plus bulk viscous, wall-slip, and line-friction dissipation. Check a residual form of the complete energy balance, including maintenance and stabilization work.

## Fitted-ALE free-surface audit

The public fitted-free-surface interface declares tangential mesh policies `Free`, `SmoothingOnly`, and `Prescribed`, including a prescribed tangential velocity. These are parsed in
[`NavierStokesRegister.cpp`](../Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesRegister.cpp), around line 2832, and checked for finite values in
[`IncompressibleNavierStokesVMSModule.cpp`](../Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp), around line 3204. No production consumer applies the selected policy or velocity.

This is a confirmed user-visible defect: all three settings currently have the same behavior.

Other fitted-path limitations are explicit:

- `SurfaceStress` is not the fitted production contact formulation;
- fitted dynamic and prescribed contact angle are unsupported;
- only a pinned/kinematic subset is available; and
- local free-surface Nitsche settings are written into module-global values.

Per-BC Nitsche parsing at `NavierStokesRegister.cpp` around line 2869 mutates global `options.nitsche_*`. Those globals control all fitted free surfaces, mesh-side penalties, and unrelated weak velocity conditions. Multiple BCs are therefore last-one-wins and registration-order dependent.

## Configuration and lifecycle defects

### Incomplete contact input can be ignored

The parser decides that contact-line configuration is present using only a subset of model/angle keys. A wall marker/normal, mobility, slip law/length, or penalty provided without the detected keys can be discarded by an early return rather than rejected as an incomplete configuration. Every contact-related key should participate in one all-or-none schema with explicit diagnostics.

### Custom operator registration can misroute the angle residual

The default unified `equations` flow registers the correct level-set row owner. A custom registration sequence can install the prescribed-angle residual on the NS operator when no phi owner exists yet; a later differently tagged level-set owner is not reconciled. This is a custom-path lifecycle defect, not a failure of the default route.

### Advertised but unavailable reinitialization modes

Options enumerate Hamilton--Jacobi and fast-marching redistancing, but production parsing rejects both. Documentation/schema should distinguish planned enum values from supported runtime methods.

## Existing numerical evidence and what it does not prove

### Static sessile caps

The corrected initializer samples the analytic cap directly at Q1 vertices. All six `60/90/120 degree` cases at `n=16,32` complete three steps. Every `n=32` row passes its per-mesh gates, while `n=16` angle gates fail at 90 and 120 degrees. Refinement fails:

- the 60-degree pressure rate: `-2.8226`; and
- the 90-degree capillary-number rate: `-0.4569`.

Static pressure and parasitic-current behavior is therefore not balanced or convergent under the fixed matrix. The best-pressure-space residual in FSR-03 provides a direct mechanism.

Artifacts: [free_surface_review_completion_20260717_static_initializer_correction](qualification_logs/free_surface_review_completion_20260717_static_initializer_correction/).

### Moving side-wall contact

The true side-wall matrix establishes correct orientation and completed horizons, but only one of four Ren--E speed gates passes. It is moving-wetting evidence and currently negative evidence.

Artifacts are the four directories beginning with
`qualification_logs/free_surface_review_completion_20260717_vertical_sidewall_physical_`.

### Capillary wave

The recorded `n=16` capillary-wave run completes 100 steps and passes its scoped gates:

- maximum relative liquid-volume drift `9.25622e-7`;
- frequency error about `1.2%`; and
- profile-amplitude error about `1.50241%`.

This is useful single-resolution smooth-interface evidence. It does not establish spatial convergence, moving contact, static balance, topological robustness, or a complete energy law.

Artifact: [capillary_wave_n16.json](qualification_logs/free_surface_review_completion_20260717_final_entity_measures_wave/capillary_wave_n16.json).

### Impermeable-wall advection

The `n=8,16` wall-advection isolation completes 512 steps at each resolution, has zero false-liquid wall nodes/sign flips, and reports RMS rate `2.04184`. It validates prescribed-velocity transport in that smooth setup, not dynamic wetting or momentum/geometry feedback.

Artifact: [qualification_summary.json](qualification_logs/free_surface_review_completion_20260717_final_entity_measures_wall_advection/qualification_summary.json).

### D18/D38

D18 accepts one reduced step to `1.5625e-5`; D38 accepts two reduced steps to `3.75e-4`. Neither reaches the nominal `5e-4` gate or the required long horizon `t>=0.281`. D38 exhibits the severe map amplification described in FSR-02.

Artifacts:

- [D18 one-step diagnostic](qualification_logs/free_surface_review_completion_20260717_final_d18_d38_one_step/d18/d18.json)
- [D38 one-step diagnostic](qualification_logs/free_surface_review_completion_20260717_final_d18_d38_one_step/d38/d38.json)

These are not wetting tests because `gamma=0`, there is no contact law, and the physical walls are no-slip.

## Literature synthesis for finite-element implementation

### Surface tension and balanced force

| Reference | Relevant result | Consequence for this code |
|---|---|---|
| [Bänsch, 2001](https://doi.org/10.1007/PL00005443) | Finite-element surface-tension treatment tied to interface evolution can improve stability. | Surface virtual work, interface update, and time discretization must be analyzed together. |
| [Groß & Reusken, 2007](https://doi.org/10.1137/060667530) | Analyzes FE error in surface-tension forces and pressure/interface compatibility. | Correct continuum traction is insufficient; measure the discrete pressure-range distance and parasitic velocity. |
| [Buscaglia & Ausas, 2011](https://doi.org/10.1016/j.cma.2011.06.002) | Variational surface tension avoids explicit curvature but retains discrete-geometry requirements. | `SurfaceStress` is the right family of formulation, but FSR-03 still needs a compatible discrete equilibrium. |
| [Barrett, Garcke & Nürnberg, 2013](https://arxiv.org/abs/1306.2192) | Stable parametric FE methods and pressure enrichment can achieve strong energy/volume/equilibrium properties. | Treat exact balance as a property of the entire space/geometry/time pair, not one force term. |
| [Popinet, 2009](https://doi.org/10.1016/j.jcp.2009.04.042) | Machine-accurate static droplet balance and capillary-wave/breakup benchmarks are used as stringent surface-tension tests. | Static parasitic currents and translating-drop/capillary-wave tests should be release gates. |
| [Zahedi et al., 2012](https://doi.org/10.1002/fld.2643) | FE level-set study specifically examines spurious currents. | Report pressure, velocity, curvature, and cut-position convergence together. |

### Unfitted FE stability and extension

| Reference | Relevant result | Consequence for this code |
|---|---|---|
| [Massing et al., 2014](https://doi.org/10.1007/s10915-014-9838-9) | A particular CutFEM Stokes formulation uses ghost penalties to obtain stability and cut-independent conditioning. | Pressure-only empirical stabilization in a different equal-order formulation does not inherit this theorem. |
| [Burman & Hansbo, 2014](https://doi.org/10.1051/m2an/2013123) | Fictitious-domain/CutFEM analysis demonstrates why extension and stabilization must be consistent. | Replace same-velocity dry diffusion with a proven stabilization or a genuinely separate extension. |
| [Frachon & Zahedi, 2019](https://doi.org/10.1016/j.jcp.2019.01.028) | Space-time CutFEM captures pressure/velocity discontinuities with stabilization designed for cut-independent conditioning and stabilized curvature. | Use it as a comparison target for pressure, velocity, and geometry stabilization, not as evidence the current scheme is equivalent. |
| [Badia et al., 2018/2019](https://arxiv.org/abs/1805.01727) | Aggregated unfitted FE spaces can avoid small-cut ill conditioning under a defined aggregate-space construction. | Prove constant reproduction, approximation, stability, and component policy for the actual aggregate constraints. |
| [Burman, Hansbo & Larson, 2022](https://arxiv.org/abs/2205.01340) | Relates locking-free ghost penalties to discrete-extension spaces. | A stable discrete extension has a defined kernel and bounds; arbitrary dry diffusion or regression does not. |
| [Olshanskii, Reusken & Schwering, 2025](https://doi.org/10.1137/24M1674182) | A narrow-band FE level-set method constructs extension through an analyzed \(L^2\)/\(H^1\) projection plus ghost penalty. | This is a direct design comparison for replacing unchecked regression and same-field dry diffusion with a stable auxiliary extension. |
| [Saye, 2022](https://arxiv.org/abs/2105.08857) | High-order quadrature over polynomial implicit surfaces/volumes attains high-order convergence with explicit geometric algorithms. | High-order configuration must require demonstrated achieved order, pointwise mapping, root accuracy, and topology handling. |

### Moving contact lines and wetting

| Reference | Relevant result | Consequence for this code |
|---|---|---|
| [Huh & Scriven, 1971](https://doi.org/10.1016/0021-9797(71)90188-3) | A sharp moving contact line with no slip is singular. | D18/D38 no-slip wall motion cannot be treated as physical wetting. |
| [Ren & E, 2007](https://doi.org/10.1063/1.2646754) | Effective wall/contact-line conditions couple dynamic angle, wall slip, and contact speed. | Verify advancing/receding sign, mobility/friction scaling, and resolved slip length as a coupled law. |
| [Sprittles & Shikhmurzaev, 2012](https://doi.org/10.1002/fld.2603) | Develops a consistent FEM framework and shows the importance of resolving contact-line scales. | Use resolved-slip convergence rather than tuning one mesh-dependent wall width. |
| [Reusken, Xu & Zhang, 2017](https://doi.org/10.1002/fld.4349) | Derives a variational sharp-interface MCL formulation, energy estimate, XFEM pressure, Nitsche wall slip, and validation caps. | This is the closest full FE contract for surface, wall, and line energy ownership. |
| [Zhao & Ren, 2020](https://doi.org/10.1016/j.jcp.2020.109582) | Energy-aware moving-mesh FE coupling of Navier slip and dynamic contact angle. | Establish the corresponding discrete energy residual for this unfitted/time-integrated method rather than assuming transfer of stability. |
| [Xu & Ren, 2016](https://doi.org/10.4208/cicp.210815.180316a) | Redistancing at a contact line requires a compatible wall/contact-angle treatment. | Add contact-angle and line-displacement gates to reinitialization. |
| [Della Rocca & Blanquart, 2014](https://doi.org/10.1016/j.jcp.2014.01.040) | Identifies a contact-line redistancing “blind spot” and tests wall-aware repair on wedges/arcs and moving drops. | Interior distance error alone cannot qualify wall redistancing; near-wall distortion and parasitic currents need direct gates. |
| [Gründing et al., 2020](https://doi.org/10.1016/j.apm.2020.04.020) | Capillary-rise benchmark compares multiple methods and shows strong dependence on physical versus numerical slip. | Capillary rise between plates is a high-value side-wall wetting benchmark with public data. |

### Interface transport and maintenance

| Reference | Relevant result | Consequence for this code |
|---|---|---|
| [Sussman, Smereka & Osher, 1994](https://doi.org/10.1006/jcph.1994.1155) | Classical level-set advection/redistancing framework. | Redistancing and interface displacement must be measured separately from transport. |
| [Enright et al., 2002](https://doi.org/10.1006/jcph.2002.7166) | Reversible 3D deformation test exposes interface loss under extreme stretching. | Use it to quantify raw volume, shape recovery, topology, and maintenance error. |
| [Rider & Kothe, 1998](https://doi.org/10.1006/jcph.1998.5906) | Deformation-field tests emphasize geometric transport and mass conservation. | CG boundedness alone is not sufficient; local interface mass and reversibility must converge. |
| [Kuzmin & Quezada de Luna, 2020](https://arxiv.org/abs/2003.12007) | Convex limiting for continuous FE scalar conservation laws can enforce invariant domains, local bounds, and entropy conditions through constrained fluxes. | If continuous FE transport is retained, compare against an actual AFC/convex-limited conservative formulation rather than a post-solve nodal preview. |

## Recommended validation hierarchy

Validation should proceed from exact operator tests to smooth physics, then wetting, then violent free surfaces, and only then splashing. Passing a later visually plausible case must never override a failed earlier balance or conservation gate.

### Tier 0: algebraic, geometric, and manufactured verification

#### 0A. Flat free surface with hydrostatics

Use a box with

\[
u=0,\quad \phi=y-h,\quad p=p_{ext}+\rho g(h-y),\quad \theta=90^\circ.
\]

Repeat for every coordinate direction, gravity sign, active-side sign, wall orientation, and several arbitrary cut offsets. Required metrics:

- assembled residual at the exact field;
- pressure and velocity error;
- surface-force/pressure-range residual;
- parasitic kinetic energy;
- volume before/after maintenance; and
- invariance under positive scaling of \(\phi\).

This is the first release gate because curvature is zero and geometry ambiguity is minimal.

#### 0B. Static closed circle/sphere and sessile cap

For a sphere of radius \(R\), use

\[
p_l=p_{ext}+\gamma(d-1)/R,
\]

and zero velocity. A published Groß--Reusken example uses a 3D sphere with `R=2/3`, `gamma=1`, and pressure jump `3`. Measure the best pressure-space residual, pressure jump, parasitic currents, area, curvature, and energy over at least three mesh levels and many cut translations.

For contact, reproduce the [Reusken--Xu--Zhang](https://doi.org/10.1002/fld.4349) stationary cap: cube `(0,0.5)^3`, `R=0.1`, `theta=60 degrees`, `gamma=0.5`, equal density/viscosity 1, gravity 0, wall-friction parameter as published, and expected pressure jump `10`. Their finest reported velocity is near `1e-8`; use the paper's angle/base-radius convergence as comparison. Rotate the cap to every wall and test `30/60/90/120/150 degrees` plus phase-sign and mirror transformations.

#### 0C. Extension invariance suite

Use a manufactured liquid velocity and vary only:

- dry-domain width;
- dry-wall value and boundary type;
- number and spacing of disconnected liquid components;
- cut location;
- \(\alpha\) and the cut floor; and
- algebraic map reconstruction.

The liquid residual and solution must be invariant to irrelevant dry-domain changes to tolerance. Record row-block deltas, interface flux, map condition, max coefficient, row-\(L^1\), constant reproduction, and \(\|E\|/\|u\|\).

#### 0D. Cut-position stability sweep

Translate a planar interface through one element so retained fractions range from about `1e-8` to `0.5`. Repeat with rotations, refinements, MPI partitions, and all stabilization modes. At each `h`, require conditioning and preconditioned iterations to remain cut-position independent relative to the fitted/reference system; across `h`, require the theoretically expected canonically scaled condition-number trend plus convergent pressure, velocity, divergence, and force residual.

#### 0E. Reversible level-set deformation

Use the Enright 3D sphere of radius `0.15` centered at `(0.35,0.35,0.35)`, reversing at period `T=3`, with

\[
u=2\sin^2(\pi x)\sin(2\pi y)\sin(2\pi z)\cos(\pi t/3),
\]
\[
v=-\sin(2\pi x)\sin^2(\pi y)\sin(2\pi z)\cos(\pi t/3),
\]
\[
w=-\sin(2\pi x)\sin(2\pi y)\sin^2(\pi z)\cos(\pi t/3).
\]

Report raw and maintained volume, symmetric-difference volume, Hausdorff/interface distance, component count, minimum sheet thickness, and final shape recovery over at least three `h` and `dt` levels.

### Tier 1: smooth capillary and wetting physics

#### 1A. Translating equilibrium drop

Use the Popinet translating-drop test (`D=0.4` in a unit square, prescribed translation, `We=0.4`, and several Laplace numbers). A correct method should translate without deformation, pressure drift, or growing parasitic currents. This simultaneously tests transport, pressure/surface balance, and Galilean behavior.

#### 1B. Linear capillary wave

Compare amplitude and phase against Prosperetti's viscous theory:

- [one viscous fluid, 1976](https://doi.org/10.1063/1.861446);
- [two superposed viscous fluids, 1981](https://doi.org/10.1063/1.863522), for a future two-phase solver.

Extend the existing single `n=16` pass to at least three spatial and temporal levels. Gate frequency, damping, phase, pressure amplitude, raw volume, maintained volume, and total-energy residual.

#### 1C. Droplet oscillation

For an inviscid one-phase spherical droplet, the Rayleigh `l=2` frequency is

\[
\omega^2=\frac{8\gamma}{\rho R^3}.
\]

Use small initial deformation first, then viscous damping comparisons from [Prosperetti](https://doi.org/10.1017/S0022112080001188) and [Miller & Scriven](https://doi.org/10.1017/S0022112068000832). Measure frequency, damping, mode purity, volume, pressure, and energy.

#### 1D. Spreading and contracting sessile drop

Reproduce Reusken--Xu--Zhang relaxation from `theta0=90 degrees` to equilibrium `30/60/120 degrees`. For their spherical-cap volume, the equilibrium radius relation is

\[
R=R_0\left[
\frac{2}{2-2\cos\theta-\sin^2\theta\cos\theta}
\right]^{1/3},\qquad r=R\sin\theta.
\]

Published base radii are approximately `0.169384`, `0.127619`, and `0.0727416`. Gate equilibrium radius/angle, line speed versus constitutive prediction, monotone energy decay, wall-slip and line-friction dissipation, and independence from wall-indicator width.

#### 1E. Resolved moving-contact benchmark

Use the finite-element framework of [Sprittles & Shikhmurzaev](https://doi.org/10.1002/fld.2603), including cases around `Re=10`, `Ca=0.01`, large nondimensional slip parameter, and `30-degree` contact angle. The slip length must be spatially resolved; run independent slip, mesh, and time refinement. Compare interface shape, contact speed, pressure, and stress rather than angle alone.

#### 1F. Capillary rise between parallel plates

Use the public [Gründing et al. benchmark](https://arxiv.org/abs/1907.05054) and [associated data](https://doi.org/10.25534/tudatalib-173). Compare rise height, overshoot, oscillation period, damping, equilibrium height, and contact-line evolution. This is the highest-priority new side-wall wetting validation because it distinguishes physical Navier slip from numerical slip.

#### 1G. Linear sloshing

Use analytical linear sloshing frequencies and [NASA SP-106](https://ntrs.nasa.gov/citations/19670006555) as references. Gate elevation histories at multiple probes, phase, damping, pressure, volume, and energy. Then proceed to nonlinear sloshing only after linear convergence.

### Tier 2: violent one-phase free-surface flow

#### 2A. Dam break and wall impact

- [Martin & Moyce, 1952](https://doi.org/10.1098/rsta.1952.0006): classical front-position/collapse history.
- [Lobovský et al., 2014](https://arxiv.org/abs/1308.0115): statistically characterized wall-impact pressure histories.
- [Kleefsman et al., 2005](https://doi.org/10.1016/j.jcp.2004.12.007): 3D dam break with obstacle, free-surface gauges and loads.
- [SPHERIC Test 02](https://www.spheric-sph.org/tests/test-02): 3D dam-break free-surface evolution.
- [SPHERIC Test 05](https://www.spheric-sph.org/tests/test-05): wet-bed dam-break evolution.

Use experimental distributions and impulse, not a single pressure peak. Report front/run-up, gauge histories, component topology, raw/corrected mass, impact impulse, and mesh/time uncertainty.

#### 2B. Solitary-wave run-up

Use [Synolakis, 1987](https://doi.org/10.1017/S002211208700329X) for nonbreaking and breaking run-up. Compare shoreline/contact position, maximum run-up, reflected wave, and mass. Interpret wall/beach contact carefully because a mathematical shoreline model is required.

#### 2C. Nonlinear sloshing and impacts

Use [Faltinsen et al.](https://doi.org/10.1017/S0022112099007569), [Cruchaga et al.](https://doi.org/10.1007/s00466-013-0877-0), and [SPHERIC Test 10](https://www.spheric-sph.org/tests/test-10). Test 10 supplies repeated lateral/roof impact histories for water and oil, but the tank is closed and contains air. A one-phase solver can compare bulk elevation and early liquid kinematics; roof pressure and trapped-air effects require a gas model and should not be acceptance gates for the present physics.

### Tier 3: splash, sheet, rim, and breakup validation

#### 3A. Wet-film crown splash: recommended first splash benchmark

Use splash onto an existing liquid film before dry-wall splash. It avoids a moving dry contact line during the earliest crown formation and is closer to the current one-phase physics while gas effects remain modest.

Recommended data:

- [Cossali et al., 2004](https://doi.org/10.1007/s00348-003-0772-0): time-resolved crown diameter, height, thickness, rim jets, and secondary drops.
- [Geppert et al., 2017](https://doi.org/10.1007/s00348-017-2447-2): quantitative one- and two-component crown benchmark over Weber number and film thickness, including base/top diameter, height, wall angle, fingers, and secondary droplets.
- [Bagheri et al., 2026 open dataset](https://tudatalib.ulb.tu-darmstadt.de/items/c83209b0-090c-4f4c-a3dc-c53fe58534f7): drop impact on `h=0.4 mm` and `0.5 mm` films, with time-resolved `D_Base`, `D_Mid`, `D_Rim`, and `H_Rim`, experimental standard deviations from 10--14 repeats, simulation sheet/film thickness, and videos.
- [Yarin & Weiss, 1995](https://doi.org/10.1017/S0022112095002266): kinematic crown-spreading theory and scaling.

Compare against uncertainty bands, not only center curves. Minimum outputs are:

- crown base, mid, and rim diameter;
- rim height and speed;
- sheet thickness versus height/time;
- residual film thickness;
- rim radius and finger count;
- secondary-droplet size/count when resolved;
- raw and corrected liquid volume;
- surface area and energy histories; and
- pruned/aggregated feature statistics.

The current P1/LinearCorner method should first target the smooth pre-breakup crown interval. Secondary atomization is not credible until sheet thickness and rim wavelength are independently resolved and a gas phase is available.

#### 3B. Contracting liquid filament and capillary breakup

Use [Notz & Basaran, 2004](https://doi.org/10.1017/S0022112004009759) for contracting filaments and [Castrejón-Pita et al., 2012](https://doi.org/10.1103/PhysRevLett.108.074506) for breakup dynamics. Measure minimum neck radius, end-pinching time, satellite volume, component count, and similarity scaling. This is an exacting topology and local-conservation test without a wall contact line.

#### 3C. Capillary jet breakup

Use the jet breakup cases in [Popinet, 2009](https://doi.org/10.1016/j.jcp.2009.04.042). Compare Rayleigh--Plateau growth rates, breakup wavelength/time, and satellite size. This directly exposes nonconservative loss and sliver deletion.

#### 3D. Dry-wall splash: future two-phase target

Dry-wall splashing is strongly affected by gas pressure and air cushioning:

- [Mundo et al., 1995](https://doi.org/10.1016/0301-9322(94)00069-V);
- [Xu, Zhang & Nagel, 2005](https://doi.org/10.1103/PhysRevLett.94.184505); and
- [Mani, Mandre & Brenner, 2010](https://doi.org/10.1017/S0022112009993594).

These are not fair quantitative acceptance tests for the current one-phase model. They should be deferred until gas inertia/viscosity and, where relevant, compressibility and dynamic wetting are represented. The present solver may use them only for qualitative topology experiments explicitly labeled outside-model.

### Tier 4: future two-phase validation

The [Hysing et al. bubble benchmark](https://doi.org/10.1002/fld.1934) is a standard future test for density/viscosity jumps, surface tension, and topology. It uses `[0,1] x [0,2]`, initial radius `0.25` centered at `(0.5,0.5)`, and reports circularity, center of mass, and mean rise velocity. Case 1 has robust intercode reference quantities; Case 2 agrees only before breakup and should not be assigned a single exact post-breakup shape.

A current one-phase free-surface solver cannot represent this benchmark's second fluid. The same applies to trapped-bubble and air-entrainment splash cases.

## Acceptance protocol

Every production qualification should meet the following protocol.

1. **Predeclare gates.** Fix errors/tolerances before running; do not tune them to observed results.
2. **Refine space and time independently.** Use at least three meaningful `h` levels and three `dt` levels, including a fixed-`h` time study and fixed-`dt` space study.
3. **Sweep cut geometry.** Translate and rotate the same physical interface relative to the background mesh.
4. **Sweep partitions.** Repeat representative cases in serial and multiple MPI decompositions.
5. **Expose maintenance.** Report raw post-transport and final post-reinitialization/post-correction values separately.
6. **Measure local conservation.** Track per-component volume, wall-film mass, crown-sheet mass, and flux balance, not only total volume.
7. **Measure force balance.** Report best pressure-space residual, pressure error, surface-force residual, divergence, and parasitic kinetic energy.
8. **Measure energy.** Report all physical energy and dissipation terms plus stabilization, extension, pruning, and maintenance work.
9. **Resolve regularization scales.** Slip length, wall smoothing width, film thickness, sheet thickness, and neck/rim radius must be resolved independently of the grid.
10. **Quantify numerical interventions.** Record step rejection, dt reduction, limiter requests, map fallback, aggregation, pruning, reinitialization displacement, and volume shift.
11. **Use experimental uncertainty.** Compare histories and distributions against repeatability bands; do not gate a chaotic impact by one peak.
12. **Respect the physics envelope.** Do not claim validation of gas-dependent splash using prescribed external pressure.

## Recommended remediation order

### P0: eliminate formulation feedback and establish exact low-level gates

1. Replace or rigorously reformulate the same-velocity dry-domain Laplacian. Add wet-block invariance and two-island tests first.
2. Guard the algebraic extension map with rank/condition, coefficient, row-norm, constant-reproduction, negative-weight, and preview-amplification limits plus a bounded fallback.
3. Turn the pressure best-space residual into a fixed physical acceptance gate; establish flat-interface and discrete static-cap balance before further wetting tuning.
4. Derive and implement a contact-angle wall treatment that is consistent across level-set transport, surface/wall energy, and reinitialization. Remove the unscaled penalty or supply a defensible scaling and convergence result.

### P1: close dynamic wetting

5. Add accepted-state/stage/revision provenance to angle and contact-speed telemetry.
6. Compare exact sharp wetted-wall integration against the current smoothed indicator; refine wall width and slip length independently.
7. Re-run advancing/receding and sessile relaxation matrices until sign, speed, equilibrium, and energy gates pass at multiple meshes.
8. Add the public capillary-rise benchmark as the primary side-wall physical validation.

### P2: conservation, cut stability, and energy

9. Select a locally conservative or demonstrably mass-preserving interface-transport strategy suitable for sheets and breakup; keep raw conservation visible even if global correction remains.
10. Establish cut-position-independent stability/conditioning for the actual velocity/pressure spaces and stabilization, including disconnected components.
11. Make volume measurement use the same pointwise physical mapping as assembly on nonaffine/high-order cells.
12. Clip every generic exterior weak/natural boundary form to the sharp wetted boundary and verify cut-fraction consistency.
13. Strengthen cut backend validation, fix high-order curvature sampling, and enforce common geometry revision between surface and contact rules.
14. Derive and gate a complete discrete energy residual, including extension and maintenance work.

### P3: broader methods and splash

15. Wire or remove fitted-ALE tangential mesh options and make Nitsche policy BC-local.
16. Complete smooth Tier 0/1 validation, then dam-break/sloshing Tier 2.
17. Begin splash qualification with the 2026 wet-film crown dataset over its smooth pre-breakup interval.
18. Add a solved gas phase before claiming dry-wall splash, trapped-air impact, entrainment, or secondary atomization accuracy.

## Detailed implementation and qualification execution notes

This section converts findings FSR-01 through FSR-18 into an implementation and test program. It is a plan, not a record that the changes or simulations have already been completed. The source should be changed only through separately reviewed work, and each method change should be qualified before a later benchmark is allowed to mask an earlier defect.

### Implementation and qualification status

A work-package box is checked only after its source changes, required tests, and exit evidence below are complete. A qualification box is checked only after the complete predeclared matrix passes and its machine-readable artifacts are archived.

- [x] WP-0: configuration containment and effective-state provenance
- [x] WP-1: remove physical dry-domain feedback and bound the transport extension
- [x] WP-2: one authoritative cut/interface/wet-wall/contact geometry snapshot
- [x] WP-3: sharply clipped exterior boundary operators, qualified for the one-phase affine C0 P1/LinearCorner envelope; higher-order selection fails closed
- [ ] WP-4: balanced capillary pressure, wall energy, and prescribed angle
- [ ] WP-5: side-wall contact-line dynamics and wall-aware maintenance
- [ ] WP-6: locally conservative interface transport and maintenance transaction
- [ ] WP-7: coherent small-cut stability and conditioning
- [ ] WP-8: geometry coupling, nonlinear convergence, and a complete energy law
- [ ] WP-9: fitted-ALE free-surface policies
- [ ] WP-10: explicit one-phase boundary and a staged two-phase extension
- [ ] Q0: harness, provenance, and negative configuration tests
- [ ] Q1: exact algebra, geometry, boundary, transport, and cut tests
- [ ] Q2: hydrostatic, capillary-equilibrium, and contact-equilibrium tests
- [ ] Q3: pure transport and smooth free-surface dynamics
- [ ] Q4: dynamic wetting and side-wall qualification
- [ ] Q5: violent but one-phase-compatible free-surface motion
- [ ] Q6: film, filament, jet, and pre-breakup crown motion
- [ ] Q7: two-phase and gas-sensitive phenomena

The objective is not to make one splash image look plausible. The objective is to establish a reproducible chain from discrete identities and cut-cell invariants through static capillarity, moving wetting, smooth dynamics, violent one-phase motion, and only then breakup or gas-dependent splash. A finding is closed only when all of the following exist:

1. a derivation or design decision that states the discrete contract and its applicability limits;
2. a source implementation with unsupported legacy paths disabled or clearly isolated;
3. serial and MPI unit, assembly, and integration tests where applicable;
4. acceptance thresholds declared before the qualifying runs;
5. machine-readable results tied to source, configuration, mesh, and reference-data hashes; and
6. a short qualification record showing that the complete matrix passes without case-specific retuning.

Passing a single benchmark, hiding raw transport error with volume correction, or obtaining nonlinear convergence after reducing the time step is not sufficient closure.

### Non-negotiable implementation invariants

The repaired infrastructure should enforce these invariants centrally rather than relying on every consumer to reconstruct them correctly.

1. **One authoritative geometry state.** One immutable, revision-tagged snapshot must own retained liquid/dry volume rules, interface rules, sharply wetted exterior-boundary rules, contact rules, normals, pointwise geometry maps and Jacobians, pruning decisions, topology IDs, and MPI ownership. Surface and contact geometry must never be independently rebuilt from nominally similar inputs.
2. **One-way transport extension.** Physical liquid velocity may generate an auxiliary extension field used to transport the level set, but the extension must not add rows or tractions to physical momentum. If a cut-conditioning operator acts on physical velocity, it must be named, derived, and qualified as cut stabilization rather than disguised as extension.
3. **Identical assembled and measured domains.** Assembly, phase-volume accounting, force and energy diagnostics, output, and maintenance must consume the same retained quadrature and pointwise physical mapping. If an unpruned geometric volume is also useful, it must be reported under a different name.
4. **A complete work and energy ledger.** Every contribution must be classified as variation of stored energy, physical dissipation, external work, or explicitly reported numerical work. Stabilization, aggregation, extension, pruning, redistancing, limiting, and global correction may not disappear from the balance.
5. **Fail-closed configuration.** Incomplete contact models, stale geometry, ambiguous active domains, unsupported fitted-ALE policies, falsely claimed quadrature order or achieved order below the predeclared required minimum, excessive extension amplification, and unresolved rootless features must fail before operator registration or step acceptance. A parsed but unused option or silent whole-face fallback is not acceptable.

### Architecture decisions to freeze before coding

The following decisions affect several findings and should be recorded as short architecture decision records. Competing prototypes should be compared on a fixed low-level gate matrix; coefficients should not be tuned separately for each validation case.

| Decision | Alternatives that need an explicit choice | Evidence required before selection |
|---|---|---|
| AD-1: unfitted transport extension | Auxiliary bounded extension only; or a mathematically analyzed discrete extension/aggregation construction | One-way wet-block invariance, reproduction, amplification, component-separation, and MPI tests |
| AD-2: capillary force/pressure pair | Discrete surface-energy variation with compatible pressure; stabilized mean-curvature vector; pressure enrichment/lifting; trace/XFEM pressure; or a fitted/parametric interface option | Flat exact balance, static-circle/sphere and sessile-cap convergence, force representability, consistency in moving cases, and an energy argument |
| AD-3: small-cut stability | A complete CutFEM formulation with the needed velocity/pressure ghost penalties; or a rigorously defined aggregate finite-element space | Cut-position-independent errors, inf-sup surrogate, cut-relative/canonically scaled conditioning, constraint amplification, conservation, and MPI invariance |
| AD-4: conservative interface representation | Locally conservative phase-indicator/phase-volume transport using AFC/FCT continuous FE, DG/FV, or a coupled level-set/volume-fraction method | Cancellation of the appropriate discrete phase flux, boundedness, component conservation, geometry accuracy, topology behavior, and momentum consistency |
| AD-5: geometry/nonlinear coupling | Complete fixed-topology shape tangent; or an energy-stable partitioned/outer iteration converged to a common stage | Directional derivatives or a discrete energy proof, refresh neutrality, stage consistency, and topology-event policy |
| AD-6: physical capability | Qualified one-phase liquid envelope; incompressible two-phase extension; later compressible gas where required | Separate qualification matrices and explicit claims for each model; no reuse of one-phase success as two-phase evidence |

### Dependency and delivery sequence

The recommended dependency chain is:

```text
typed configuration and containment
    -> authoritative geometry and bounded one-way extension
    -> balanced capillarity, sharp wet boundaries, and coherent cut stability
    -> contact-line geometry/reinitialization and conservative transport
    -> complete energy/time integration and fitted-ALE closure
    -> smooth and violent one-phase qualification
    -> film, filament, jet, and pre-breakup crown qualification
    -> two-phase and gas-sensitive splash qualification
```

This order is intentional. Dynamic-contact coefficients should not be calibrated while fictitious dry momentum, unbounded extension, unbalanced surface force, or whole-face dry boundary work remain active. Likewise, breakup data should not be used to tune global volume correction before local transport conservation and sliver handling have been exposed.

Before the first implementation change, preserve the current failing and passing artifacts as a read-only baseline, including the effective inputs, current source hashes, D18/D38 map histories, pressure-representability output, four Ren--E gate results, raw/post-maintenance volume, solver histories, and resource use. Develop each work package through reviewable changes with its new tests added before or with the implementation. After every package, rerun its prerequisites and fixed baseline cases with a one-factor A/B comparison; do not combine several method replacements into one unexplained benchmark improvement.

### WP-0: configuration containment and effective-state provenance

**Findings addressed:** FSR-10, FSR-11, and FSR-12, plus fail-closed protection for all later work.

Implementation notes:

- Introduce a typed, discriminated contact configuration with explicit `None`, `Pinned`, `PrescribedAngle`, and `DynamicRenE` alternatives. If any contact-related key is present, parse the entire selected block and diagnose every missing, conflicting, unknown, or model-inapplicable key. Do not silently return when only a subset of aliases is present.
- Move free-surface Nitsche settings into each `FreeSurfaceBoundary`. Keep module-global Nitsche values only for generic weak velocity conditions. Registration order must not affect either boundary.
- Treat the fitted-ALE tangential options as unsupported until WP-9 wires them into mesh motion. A requested but unimplemented policy must stop configuration rather than run as a no-op.
- Add a two-pass setup: first create and validate typed fields, boundary policies, level-set ownership, and geometry requests; only then register forms and constraints. Remove fallback owner selection for contact-angle operators.
- Write one effective-configuration artifact after validation. It should include defaults after expansion, units, active phase sign, wall/contact markers, angle convention, slip length, mobility or line friction, smoothing widths, free-surface and Nitsche policy, maintenance policy, stabilization, pruning, extension guards, and capability label.
- Version the schema and provide explicit checkpoint/configuration migration. Old inputs that select legacy behavior must say so and may not inherit production-qualified status.

Required tests and exit evidence:

- Exercise every contact alias alone and verify that it fails as incomplete; exercise valid configurations for all four model alternatives.
- Test multiple walls, mismatched cardinality, degrees/radians ambiguity, unknown keys, duplicate owners, and keys that are illegal for the selected model.
- Register two fitted free surfaces with different Nitsche policies in both orders and verify identical matrices and solutions. Verify that an unrelated weak velocity boundary retains its own policy.
- Snapshot-test the effective configuration and prove that no accepted option is parsed but unused.
- Close the work package only when unsupported combinations fail before assembly and configuration order has no numerical effect.

### WP-1: remove physical dry-domain feedback and bound the transport extension

**Findings addressed:** FSR-01, FSR-02, and FSR-18.

Implementation notes for the physical velocity:

- Remove `applyFreeSurfaceVelocityExtension` from the production physical momentum path. During transition, rename it as a legacy dry-velocity diffusion mechanism and fail closed when it is selected by a purportedly qualified case.
- Reinstate a precise inactive-velocity constraint policy. Constrain only degrees of freedom with no retained liquid support; never pin a wet-supported cut degree of freedom.
- If removal exposes small-cut conditioning, solve that problem in WP-7 with the selected consistent cut formulation. Do not reintroduce dry traction through a differently named coefficient.

Implementation notes for the auxiliary field `E`:

- Move extension-map construction out of `ApplicationDriver.cpp` into a testable component that produces an immutable map snapshot tied to mesh, level-set value, topology, ownership, geometry, and active-set revisions.
- Replace pivot-only stencil acceptance with rank-revealing QR/SVD or an equivalent condition estimate. Record rank, condition estimate, maximum coefficient, row sum, row `L1` norm, negative-weight count/magnitude, reproduction error, extrapolation distance, preview amplification, component assignment, and revision-to-revision map change.
- Use fixed guards for all of those quantities. A failed high-order reconstruction should take a bounded fallback, such as nonnegative inverse-distance or nearest-normal interpolation with row sum one; it should never accept an arbitrarily amplified row.
- Reproject `E` onto every newly accepted map. A warm start that violates the refreshed exact constraints must not be retained.
- Replace cell-node clique adjacency with actual mesh-edge adjacency or a bounded geometric-neighbor graph. Component IDs are bookkeeping only. Assign collision-zone vertices using nearest-interface or normal-geodesic geometry; use a documented bounded blend or fail on a true unresolved tie, never select the lower numeric component ID.
- Build matrix sparsity from the same admissible graph and maximum fallback stencil used by the map. Enforce the one-way dependency `u -> E -> phi`; no `E` row may enter physical Navier--Stokes momentum.

Required low-level tests:

- Compare the liquid-supported residual and Jacobian blocks while changing dry depth, exterior dry velocity values, legacy diffusivity, and disconnected dry paths. Define scaled norms such as `||Delta R_w||/(a_R + ||R_w||)` and `||Delta J_ww||_F/(a_J + ||J_ww||_F)`, with documented dimensional absolute floors; exclude pressure-gauge rows and unrelated global constraints. Candidate assembled-block gates are `1e-11` in serial and `1e-9` under MPI, subject to ratification before running. Compare solved fields separately against declared nonlinear, linear-solver, and reduction tolerances rather than requiring roundoff equality.
- Verify zero matrix coupling between two liquid islands through dry cells and unchanged manufactured liquid velocity after adding dry layers.
- Test constant reproduction, permitted affine reproduction, exact frozen-map Jacobian, wall-normal compatibility, and residual neutrality after same-state map refresh.
- Use collinear/coplanar, near-singular, thin-band, isolated-island, and approaching-component stencils. Sweep band widths of 1, 2, 4, and 8 cell layers and permute node/component numbering and MPI partitions.
- Verify that quadrilateral/hexahedral diagonals do not become graph edges. Translate two nearby but physically disconnected drops or sheets through a collision-band overlap and require no cross-component contamination before actual contact.
- Require row-sum error below the predeclared floating-point tolerance, bounded row `L1` norm, bounded `||E||/||u||`, no unapproved negative weights, and identical physical results after deliberate component relabeling. A convex fallback should have nonnegative coefficients and row `L1` no greater than one plus tolerance.

Simulation exits:

- Re-run wall advection, translating drop, reversible Enright deformation, and the two-island case, plus a reduced deterministic reproducer of the D38 map failure. The reproducer must eliminate the previously observed `10^2`--`10^6` extension amplification and map-refresh velocity spike. Full-horizon D18/D38 runs remain Q5 tests after Q0--Q4, where they must also be independent of unused dry-domain geometry.
- Preserve a machine-readable per-revision map report, including every fallback and guard rejection. Closure requires both the one-way wet-block result and the replacement cut-conditioning evidence from WP-7.

Qualification evidence recorded on 2026-07-20:

- The frozen `free_surface_wp1_extension_v1` matrix passed 53 distinct low-level tests across five serial/MPI groups at source commit `398a24773be4c2e757aa642ce642a029f8be1381`. The matrix includes the complete current cut-stability suite as the same-revision WP-7 dependency evidence.
- Serial and two-rank wet-block invariance, zero dry-path coupling between liquid islands, inactive-support constraints, legacy-path rejection, exact frozen-map coupling, wall compatibility, 1/2/4/8-layer bands, reversed node/component numbering, partition changes, and per-revision artifact publication all passed their predeclared gates.
- The reduced D38 stencil demonstrates an unguarded row `L1` norm above `1e5` and amplification above `1e5`; the accepted map instead takes a nonnegative bounded fallback, has wet-to-dry amplification no greater than one in that fixture, and rebuilds at the same state with zero preview change.
- The wall-film, separated-drop, translating-drop, and reversible-deformation exits passed. The archived `64^2` translating-drop and `32^3` Enright points are WP-1 exits only; they do not qualify the complete Q3 refinement matrix, and no full-horizon D18/D38 claim is made here.
- Machine-readable results, exact test lists, declared gates, source/build hashes, every phase-flux ledger row, resource records, and checksums are archived in [the WP-1 qualification record](qualification_logs/free_surface_wp1_extension_20260720_398a2477/record.md) and [summary](qualification_logs/free_surface_wp1_extension_20260720_398a2477/summary.json). The archive commit is `10c711061fd442977d80cccdb8bb63a0fc25abdf`.

### WP-2: one authoritative cut/interface/wet-wall/contact geometry snapshot

**Findings addressed:** FSR-13, FSR-14, FSR-15, and FSR-17; this work is also a prerequisite for FSR-03 through FSR-07 and FSR-09.

Implementation notes:

- Extend the current cut-integration context or introduce an equivalent immutable `FreeSurfaceGeometrySnapshot`. It should contain retained positive/negative volume quadrature, interface quadrature, sharply clipped positive/negative external-boundary quadrature, contact quadrature, source-fragment stable IDs, topology and component IDs, ownership, achieved order, pruning/aggregation status, reference and physical points, normals/conormals, pointwise `J` and `|det J|`, and all source revisions.
- Derive interface, wetted wall, and contact rules from the same cell decomposition. The contact line is the boundary of the wetted wall patch and must point back to its source interface fragment; it is not an independently discovered root set.
- In the scalar contact builder, use `interface_domain.fragments()` as the authoritative candidates. Any higher-accuracy wall root reconstruction must remain inside and verify correspondence with its parent fragment. Reject orphan contact roots and surface fragments that should meet a wall but have no corresponding contact rule.
- Compute physical phase volume by evaluating the same pointwise geometry map and `|det J(xi_q)|` used by assembly at every retained quadrature point. Test this through assembly of the constant-one form. Report an unpruned lifecycle volume and an assembled retained volume separately if both are needed.
- Pair high-order curvature values and coordinates at the same reference point: select `xi`, map it through the geometry mapping to `x(xi)`, and evaluate `phi(xi)`. Replace nodal-extrema screening with a certified polynomial bound/subdivision or an authoritative high-order cut-backend query.
- Add a single validator that checks every point, not only points marked `root_polished`: finite positive weights, parent-cell containment, declared phase sign, root residual, normal angular error, positive/negative volume partition, wet/dry boundary partition, surface/contact provenance, polynomial moments through claimed order, achieved-order policy, revision equality, and unique MPI ownership. Root and normal checks must use the geometry represented by that backend: for `LinearCorner`, test roots and normals of the linear interpolant; error relative to a higher-order FE field or analytic surface is a separate approximation metric.
- Keep the snapshot compact and cache by revision. Track cache size and peak resident memory; evict superseded snapshots only after no assembly or diagnostic consumer can reference them.

Required tests:

- Use affine and distorted Q1 quadrilaterals/hexahedra and curved Q2/Q3 cells. Compare phase volumes and surface moments against high-order reference integration; require constant-one diagnostic/assembly equality to quadrature tolerance.
- Sweep analytic planar, circular, and spherical cuts through vertices, edges, faces, near tangencies, and volume fractions from `1e-8` through `0.5`, including phase reversal and rotations.
- Inject off-interface roots, nearly orthogonal normals, outside-parent points, swapped phase labels, wrong moments, false achieved order, stale revisions, orphan contact rules, and duplicate MPI ownership. The validator must reject every injected defect.
- Exercise vertex/edge/tangent/aligned-zero wall intersections and translate/rotate a sessile cap through cell and MPI boundaries. Physical contact measure and position should converge continuously; source IDs may legitimately change at a cell or ownership crossing but their remapping must be deterministic and revision correct.
- Test warped high-order cells where a mapped reference centroid differs from an average of physical nodes and cases with interior roots missed by nodal extrema.
- Run distorted-mesh translating-drop/volume-correction tests and high-order static-drop and capillary-wave curvature studies. The correction target must equal assembled liquid measure, and curvature error must recover its expected refinement trend.

Exit evidence:

- A geometry ledger must show that every contact fragment references exactly one valid source-surface fragment, while allowing one surface fragment to have zero or multiple wall intersections. It must also show zero stale/orphan contact rules, quantitative maximum root/normal/moment errors, and constant-one volume equality.
- All consumers must declare the same complete snapshot revision key. A mismatch should be a hard error, not a warning or implicit refresh.

Qualification evidence recorded on 2026-07-22:

- The frozen `free_surface_wp2_geometry_v4` matrix passed from clean source commit `5cf65650f93faf8d6f4c264ca50d03c70daea373` and source tree `24b20591ab27156b5dc8a25ea1243d99810a04df`.
- All nine predeclared serial/MPI groups passed, covering 200 distinct tests and 130 quantitative evidence checks. The evidence includes affine, warped, and high-order geometry; tiny-cut and intersection sweeps; fault injection; contact provenance and ownership; revision consistency; constant-one assembly equality; curvature refinement; and snapshot lifecycle bounds.
- Application, FE, and Physics targets were configured and built in three independent clean build homes before execution. Final provenance bound the clean source state, registry hash `7e89ac445669a7f3e9b5217132949ecbe59f276ccd2e0326b12982fd26c28361`, runner hash `87152bb031ac5b4cbcafb71bb39cd02fb3c5ce893d6f093d39f60e3429881583`, and executed binaries; every final provenance check passed with no diagnostic.
- The checksum-bound [WP-2 qualification record](qualification_logs/free_surface_wp2_geometry_20260722_5cf65650/record.md), [summary](qualification_logs/free_surface_wp2_geometry_20260722_5cf65650/summary.json), and machine-readable artifacts are complete in cumulative commit `72e72410034f2a641fb5b8a5024d8eabbae8caf2`. That commit adds the 27 checksum-bound build and preflight records omitted by ignore rules from the initial artifact commit `88ce5b60d28d273ebbe45eaa219da0960646feb2`; a clean extraction contains all 70 archive files and all 69 recorded artifact checksums pass.
- This evidence closes WP-2 only. WP-3 through WP-10 and Q0 through Q7 remain separate gates and are not credited by this campaign.

### WP-3: sharply clipped exterior boundary operators

**Finding addressed:** FSR-16, with direct relevance to FSR-01 and FSR-05.

Implementation notes:

- Add a generated codimension-one integration domain for the intersection of every background exterior face with the active liquid phase. Each point needs parent cell/facet, reference and physical coordinates, physical normal, wet measure/Jacobian, active side, interface and wall markers, ownership, and the common geometry revision.
- Allow forms to request either the full physical boundary or an active-phase subset explicitly. When an unfitted active domain cuts a boundary, route traction, Robin, outflow, pressure-flux, weak Nitsche, wall slip, and any other natural/weak form through the sharp subset.
- Reject ambiguous multiple active domains and reject an implicit fallback to the whole background `.ds`. The current smoothed wet-wall indicator may remain only as a separately named regularized experimental model with its width stated in physical units and independently refined.
- Include cut-stable trace scaling for Nitsche and slip on arbitrarily small wet patches; qualify that scaling jointly with WP-7.

Required tests and exits:

- Integrate constants and polynomials over analytically half-wet, quarter-wet, and obliquely cut faces. A completely dry face must contribute exactly zero.
- Sweep wet fractions `1e-8`, `1e-6`, `1e-4`, `1e-2`, `0.1`, `0.25`, `0.49`, and one; test traction, Robin/outflow, symmetric and unsymmetric Nitsche, and wall slip under active-side reversal and MPI repartitioning.
- Run a manufactured channel while the interface crosses an inlet, outlet, and side wall. Force, flux, and penalty work must vary with the analytic wet measure. Compare left and right limits as a cut crosses a mesh vertex and require any global numerical jump to converge to zero with refinement.
- Closure requires that no generic weak or natural operator on a supported cut physical boundary retains whole-face integration.

Qualification evidence recorded on 2026-08-18:

- The frozen `free_surface_wp3_wp7_symmetric_nitsche_certified_trace_prerequisite_v2` matrix passed from clean source commit `e9ae9f8211ff8cac59bf9e128bfcd461ebeb7ff8` and source tree `9fa1617410d81f3801bb53c6fdeec6619a06fd9d`.
- All four predeclared serial/MPI groups passed, covering 26 distinct tests and eight quantitative evidence checks. The 108-case production diagnostic covered both active sides, axis-aligned and oblique cuts, three physical scales, and wet fractions from zero through one. Its maximum certified trace upper bound was `1.3865887291231187`, minimum finite-fixture energy lower bound was `0.6600749189008555`, and minimum sampled eigenvalue gap was `2.7395417703357516e-05`.
- FE and Physics targets were configured and built independently from clean caches. The complete 66-file archive is checksum bound in [the version-2 aggregate-trace qualification record](qualification_logs/free_surface_wp3_wp7_nitsche_coercivity_v2_20260818_e9ae9f82/record.md) and [summary](qualification_logs/free_surface_wp3_wp7_nitsche_coercivity_v2_20260818_e9ae9f82/summary.json).
- This evidence closes only the certified finite-dimensional aggregate-trace prerequisite in its declared affine P1, constant-viscosity envelope. The method-wide coercivity bound remains unfrozen; FSR-16, FSR-07, WP-3, WP-7, and Q1 remain open.

Accepted-state qualification evidence recorded on 2026-08-25:

- The frozen `free_surface_wp3_wp7_symmetric_nitsche_accepted_state_floor_prerequisite_v3` matrix passed from clean detached source commit `cb6cf91a090414eef020e3c30924b0b30570ed27` and source tree `2b29996981ee36b5e9e69f45e16d5fb1bd6ce04e`.
- All five predeclared serial/MPI groups passed, covering 33 distinct tests and eight quantitative evidence checks. The 108-case production diagnostic covered both active sides, axis-aligned and oblique cuts, three physical scales, and wet fractions from zero through one. Its maximum trace upper bound was `1.3865887291231187`, minimum accepted-state energy floor was `0.25`, and minimum sampled eigenvalue gap was `0.7499999999993009`. Exact quotient proofs covered 16 patches, with factorized proof flags, counts, work metrics, and the nonzero-input digest bound into the certificate digest.
- FE and Physics targets were configured, cleaned, and rebuilt in fresh external caches before execution. Final provenance bound the clean detached source, matrix hash `e1f2bcdc07daa40c9298244be57046c2353fa9c001f632b22407256818c5d381`, runner hash `242713d48ba40adb15b52c54c2fbe4afba8277db5c0e7f5cdc5b36fa219d1365`, focused-test hash `a711e17e11122fb3700f2cdd48aeefe7e795d1dff3172dfea57e05682be4314a`, and executed binaries. The untouched 70-file archive and all 69 recorded checksums pass in [the version-3 accepted-state qualification record](qualification_logs/free_surface_wp3_wp7_nitsche_coercivity_v3_20260824_cb6cf91a/record.md) and [summary](qualification_logs/free_surface_wp3_wp7_nitsche_coercivity_v3_20260824_cb6cf91a/summary.json).
- This evidence establishes only the predeclared `c*=1/4` floor for every accepted current state of the supported production Navier--Stokes viscous/Nitsche subform. The generic FE gate remains conditional on a caller-supplied coercive bulk form, unconditional cut- and mesh-family acceptance is not established, and FSR-16, FSR-07, WP-3, WP-7, and Q1 remain open.

Current handoff after the 2026-08-25 accepted-state qualification:

- The standalone `Physics_FreeSurfaceCutStability_MPI_2` prerequisite passed after a fresh Physics rebuild in Sherlock job `40669404` before the qualification run began.
- Sherlock job `40737447` completed the frozen V3 matrix successfully. The archive above was published unchanged in descendant commit `fec50ece980a5d5d811a4341a6f7800094106902`, and focused runner validation at that archive-bearing descendant passed all 72 tests.
- No formal closure status changed: FSR-16, FSR-07, WP-3, WP-7, and Q1 remain open, as do every other unchecked work package and qualification box.
- Resume with WP-3 closure: inventory every production natural and weak exterior-boundary operator, remove or reject any whole-face fallback, fill remaining production manufactured-channel, wet-fraction, and MPI coverage gaps, and only then freeze a WP-3 closure matrix. Treat the full WP-7 conditioning matrix as subsequent work rather than credit from this prerequisite.

WP-3 progress recorded on 2026-08-25 after that handoff:

- A source-level production inventory confirmed sharp generated-active-boundary routing for traction, Robin, pressure outflow, coupled RCR/RCRCR boundaries, weak Nitsche, dynamic-contact wall slip, and all four supported VMS/PSPG wall-boundary forms. The common routing helper selects the generated active subset when one is registered, and the Navier--Stokes marker lookup rejects ambiguous active-domain ownership.
- Focused production-assembly coverage was added in `Code/Source/solver/Physics/Tests/Unit/test_SharpBoundaryOperatorMPI.cpp` for the three previously uncovered VMS/PSPG wall forms: normal pressure gradient, tangential pressure gradient, and tangential momentum residual. The new serial and two-rank cases exercise both active sides; wet fractions `1e-8`, `1e-6`, `1e-4`, `1e-2`, `0.1`, `0.25`, `0.49`, and one; exact dry-face zero contribution; nonzero full-face work; analytic wet-measure/work proportionality; and residual, Jacobian, work, rule-count, and measure invariance under repartitioning. Existing focused coverage already exercises the fourth form, pressure flux.
- The incremental Physics rebuild succeeded in Sherlock job `40775275` on `sh02-10n30`, after which `PspgBoundaryPressureFluxUsesGeneratedWetWallMeasure` and `AdditionalPspgBoundaryFormsUseGeneratedWetWallMeasure` both passed in the 123.1-second serial run. That batch job then exited before MPI execution because its original request declared only one task slot. The corrected two-task job `40776364` completed `0:0` on `sh02-10n26`; `AdditionalPspgBoundaryFormsArePartitionIndependent` and the existing `OperatorWorkIsPartitionIndependent` both passed in 10.2 seconds. Durable scripts and logs, including the superseded launcher attempts, are under `/scratch/users/zsexton/wp3-boundary-dev-20260825/`.
- This is prerequisite coverage, not WP-3 closure. The next blocking evidence is a native, trace-certified production manufactured channel whose moving interface clips an inlet, outlet, and side wall and whose force, flux, and weak-Nitsche penalty work follow analytic wet measure through vertex crossings. After that, the complete WP-3 matrix still needs fresh independent builds, finalized hashes, frozen provenance, and serial/MPI execution. No work-package or qualification checkbox is credited by this entry.

WP-3 native manufactured-channel implementation checkpoint recorded on 2026-08-25:

- The current worktree adds `ApplicationDriverLevelSetWorkflows.NativeCertifiedManufacturedChannelTracksSharpBoundaryWork` in `Code/Source/solver/Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflows.cpp`. Its native three-dimensional tetrahedral channel moves a level-set interface across separately marked inlet, outlet, and side-wall faces. The harness routes inlet traction, outlet pressure, and symmetric weak-Nitsche velocity through generated active subsets; probes force, flux, and penalty work independently; and covers both active signs plus wet fractions zero, `1e-8`, `1e-6`, `1e-4`, `1e-2`, `0.1`, `0.25`, `0.49`, `0.5`, `0.51`, and one. The assertions require exact dry behavior, analytic wet-measure/work proportionality, active-side reversal, an aligned vertex crossing, and matching one-sided limits.
- The same harness requests an eager aggregate trace certificate and checks side-wall record ownership, generated-only routing, snapshot measures, nonzero certificate digest and revision data, and exact factorized proof flags, counts, work metrics, and input digest. Independent job `40780581` on `sh03-08n17` compiled `bin/test_application` successfully in `build-application-gcc12-vtk`; the job failed only because its first launcher used the cache root instead of the binary's `bin/` path. The corrected launcher is retained as `/scratch/users/zsexton/wp3-native-channel-20260825/build_and_test.sbatch`.
- The first focused execution exposed a fixture problem at the dry endpoint: the lower buffer had no fully active aggregate root. The channel now has an additional lower cell layer and anchors its aggregate at the new outer plane. The next execution reached the `1e-8` sample and exposed a production consistency gap: source geometry had authoritatively collapsed a negligible interface fragment to the dominant full-cell phase, while the active-boundary builder independently retained the discarded scalar cut. `GeneratedActiveBoundaryDomain.cpp` now uses the source domain's unique full-cell phase when no authoritative interface fragment remains, with a regression covering both phase signs.
- A subsequent execution exposed a second production consistency gap: snapshot volume retention pruned negligible active parent volumes while their exterior boundary fragments remained available to trace certification. Snapshot boundary records now inherit parent-volume retention, and `CutIntegrationContext` imports only retained fragments whose active volume side was imported. A regression covers a thin tetrahedral wedge whose boundary survives geometric clipping but whose negative parent volume is below the retention threshold.
- Further focused executions repaired the aligned-zero crossing rather than excluding it from the matrix. A caller-selected parent-side policy now publishes one complete aligned edge in two dimensions or face in three dimensions, is carried through application setup, lifecycle, restart, topology digesting, and debug output, and is covered for both active signs. Tetrahedral clipping no longer duplicates a generated cut face that is geometrically identical to a parent face. The authoritative snapshot contract continues to require both positive-volume regions for ordinary cuts, but narrowly accepts one aligned interface fragment paired with the selected full-cell region; that region is marked full-cell equivalent. Low-level tests cover the two- and three-dimensional aligned cases, both signs, the independent reference-subcell measure, accepted aligned snapshot families, and rejection of incomplete non-aligned families.
- Trace-certificate matching now uses the owner-filtered, retained authoritative snapshot records, and the application harness distinguishes raw generated-boundary rule counts from the rules retained after parent-volume pruning. The former fully wet exact-proof dimension overflow was localized to a full-active physical-boundary cell that had been attached to a larger aggregate solely because it lay in that aggregate's support. A full-active cell whose closed tangent masters all remain cell-local now receives an exact singleton proof patch backed by its retained volume block; any cell with a nonlocal master remains on the canonical aggregate. The fixed exact backend cap remains 32. The focused regression `FullActiveBoundaryInAggregateUsesClosedCellPatch` verifies the singleton patch, six raw and terminal degrees of freedom, three rigid modes, factorized proof metadata, and proof digest.
- The rebuilt FE targets passed all 12 `GeneratedBoundaryAggregateTraceCertificate.*` tests and all 16 `DenseLinearAlgebra.Exact*` tests. This includes the full-active local-patch regression plus coverage for strictly increasing sparse-map rows, malformed offsets, multi-term weights, and both multiplier paths. The exact backend now reports the requested output dimension and fixed cap when it rejects an oversized quotient.
- Independent Physics job `40785164` completed a fresh rebuild and passed both focused serial sharp-boundary tests. Its MPI launch was superseded because the request exposed only one task; a second launch was also superseded after the inherited export policy omitted runtime library paths. Corrected two-task job `40785847` completed successfully on `sh03-08n18`: `AdditionalPspgBoundaryFormsArePartitionIndependent` and `OperatorWorkIsPartitionIndependent` both passed. Durable scripts and logs for the successful and superseded runs are under `/scratch/users/zsexton/wp3-native-channel-20260825/`.
- The sign-reversed parent-25 inconsistency was traced to cancellation in the positive-side tetrahedral moments: the positive moments had been formed by subtracting the directly integrated negative moments from the parent, so a sufficiently small positive corner collapsed to zero while the corresponding negative-small orientation survived. The two- and three-dimensional builders now integrate both side geometries directly, preserve the smaller direct fraction, and form the larger fraction as its exact complement. Aligned selected-parent-side cases are handled separately with exact zero/one fractions so roundoff cannot create a ghost opposite-side region. `PreservesSmallPlanarCornerVolumeUnderSignReversal` and `PreservesSmallTetrahedralCornerVolumeUnderSignReversal` exercise both signs, both regions, quadrature measure, and reference subcells. `FreeSurfaceGeometrySnapshot.AcceptsParallelTetrahedralCutsUnderActiveSideReversal` covers both sides at thin, ordinary, and aligned cuts without weakening rejection of incomplete non-aligned families.
- An independent fresh FE geometry run in Sherlock job `40788909` passed all 266 selected tests from 28 suites. During that run, a stale completely-dry fixture that supplied scalar data but an empty authoritative interface domain was repaired to construct its positive full-cell source geometry. `RejectsBoundaryClippingWithoutAuthoritativeCellPhase` now explicitly preserves the fail-closed behavior for a genuinely incomplete source domain. The successful script and output are under `/scratch/users/zsexton/wp3-native-channel-20260825/`.
- The manufactured-channel harness now compares the mapped physical measures in retained snapshot rules with the analytic inlet, outlet, and side-wall areas; the previous comparison mixed reference-cell totals with physical areas. Per-sample traces and direct checks report physical parent measure, active measure, and operator work. With that correction, the pre-adjustment channel completes both active-sign sweeps and has exact side-reversal and vertex-limit agreement, but inlet traction work is absent at the thinnest wet strips. Diagnostics show that the inlet's leading wet triangle was supported only by cut-volume tetrahedra whose active fractions shrink quadratically or cubically and are pruned at the declared retention threshold, whereas the outlet and side-wall leading triangles have linearly shrinking retained support. The left half of the conforming channel fixture now uses the mirrored tetrahedralization so the inlet leading triangle also has linearly shrinking volume support; the right half retains the original orientation favorable to the outlet and side wall.
- Resume by rebuilding the Application test target with that latest conforming mirrored fixture and rerunning `NativeCertifiedManufacturedChannelTracksSharpBoundaryWork` for the complete two-sign wet-fraction sweep. If it passes, add and run its two-rank/repartition evidence, then rebuild FE, Application, and Physics independently, finalize matrix and runner hashes, and freeze provenance only after every prerequisite result is clean. No qualification runner has been launched, and no formal status changed: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open.

WP-3 exact-proof localization continuation recorded on 2026-08-25:

- A forced recompilation of the mirrored manufactured-channel fixture reached the corrected inlet topology and replaced the former thin-strip inlet-work failure with an exact trace-proof rejection: the canonical aggregate had 42 terminal tangent coordinates and six structural rigid modes, leaving quotient output dimension 36 against the fixed exact cap of 32. This confirmed that the mirrored boundary support repair was present and exposed a separate proof-locality requirement; the cap has not been increased and the test has not been weakened.
- The current worktree deterministically localizes a boundary proof when its canonical pre-quotient terminal dimension exceeds the smaller of the caller's requested reduced-dimension limit and the fixed exact-backend limit. Such a rule uses retained support from its boundary parent and declared aggregate root; a tangent-closed full-active parent continues to use its exact singleton support. Terminal master coordinates may remain external to those raw support cells because the retained cell-energy blocks are assembled after tangent substitution. Repeated root support remains charged through the certificate's existing overlap-weighted global bound. Canonical patches below the effective limit remain unchanged.
- The public exact dimension limit is now shared by the proof-selection and factorized backends. Trace-certificate digest version 5 binds each localized-proof flag and the certificate-wide localized count; validation checks the reported count, patch count, and localized/canonical-index correspondence before accepting the digest. The manufactured-channel harness records localized parent/root counts and the maximum factorized input dimension, requires at least one root-localized proof in this fixture, and requires every exact input dimension to remain within the fixed cap.
- The direct trace-certificate coverage now includes certificate-wide localized-count expectations, a metadata-consistent localized-to-canonical rewrite rejected by the stale digest, and homogeneous pinned tangent rows. The pinned-row regression uses a vector system constraint and requires an exact canonical rooted-aggregate certificate, proving that a zero tangent row is not mistaken for a missing row during dimension preflight. After the latest localization-threshold change and incremental rebuild, all 13 focused `GeneratedBoundaryAggregateTraceCertificate.*` tests and all 16 focused `DenseLinearAlgebra.Exact*` tests passed on the four-CPU `amarsden` development allocation `40773009`.
- Application job `40791679` completed after 5 minutes 14 seconds and failed because the first localization preflight treated a present homogeneous pinned tangent row as absent; that distinction is now fixed and directly covered. Replacement job `40793361` completed after 5 minutes 22 seconds and exposed a second preflight error: it assumed the full structural rigid-mode nullity would survive, while active-side pins removed those modes and left exact output dimension 36. The policy now uses pre-quotient terminal dimension against the effective fixed limit, independent of the rigid modes that later survive. Neither failure was bypassed by increasing the fixed limit or weakening the manufactured-channel assertions.
- Replacement Application job `40793991` rebuilt the current sources and advanced through the exact proof, then correctly rejected the configured penalty at wet fraction `0.49`: its localized certificate bound was `7.3799510101871197`, so gamma 12 gave ratio `0.61499591751559346` above the downward-safe `0.56249999999999978` cap for the declared quarter-energy floor. The production exception now retains the operator, grouped and route ratios, safe cap, certificate bound, effective penalty, and physical/generated markers. The manufactured channel uses the predeclared P1 penalty gamma 16; this changes neither its cut matrix nor any analytic work assertion.
- With gamma 16, the complete serial sweep passed both active signs and all 11 wet fractions in 7.3 seconds on development allocation `40773009`. Recorded maxima were trace bound `7.379951`, grouped ratio `0.461247`, support overlap two, four localized root patches, and factorized input dimension 15. The measured force, flux, penalty work, vertex limits, and active-side reversals agreed at the recorded zero-to-six-decimal scale, while the direct assertions retain their `5e-10` or tighter tolerances. Resume by adding and executing the native two-rank/repartition case, then rebuild FE, Application, and Physics independently, finalize matrix and runner hashes, and freeze provenance only after all prerequisite results are clean. No qualification runner has been launched, and no formal status changed: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open.

WP-3 native two-rank/repartition continuation recorded on 2026-08-25:

- The current worktree now contains the shared fixture `Code/Source/solver/Application/Tests/Unit/NativeManufacturedChannelMPIHarness.h` and the focused test `ApplicationDriverLevelSetWorkflowsMPI.NativeCertifiedManufacturedChannelIsRepartitionIndependent`. The test runs block and Metis ownership maps on exactly two ranks, proves that the ownership maps differ, and repeats both active signs and the complete 11-point wet-fraction sweep. It checks analytic inlet force, outlet flux, side-wall weak-Nitsche penalty work, exact dry behavior, vertex limits, side reversal, generated markers and rule counts, factorized-proof metadata, and certificate data across repartitioning.
- The distributed fixture now attaches its level-set field on local and ghost vertices before application setup. Its aggregation overlap is eight layers because the one-layer diagnostic correctly stopped when a canonical aggregate master was not locally relevant; the larger declared overlap satisfies the distributed aggregation contract without weakening that check. The Application MPI target compiles with these changes.
- A focused two-rank diagnostic on development allocation `40773009` completed both partition methods in about 15 seconds. All reported failures are confined to the cross-partition trace-certificate digest, global conservative upper bound, and grouped symmetric ratio assertions at lines 5264, 5266, and 5268 of the MPI test. The analytic measure/work, dry-state, active-side reversal, vertex-limit, generated-routing, rule-count, marker, proof-cap, and per-rank digest-consensus assertions reported no failure. One representative positive-side crossing has block bound `4.4646106920312647` and ratio `0.2790381682519541`, versus Metis bound `3.999999999999996` and ratio `0.24999999999999978`; certificate digests also differ across the partition methods.
- Resume by finding and removing the partition-dependent proof-support or aggregation choice rather than relaxing the equality gates. Rebuild and rerun this focused case locally, then run it as a proper two-task `amarsden` batch job. Only after the MPI case is clean should FE, Application, and Physics receive fresh independent rebuilds, finalized matrix and runner hashes, and frozen provenance. No qualification runner has been launched, and no formal status changed: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open.

WP-3 native two-rank exact-bound continuation recorded on 2026-08-25:

- The repartition diagnostic found identical physical patch roots, support cells, and tangent rows for the block and Metis decompositions. The factorized exact dyadic proof also returned the same directly proven bound, `4`, in both cases. The differing accepted bound came from padding that exact result with a floating Gershgorin diagnostic: an algebraic reordering left a tiny off-diagonal term in one decomposition and raised its published value to `4.4646106920312647`. The certificate now publishes the directly proven exact dyadic bound as `conservative_upper_bound`; floating eigenvalue and Gershgorin results remain diagnostic and do not control acceptance. Focused certificate tests require equality between the published and exact bounds.
- The whole-certificate digest is intentionally canonical only within one communicator and algebraic partition because it binds owner ranks and degree-of-freedom numbering. The two-rank test therefore continues to require a nonzero digest and rank consensus within each decomposition, but no longer treats different algebraic reorderings as the same byte stream. Across block and Metis it instead compares the partition-invariant structured evidence: physical roots and support, rule and proof counts, retained measures, support overlap, exact bound, quotient status, proof-input type, exact factorized flags, and invariant rank, dimension, nullity, block, row, and weight counts. Each factorized input digest must remain nonzero, but ordering-specific digests are not equated.
- After those corrections, `ApplicationDriverLevelSetWorkflowsMPI.NativeCertifiedManufacturedChannelIsRepartitionIndependent` passed on both ranks in 15.25 seconds on development allocation `40773009`. It exercised distinct block and Metis ownership maps, both active signs, all 11 wet fractions, exact dry behavior, physical work and measure identities, vertex limits, active-side reversal, and structured exact-proof invariants. The serial companion `ApplicationDriverLevelSetWorkflows.NativeCertifiedManufacturedChannelTracksSharpBoundaryWork` also passed in 6.90 seconds after the production change.
- The first scheduled FE job `40800386` built the wrong executable and therefore selected zero tests; that result was rejected rather than counted. Corrected job `40801041` built `test_fe_assembly`, `test_fe_math`, and `test_fe_assembly_mpi` and enforced explicit result counts. It passed all 13 certificate tests, all 16 exact-algebra tests, and all three two-rank certificate tests on each rank. Dependent job `40801049` then rebuilt and passed the native channel through a proper two-task scheduler launch on `sh03-08n20`. Both ranks ran one test, and the rank-zero summary recorded two partitions, two active sides, 11 wet fractions, maximum exact trace bound `7.37995`, maximum grouped ratio `0.461247`, maximum factorized input dimension 15, maximum support overlap two, zero partition measure difference, maximum partition work difference `2.22045e-16`, zero side-reversal work difference, and maximum vertex-limit mismatch `1.11022e-16`.
- Fresh independent Application and Physics caches are now rebuilding on the development and batch nodes without exceeding the two-node, 40-GB allocation ceiling. Resume by completing those regressions and a fresh FE cache, then finalize the closure matrix and runner hashes, create a clean frozen commit, and only afterward launch its qualification runner. No formal status changed: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open.
- Post-pass evidence tightening is now rebuilt and exercised. Floating Application result properties and the MPI summary use `max_digits10`. Because generated boundary-fragment stable identifiers bind ownership revision and local face numbering, the repartition comparison now translates each owned rule to the physical key `(canonical parent-cell GID, cell-local facet index)`, requires a complete unique mapping, and compares those sorted keys together with the structured factorized-proof evidence. Fresh Application MPI job `40803792` completed on `sh02-07n61`; both ranks passed the focused test. Its rank-zero summary recorded two partitions, two active sides, 11 wet fractions, exact trace-bound maximum `7.3799510101871197`, grouped-ratio maximum `0.46124693813669504`, maximum factorized input dimension 15, maximum support overlap two, zero partition-measure difference, work difference `2.2204460492503131e-16`, zero side-reversal work difference, and vertex-limit mismatch `1.1102230246251565e-16`.
- Fresh Application serial job `40803202` completed from cache `/scratch/users/zsexton/wp3-native-channel-20260825/fresh-application-dev-20260825` with 86 selected tests, zero failures/errors, 82 passes, and four explicit skips. The first fresh run exposed a stale provenance-test assumption: the system's mesh view was reference-frame bound, so a current-frame-only mutation correctly left its geometry revision unchanged. The test now proves that behavior and separately proves invalidation after a reference-frame mutation. Fresh Physics cache `/scratch/users/zsexton/wp3-native-channel-20260825/fresh-physics-20260825` produced 17 serial tests with zero failures/errors. The original job `40801362` was rejected as scheduler evidence because its wrapper incorrectly required the rank-zero test printer on the non-root rank even though both tasks returned zero; corrected two-rank job `40802872` completed on `sh03-08n20`, verified all three focused MPI tests on rank zero, the non-root diagnostic stream, and no failure marker.
- Fresh FE cache `/scratch/users/zsexton/wp3-native-channel-20260825/fresh-fe-dev-20260825` passed 53 `DenseLinearAlgebra.*` tests, 13 aggregate trace-certificate tests, 266 geometry tests, 296 LevelSet tests, and the Systems group with zero failures/errors. The Systems executable registered 593 tests: 583 ran, 575 passed, eight were explicitly skipped, and ten remained disabled by the existing suite configuration. Fresh two-rank job `40804464` completed on `sh03-08n20`; each rank ran and passed all three focused MPI certificate tests. Resume by finishing the production-diff review and freezing the prerequisite implementation commit with the required identity and content checks. Then create and freeze a new versioned closure matrix and runner with reciprocal hashes, use fresh detached worktrees and caches, and only afterward launch qualification. No qualification runner has been launched, and no checkbox or formal status changed: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open.
- The prerequisite production-diff review, staged-content checks, and author/committer checks are complete, and the implementation is frozen in the commit containing this checkpoint. Review confirmed that the fixed exact cap remains 32; the factorized sparse-quotient proof fields and counts are bound into certificate validation and digest version 5; caps are enforced before content scans; sparse-map rows are strictly increasing; and malformed-offset, multi-term-weight, and both multiplier-path regressions are present.
- Resume by creating exactly one new versioned WP-3 sharp-boundary closure matrix, runner, and focused contract test as the direct child of this implementation commit. Finalize their reciprocal hashes first, then execute only from a fresh detached worktree and new scratch caches. If every predeclared serial/MPI row passes and the evidence archive is frozen, reassess WP-3 and FSR-16 closure only within the declared affine P1/LinearCorner envelope; do not infer WP-7, Q1, higher-order, or uniform-conditioning closure. No qualification runner has been launched, and no checkbox or formal status changed: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open.

WP-3 closure-qualification freeze and launch checkpoint recorded on 2026-08-25:

- The reviewed prerequisite implementation is frozen in commit `6984b783d87fc56859ff55a06321b46663b68ab0`. Its fresh pre-freeze inventory found every one of the 80 declared selectors in the independently built FE, Application, and Physics executables. The exact matrix selections passed in direct serial and two-rank runs, including the native manufactured channel, all 13 aggregate trace-certificate tests, all 16 exact-algebra tests, aligned-policy lifecycle and restart coverage, and both block and Metis multiplier paths. All 85 declared serial numeric gates and all 70 declared MPI recorded-property checks passed.
- Direct child commit `c271810e22b16f8f13c5c90bb8f90fb5dd67b900` freezes exactly one version-3 closure matrix, its runner, and its focused contract test. The matrix declares 13 groups and 80 distinct tests, requires one `amarsden` node with eight tasks, one CPU per task, 20 GB, and build parallelism eight, and binds normalized matrix hash `5953c0a4fd5141d5a8d58e76f8a242a65c97497784ee5cf1ac421e31cb1aefa6`, runner hash `640a2cd920fce4309e5d625d2c9c5345ebffd741bb0a0dcd128cc73f0ebd37f7`, and focused-test hash `bf196e8f5db6cb2e3e73dce88d5b04e9bdfcd821dc0276e989d6c90ffb4b7934`. The focused suite passes all 16 tests, and frozen validation matches all 43 source blobs.
- Qualification job `40811096` was launched from clean detached scratch worktree `/scratch/users/zsexton/wp3-sharp-v3-c271810-worktree` at the frozen bundle commit, using new external FE, Physics, and Application caches and evidence root `/scratch/users/zsexton/wp3-sharp-v3-qualification-c271810/evidence-v3`. It is the only closure runner launched for this bundle and was still running when this checkpoint was written.
- This checkpoint records completed prerequisite, contract, and launch work only. No generated qualification result has yet been accepted or archived, so no checkbox or formal status changes here: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open. On completion, validate the checksum-bound output before reassessing only WP-3 and FSR-16 within the declared affine P1/LinearCorner envelope; WP-7, Q1, higher-order behavior, and uniform conditioning remain separate open gates.

WP-3 V3 qualification rejection and recovery checkpoint recorded on 2026-08-25:

- Job `40811096` rebuilt all three fresh caches successfully, recreated every declared binary, preserved clean source commit `c271810e22b16f8f13c5c90bb8f90fb5dd67b900` and tree `191f3418e4df3b654cb131178a63644c1a4d6315`, and matched every selector in the serial FE and Application binaries. It was then rejected before any qualification group ran because direct `--gtest_list_tests` launches of the MPI-initializing Physics, FE Assembly MPI, and Application MPI binaries inherited the eight-task scheduler world and each reached the frozen 60-second discovery cap. The checksum-bound failed scratch result reports `FAIL_METHOD`; it is not closure evidence and will not be archived as a passing record.
- Controlled eight-task `amarsden` job `40813380` retained the same module and runtime environment but placed each affected discovery command in an explicit one-rank MPI world. Physics, FE Assembly MPI, and Application MPI discovery then completed in `0.91`, `0.79`, and `1.74` seconds respectively, with zero stderr and exit code zero. This isolates launcher context rather than binary content or selector availability as the rejected preflight's cause.
- The recovery adds a shared fail-closed discovery helper: any binary selected by at least one distributed matrix group is listed through `mpiexec --oversubscribe -n 1`, while binaries used only by serial groups retain direct discovery. The helper rejects unknown keys, invalid ranks, duplicate binary paths, undeclared paths, nonpositive timeouts, and duplicate listed identifiers. Resume by freezing exactly one version-4 matrix, runner, and focused contract test that bind this helper and the unchanged production source inventory, then execute from another clean detached worktree and new caches. WP-3, FSR-16, WP-7, and Q1 remain open.

WP-3 V4 qualification freeze and clean-retry checkpoint recorded on 2026-08-25:

- Commit `5fa2bd550c6b3eb89eaaba517e7f051178487f45` freezes the fail-closed MPI-aware discovery recovery and its six passing focused tests. Its helper hash is `cdf0c84761d8b78989291859d1e073b15822ce98ee011325ca1dc49b2d1a0f3a`, and its focused-test hash is `e23f9157ff9ac3769143852811ea2efb5715336f1808f7be1b885a94d37c9fd8`. Discovery for Application MPI, FE Assembly MPI, and Physics is now isolated in explicit one-rank MPI worlds; the six serial-only executables retain direct discovery.
- Direct child commit `f31bddd390f75f01d96d366e41bee6a251428f98` freezes exactly one version-4 matrix, runner, and focused contract test. The bundle binds all 45 declared source items and the recovery contract; declares 13 groups, 80 distinct tests, 85 serial numeric gates, and 70 MPI recorded-property gates; and preserves the one-node, ten-task, 20-GB resource envelope. Its normalized matrix, runner, and focused-test hashes are respectively `d1307367a3deae5d0c1353adc68812022d0549beaf837bbf304c80df60d936c4`, `743900c0ecb654e192be4bbd3b25b15939ca9fe7871e92544a2ecd5f38ed187c`, and `3b8eba001d0aaacfa955b57a585b32502f57549d7adaf51c289dd4a1ef7da61b`. All 25 helper and V4 contract tests pass, and frozen validation reports all 45 source blobs matching with `PASS_FROZEN_VALIDATION`.
- Initial job `40814555` was rejected after 28 seconds, before cache rebuilds, evidence generation, or any qualification group, because the external launch preparation omitted the three bootstrap cache locators required by preflight. That empty setup attempt is not qualification evidence and remains separate from the clean retry. No bundle or production defect was found, so the frozen V4 inputs were not revised.
- Retry job `40815008` was launched on `amarsden` from clean detached worktree `/scratch/users/zsexton/wp3-sharp-v4-f31bddd-r2-worktree` at the exact frozen commit, with new root `/scratch/users/zsexton/wp3-sharp-v4-qualification-f31bddd-r2`, three independently bootstrapped cache locators, and evidence target `evidence-v4`. It retained the inherited module environment and stayed within the two-node, 40-GB aggregate ceiling. All three bootstrap configurations completed, and the locked runner had begun its own fresh configure/build sequence when this checkpoint was written.
- No checkbox or formal status changes with this checkpoint: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open. Resume by letting `40815008` finish, independently validating its checksum-bound evidence, confirming all declared build, discovery, test, numeric, MPI, provenance, cleanliness, and resource gates, and archiving an accepted record before reassessing only WP-3 and FSR-16 within the declared affine P1/LinearCorner envelope. WP-7, Q1, higher-order behavior, and uniform conditioning remain separate open gates.

WP-3 V4 terminal qualification-harness checkpoint recorded on 2026-08-25:

- Retry job `40815008` ended after `00:27:52` with scheduler exit `2:0` on one `amarsden` node using ten CPUs and 20 GB. The source worktree remained clean at exact bundle commit `f31bddd390f75f01d96d366e41bee6a251428f98`, and each of the three runner-owned fresh builds passed: Application built `test_application` and `test_application_mpi`; FE built the Assembly, Assembly MPI, Geometry, LevelSet, Math, and Systems test executables; and Physics built `test_physics`.
- The checksum-bound evidence at `/scratch/users/zsexton/wp3-sharp-v4-qualification-f31bddd-r2/evidence-v4` is internally intact, but it records `FAIL_METHOD` at `binary_link_provenance` before discovery, qualification groups, numeric gates, or MPI gates. The first `ldd` probe failed while mapping `libvtkFiltersParallelImaging-9.4.so.1` with `Cannot allocate memory`; no later binary-provenance records or group results were created.
- Root-cause inspection identifies a frozen harness defect: the inherited shared provenance launcher applies a hard 256-MiB `RLIMIT_AS` to `ldd`. That cap constrains virtual address space rather than resident memory and is too small for this Application/VTK dependency graph. The probe used only about 25 MiB peak resident memory, the runner observed at least 179034 MiB host memory available, and the complete scheduler step peaked at 8441900 KiB, so neither the 20-GB allocation nor the method exhausted memory. The frozen runner nevertheless classifies the provenance-process failure as `FAIL_METHOD`; this result is rejected as closure evidence rather than reinterpreted.
- Resume by replacing the provenance probe's fixed address-space cap with a predeclared, tested resource policy that accommodates the supported dynamic-library graph while retaining fail-closed wall-time, resident-memory, output-size, and allocation safeguards. Freeze a version-5 matrix, runner, and focused contract test only after reproducing the original failure and proving the corrected provenance path. Then use another clean detached worktree and fresh caches for exactly one qualification run. No checkbox or formal status changes here: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open, including FSR-16, WP-7, and Q1.

WP-3 V5 provenance recovery and qualification-launch checkpoint recorded on 2026-08-25:

- A controlled preflight reproduced the rejected V4 provenance probe under its 256-MiB virtual-address limit and passed the same Application/VTK dependency inspection under a 1024-MiB limit. The accepted policy retains the 60-second wall-time cap, a 1024-MiB sampled aggregate resident-memory cap, a 4-MiB output cap, and fail-closed process monitoring. The successful probe enumerated 180 dependency records, observed 4608 KiB peak aggregate resident memory and 17148 KiB reaped-process maximum resident memory, and stayed far below the 20-GB scheduler allocation.
- Direct child commit `a4bd2ea1a7ba38bd7c8fc10b571deeaca7c52333` freezes exactly one version-5 matrix, runner, and focused contract test. It preserves the V4 scientific matrix byte for byte and changes only the provenance resource policy. The raw matrix, normalized matrix, runner, and focused-test hashes are respectively `facfb41d9becfc535e6492bd45c68398cb0fae1d5a47b905524ea52b69fef39d`, `009784c169e67fa32c4b505918b392710d7f78ab7e9594ca8358dffaea2da985`, `2296b47589d40a57a439d42637ff107f65028d4a78eb7af58257eeccc0aebbcc`, and `3d6c66f740863af2c2bb8b9932d7930e6b29073f625e95b485d6d7ca9e49056a`. All 11 V5 contract tests and all 19 inherited V4 contract tests pass; frozen validation matches all 45 source items and still declares 13 groups, 80 distinct tests, 85 numeric gates, and 70 MPI property gates.
- Qualification job `40819989` was launched on `amarsden` from clean detached worktree `/scratch/users/zsexton/wp3-sharp-v5-a4bd2ea-worktree` at the exact frozen commit, with new root `/scratch/users/zsexton/wp3-sharp-v5-qualification-a4bd2ea`, three independently bootstrapped external caches, and evidence target `evidence-v5`. The bootstrap completed and the locked runner began its own fresh Application build. The qualification and development allocations together remain within two nodes, 14 CPUs, and 40 GB.
- This checkpoint records completed recovery, freeze, validation, bootstrap, and launch work only. Job `40819989` is still running, and no generated result has been accepted or archived. No checkbox or formal status changes here: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open. Resume by allowing the job to finish, independently validating every checksum-bound build, discovery, test, numeric, MPI, provenance, cleanliness, and resource gate, and archiving the accepted evidence before reassessing only WP-3 and FSR-16 within the affine P1/LinearCorner envelope. WP-7, Q1, higher-order behavior, and uniform conditioning remain separate open gates.

WP-3 V5 terminal serial-launch checkpoint recorded on 2026-08-25:

- Job `40819989` ended after `01:08:53` with scheduler state `FAILED` and exit `2:0` on one `amarsden` node using ten CPUs and 20 GB. Its 110-entry checksum manifest validates without error. All three fresh builds, clean-source checks, all nine binary-provenance records, final provenance, and 12 of 13 qualification groups passed. The sole rejected group was `sharp_boundary_operators_serial`; it reached its exact 2400-second envelope without producing its GoogleTest document, so the runner terminated the process session and recorded `FAIL_METHOD`, return code `-11`, 130956 KiB peak sampled resident memory, complete process-session coverage, and `wall_time_envelope_exceeded`. The complete failed record remains under `/scratch/users/zsexton/wp3-sharp-v5-qualification-a4bd2ea/evidence-v5` and is not accepted or archived as closure evidence.
- The same frozen Physics binary and exact 20-test filter passed directly on the development allocation in 151.479 seconds. A one-test controlled reproduction then isolated the scheduler-context defect: the focused DynamicContactAngle translation test passed in approximately 20 milliseconds under the same 8192-MiB address-space limit, but setting `SLURM_NPROCS=10` alone made it fail to complete within five seconds and setting `SLURM_NPROCS=1` restored the pass. Setting `SLURM_NTASKS=10` or `SLURM_TASKS_PER_NODE=10` alone did not reproduce the stall. The batch job inherited `SLURM_NPROCS=10` because its build allocation requested ten tasks, while the direct serial group launched an MPI-initializing Physics executable outside an explicit one-rank MPI world.
- V4 had already corrected the analogous discovery path by routing the Physics binary through `mpiexec --oversubscribe -n 1`; V5 preflight used that route successfully, but serial group execution still used `direct_serial`. Resume by extending the same declared MPI-single-rank classification to serial execution, recording the launcher and two-process monitoring contract, and adding focused rejection and positive-path tests for the inherited `SLURM_NPROCS` context. Only then freeze a version-6 bundle and run it from another clean detached worktree and new caches. Raising the 2400-second cap would conceal the launch defect and is not the recovery.
- No checkbox or formal status changes with this terminal result: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open, including FSR-16, WP-7, and Q1.

WP-3 V6 scheduler-safe serial-execution freeze and launch checkpoint recorded on 2026-08-26:

- Implementation commit `a98522eac0162085652e4750e8224193ca6e2742` adds a hash-locked GoogleTest execution helper and seven focused tests. The helper derives MPI-initializing binary keys from the distributed groups already frozen in the scientific matrix; routes serial groups for those binaries through `mpiexec --oversubscribe -n 1`; retains the direct path for non-MPI binaries and the inherited path for multi-rank groups; records the actual launcher route; requires MPI-mode monitoring with two simultaneous process samples; and fails closed if the parent launch, writer, binary inventory, launcher, or resource record drifts. The focused reproduction preserves `SLURM_NPROCS=10` and verifies the explicit one-rank route rather than weakening the 2400-second group envelope.
- Direct child commit `a73c77f44ac1741df730dc4102ac938b9b1b6922` freezes exactly one version-6 matrix, runner, and focused contract test. It preserves all 13 groups, 80 distinct tests, 85 numeric gates, 70 MPI property gates, and the V5 resource safeguards while extending the source inventory from 45 to 47 hash-bound items. The raw matrix, normalized matrix, runner, and focused-test hashes are respectively `597bb62a306991f78e87a697693fd485dbd3acf3a93273c289ff514eb5ccf7e5`, `c73d95d39eaae6415da4cce87bfd09c6b8a4268b6938d246e1cc926050c3ec39`, `0a2999e91a369bd2c77e1aa6f214895177fd32485d290f348a04a356e584f4ff`, and `a922d7c239aa3adab79b4d6c28185a4b02e7f525ac6576bb9d192ca5e2c7ef0d`. All 54 execution-helper, discovery-helper, inherited V4/V5, and V6 contract tests pass; frozen validation resolves the canonical bundle history, matches all 47 source items, and reports execution ready.
- Qualification job `40858374` was submitted to `amarsden` for one node, ten CPUs, and 20 GB from clean detached worktree `/scratch/users/zsexton/wp3-sharp-v6-a73c77f-worktree` at the exact frozen bundle. Its new root is `/scratch/users/zsexton/wp3-sharp-v6-qualification-a73c77f`, with three fresh cache paths and evidence target `evidence-v6`. At this checkpoint the job is pending and has not created qualification evidence. Together with the development allocation, the request remains within two nodes and 40 GB; the batch script uses the inherited module environment and contains no nested scheduler launch shell.
- No checkbox or formal status changes with this freeze and submission: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open. Resume by allowing job `40858374` to reach a terminal state, then independently verify its checksum manifest and every build, discovery, group, quantitative, MPI-property, provenance, cleanliness, execution-route, and resource gate. Archive only a complete passing record before reassessing WP-3 and FSR-16 within the affine P1/LinearCorner envelope; WP-7, Q1, higher-order behavior, and uniform conditioning remain separate open gates.

WP-3 V6 terminal qualification and independent-check checkpoint recorded on 2026-08-26:

- Qualification job `40858374` completed `0:0` after `00:32:04` on `sh02-07n61`, using one `amarsden` node, ten CPUs, 20 GB, and 8857628 KiB batch peak resident memory. The locked summary under `/scratch/users/zsexton/wp3-sharp-v6-qualification-a73c77f/evidence-v6` reports `PASS`: all 13 groups passed, all 80 distinct tests were covered, all 85 quantitative checks passed, all 70 MPI recorded-property checks passed, and final provenance produced no diagnostic. The previously stalled `sharp_boundary_operators_serial` group passed through the declared explicit one-rank MPI route in 163.334 seconds, with 2035 successful two-process samples and 265844 KiB peak aggregate resident memory.
- A separate read-only checksum verification covered all 111 entries in `checksums.txt`; every entry matched, and the evidence root contains 112 regular files including the manifest and no symbolic links. The verification transcript is `/scratch/users/zsexton/wp3-sharp-v6-qualification-a73c77f/independent-checksums-v6.txt`. The frozen detached worktree and evidence remain unchanged at bundle commit `a73c77f44ac1741df730dc4102ac938b9b1b6922`.
- This records a terminal passing runner result and a complete checksum verification, but not yet formal acceptance. Resume by independently checking the build and final-provenance records, all group resource and execution-route records, exact test-set union, quantitative and MPI-property totals, matrix and runner bindings, clean frozen worktree, and scheduler envelope. Only after those checks should the byte-preserved evidence be archived, the V6 implementation and bundle history be integrated, and WP-3 plus FSR-16 be reassessed within the declared affine P1/LinearCorner envelope. WP-7, Q1, higher-order behavior, and uniform conditioning remain open.
- No checkbox or formal status changes with this checkpoint: WP-0 through WP-2 remain complete; WP-3 through WP-10 and Q0 through Q7 remain open.

WP-3 V6 independent semantic-validation checkpoint recorded on 2026-08-26:

- A separate scratch-only verifier at `/scratch/users/zsexton/wp3-sharp-v6-qualification-a73c77f/verify_wp3_v6_evidence.py` completed a read-only reconstruction of the frozen V6 record. It rehashed the exact 111-entry manifest and 112-file regular-file set; confirmed the clean detached bundle commit, tree, parent, and changed-path contract; recomputed the raw-matrix, normalized-matrix, runner, and focused-test bindings; checked three configure/build sequences and nine binary-provenance records; parsed all serial and per-rank GoogleTest documents; reconstructed the exact test union; and recalculated every numeric and MPI recorded-property relation from the raw properties rather than trusting the runner summary.
- The independent result is `PASS`: 13 groups, 80 distinct tests, 85 quantitative relations, 70 MPI recorded-property relations, nine binaries, 111 matching checksums, and 112 regular files. It also confirms the group resource and execution routes, final provenance, common scope, closed-outcome scope, and the scheduler envelope. The durable validation transcript is `/scratch/users/zsexton/wp3-sharp-v6-qualification-a73c77f/independent-validation-v6.txt`; the separate scheduler transcript is `/scratch/users/zsexton/wp3-sharp-v6-qualification-a73c77f/independent-scheduler-v6.txt`.
- This completes independent semantic validation of the unchanged scratch record, but the byte-preserved archive and bundle-history integration are still pending. Resume by completing the repository content-policy scan over the checksum-bound bytes, copying the unchanged evidence into `Documentation/qualification_logs`, verifying the copied manifest and exact file set, and integrating commits `a98522eac0162085652e4750e8224193ca6e2742` and `a73c77f44ac1741df730dc4102ac938b9b1b6922`. Only after those steps should WP-3 and FSR-16 be reassessed within the declared affine P1/LinearCorner envelope. WP-7, Q1, higher-order behavior, and uniform conditioning remain open; no checkbox or formal status changes with this checkpoint.

WP-3 V6 byte-preserved archive checkpoint recorded on 2026-08-26:

- The unchanged accepted evidence now resides at `Documentation/qualification_logs/free_surface_wp3_sharp_boundary_v6_20260826_a73c77f4`. A recursive byte comparison against `/scratch/users/zsexton/wp3-sharp-v6-qualification-a73c77f/evidence-v6` reports no differences; all 111 checksum entries pass in the copied tree; and the archive contains exactly 112 regular files, no symbolic links, and 9683207 regular-file bytes.
- A separate whole-file content-policy scan reports zero occurrences in every restricted-token class across both the scratch evidence and repository copy. The independent semantic-verification transcript and scheduler transcript remain outside the checksum-bound archive at the paths recorded above; the archive itself retains only runner-produced V6 evidence.
- Bundle-history integration is the remaining WP-3 closure prerequisite. Resume by integrating the direct implementation/bundle chain ending at `a73c77f44ac1741df730dc4102ac938b9b1b6922`, verifying ancestry and repository cleanliness, then reconciling the WP-3 and FSR-16 status text strictly within the affine P1/LinearCorner scope declared by the passing matrix. WP-7, Q1, higher-order behavior, uniform conditioning, and all unrelated work packages remain open; no checkbox changes with this archive checkpoint alone.

WP-3 V6 history-integration and closure-review checkpoint recorded on 2026-08-26:

- Archive commit `a159c39c8fc48eff010b467ad6d16901c41087e0` preserves the accepted 112-file V6 evidence tree. Merge commit `55e5f901c4d58bbe647a3f1d6d3adb51af7a46e5` integrates the direct implementation and bundle chain ending at `a73c77f44ac1741df730dc4102ac938b9b1b6922`; both that bundle commit and implementation commit `a98522eac0162085652e4750e8224193ca6e2742` are verified ancestors of the current branch. After integration, all 111 archived manifest hashes still pass, the archive still contains exactly 112 regular files and no symbolic links, and the worktree was clean before this checkpoint edit.
- No formal status changes are taken by this integration record: WP-0 through WP-2 remain complete, while WP-3 through WP-10 and Q0 through Q7 remain open. Resume WP-3 at the final descendant-delta review: compare the 47 hash-bound implementation-source items frozen at the V6 bundle against the current tree, classify every changed item, and confirm that no post-qualification change invalidates a qualified WP-3 production route. Reuse current-tree regression evidence only where it exercises the exact affected route; otherwise run the focused replacement before changing status text.
- If that review is clean, reconcile the FSR-16 finding row and section, the WP-3 checklist entry, and the traceability row only within the declared one-phase affine C0 P1/LinearCorner envelope. Keep WP-7, Q1, higher-order behavior, uniform conditioning, WP-4 through WP-10, and all other qualification boxes open. The separately recorded two-rank flat-equilibrium work remains WP-4 development evidence and does not receive checklist credit.

WP-3 formal scoped closure recorded on 2026-08-26:

- The accepted V6 archive passes all 13 predeclared groups, 80 distinct tests, 85 quantitative checks, and 70 distributed recorded-property checks. Its 111-entry manifest, independent semantic reconstruction, clean frozen source and build provenance, exact declared test union, execution routes, and scheduler/resource envelope all pass. The matrix disposition explicitly closes FSR-16 and WP-3 only within the one-phase affine C0 P1/LinearCorner exterior-boundary envelope.
- The final descendant-delta review compared all 47 hash-bound implementation-source items with the current tree: 45 remain byte-identical, and only the serial and distributed Application test sources differ. Their qualified manufactured-channel and multi-domain-refresh test bodies are byte-unchanged; their intervening edits add the separately scoped WP-4 flat-equilibrium fixtures and repair an adjacent maintenance fixture. The only other post-qualification source changes are the WP-4 constant-pressure zero-load normalization and its focused test; no qualified WP-3 production source changed.
- After rebuilding both Application test targets from the current checkout under the inherited GCC 12/OpenMPI module environment, the exact two-test serial qualification group passed with zero failures in 7.2 seconds and the exact two-rank native-channel group passed on both ranks in 15.0 seconds. The distributed rerun retained two partitions, both active sides, 11 wet fractions, eight overlap layers, exact-proof input dimension 15, maximum trace bound `7.3799510101871197`, maximum trace ratio `0.46124693813669504`, zero partition-measure difference, `2.2204460492503131e-16` maximum partition-work difference, zero side-reversal work difference, and `1.1102230246251565e-16` maximum vertex-limit mismatch. Current-tree records are under `/scratch/users/zsexton/wp3-descendant-review-20260826`.
- The separate read-only verifier in that scratch directory reports `PASS`: 47 inventory items with exactly the two expected test-source deltas, four expected post-qualification code paths, three byte-identical qualified test bodies, 112 archive files, 111 matching archive hashes, two passing serial tests, one passing test on each of two ranks, and all 14 distributed property gates satisfied. Its transcript is `independent-descendant-validation.txt`.
- FSR-16 and WP-3 are therefore closed within that declared envelope, and the WP-3 checklist box is checked. Higher-order exterior restriction remains unsupported and fails closed. WP-7, Q1, uniform cut conditioning, WP-4 through WP-10, Q0, and Q2 through Q7 remain open; this closure supplies no credit to those gates.

### WP-4: balanced capillary pressure, wall energy, and prescribed angle

**Findings addressed:** FSR-03 and FSR-04.

Implementation notes:

- Make the distance of the capillary load to the discrete pressure-gradient range a required physical diagnostic, not just an LSQR solver-convergence statistic.
- Define one discrete functional that owns liquid--gas surface area `A_lg,h`, solid--liquid wetted area `A_sl,h`, and liquid volume `V_h`. Derive capillary surface work, Young wall work, and constant equilibrium-pressure work from variations of those exact discrete quantities and the authoritative geometry snapshot.
- Verify finite-difference directional derivatives of `gamma A_lg,h - gamma cos(theta_e) A_sl,h` and `lambda V_h`, where `lambda` is the constant volume-constraint/equilibrium-pressure multiplier, before selecting a balanced-force formulation. Verify a general FE pressure field separately against the corresponding discrete divergence/domain-variation identity; it is not the variation of the scalar product `p V_h`. Do not merely project away the component of surface force that pressure cannot represent, because that could suppress real moving-interface dynamics.
- Evaluate the AD-2 alternatives on the same static and moving tests. The selected method may require a compatible curvature projection, pressure enrichment/lifting, trace/XFEM pressure, or a different surface representation. Record why the chosen pressure and velocity traces can represent the needed equilibrium.
- Initialize a discrete static cap by constrained minimization of the same discrete surface/wall energy at the same discrete volume. Separately measure convergence of that discrete equilibrium to the analytic cap; do not expect sampled analytic geometry to be an exact algebraic equilibrium on every coarse mesh.
- Retire the literal unscaled codimension-two prescribed-angle penalty as a qualified path. Implement the wall condition `grad(phi) dot n_w = -cos(theta) |grad(phi)|` through a derived constrained update, ghost extension, or consistently scaled Nitsche/geometric treatment owned by the level set.
- Preserve a clear division of labor: momentum-side Young wall energy supplies physical force, while level-set wall geometry and redistancing preserve the requested interface orientation. Show that the two do not impose contradictory work.

WP-4 physical flat-equilibrium implementation checkpoint recorded on 2026-08-25:

- The constant-pressure KKT diagnostic now handles the zero-load limit without forming an order-one ratio from two roundoff-scale norms. It reports the scaled distance only when the capillary-load norm exceeds the coefficient-scaled roundoff floor; below that floor, a roundoff-scale residual maps to zero relative distance while the independent absolute-residual gate remains in force. The focused `NewtonSolver.*ConstantPressureKkt*` run executed six tests, including the new positive, nonzero roundoff-load regression, with zero failures.
- `ApplicationDriverLevelSetWorkflows.StaticCapillaryInitializationAcceptsPhysicalFlatSurfaceStressEquilibrium` exercises the production Application initializer, Navier--Stokes module, generated LinearCorner interface, negative CutVolume side, SurfaceStress form, two vertical dynamic-contact walls at 90 degrees, and the FSILS backend on a four-triangle fan mesh. The exactly horizontal interface retained liquid volume `1.5`, liquid--gas surface energy `3`, zero Young wall energy, and zero level-set update. Its constant-pressure KKT residual was `2.2204460492503131e-16`, relative distance was zero, pressure-jump absolute error was `1.3866695599588098e-32`, volume error was `2.2204460492503131e-16`, and surface-energy error was zero.
- Fresh Eigen-enabled FE job `40818626` completed on one `amarsden` node using eight CPUs and 20 GB, rebuilt the FE test executables, and passed the two existing physical-KKT rejection tests. Its first Newton filter was mistakenly directed to `test_fe_systems` and matched zero tests, so that invocation is not counted as evidence. The correct `test_fe_timestepping` target was then built in the inherited development allocation and all six selected KKT tests passed. The focused physical Application test passed separately; the adjacent static-capillary Application filter had five selected tests, with the parser and physical case passing and three Eigen-dependent cases skipped in that Eigen-disabled Application cache.
- This checkpoint proves one serial two-dimensional, negative-side, horizontal-interface, two-wall, 90-degree flat equilibrium and the corrected roundoff normalization only. It does not cover the required coordinate directions, wall orientations, phase signs, gravity directions, cut offsets, pressure gauges, MPI decompositions, three-dimensional cases, curved equilibria, other contact angles, convergence, or energy-variation checks. WP-4 and Q2 therefore remain open and no checklist item changes here. Resume by expanding the exact flat-equilibrium matrix before freezing any WP-4 qualification thresholds.

WP-4 physical flat-equilibrium matrix continuation recorded on 2026-08-26:

- The production-driver regression now covers both two-dimensional coordinate-normal directions, both corresponding contact-wall orientation families, both level-set active signs, and normal offsets `0.35`, `0.50`, and `0.65`. All 12 combinations use the generated LinearCorner interface, CutVolume active domain, SurfaceStress form, two 90-degree dynamic-contact walls, zero gravity, a free constant-pressure mode, and one explicit MPI rank. The sign-reversed cases retain the same physical liquid region, so every case checks the exact liquid volume `3 times offset`, liquid--gas surface energy `3`, zero Young wall energy, and an unchanged level-set field.
- The rebuilt external Application development target passed the 12-case test in 0.587 seconds through `mpiexec --oversubscribe -n 1`. Across the matrix, the maximum constant-pressure KKT residual was `2.2204460492503131e-16`, maximum relative distance was zero, maximum pressure-jump absolute error was `1.3866695599588098e-32`, maximum volume error was `2.2204460492503131e-16`, maximum surface-energy error was zero, and maximum level-set update was zero. The adjacent five-test static-capillary filter also completed without failures: the parser and physical matrix passed, while the same three Eigen-dependent cases remained explicitly skipped in this Eigen-disabled Application cache.
- This is a verified serial flat-interface expansion, not a frozen WP-4 qualification. Nonzero gravity remains rejected because the discrete equilibrium functional does not yet include gravitational potential and matching body-force work; a fixed pressure gauge does not retain the free constant mode required by this certificate; and two-rank equivalence, three-dimensional cases, curved equilibria, non-right contact angles, convergence, and energy-variation checks remain unimplemented. WP-4 and Q2 remain open, and no checklist item changes here.
- Resume first by taking WP-3 job `40858374` to a terminal state and independently accepting or rejecting its V6 evidence. For WP-4, next add the missing gravity functional/body-force identity and a certificate valid under fixed pressure gauges, then extend the exact flat matrix to MPI before freezing thresholds. Do not infer WP-4 or Q2 closure from the 12 serial cases.

WP-4 physical flat-equilibrium MPI implementation checkpoint recorded on 2026-08-26:

- The current worktree adds `ApplicationDriverLevelSetWorkflowsMPI.PhysicalFlatCapillaryEquilibriumMatchesAcrossTwoRankPartition`. It runs the same 12 physical cases on an explicit two-rank block partition with a ghost layer: two coordinate-normal directions, two contact-wall orientation families, both active signs, and offsets `0.35`, `0.50`, and `0.65`. Every case uses the production Navier--Stokes module, generated LinearCorner interface, CutVolume active domain, SurfaceStress form, two 90-degree dynamic-contact walls, FSILS backend, zero gravity, and a free constant-pressure mode. It also requires rank agreement for certificate metrics and the final algebraic revision.
- The new test exposed drift in the adjacent synthetic MPI publication fixture after the active-volume initializer contract was hardened. That fixture now uses a two-component vector velocity field, declares the zero-gravity active-volume energy state with density one, and gives FSILS four degrees of freedom per node. After rebuilding `test_application_mpi`, a two-rank filter containing both the repaired publication test and the new physical test passed both tests on both ranks in approximately 0.783 seconds. The durable log is `/scratch/users/zsexton/wp4-physical-flat-mpi-adjacent-20260826-r2.log`.
- Across the physical two-rank matrix, the maximum constant-pressure KKT residual was `2.2204460492503131e-16`, maximum relative distance was zero, maximum pressure-jump absolute error was `1.1638931388506733e-32`, maximum volume error was `2.2204460492503131e-16`, maximum surface-energy error was zero, and maximum level-set update was zero.
- This MPI work is focused development evidence and is not yet frozen. Resume by naming the recorded partition quantity as a partition-layout count, checking the complete diff, rebuilding as needed, and running the full Application MPI executable on two ranks before freezing this implementation. It still covers only one partition layout, two dimensions, zero gravity, a free pressure gauge, right-angle contact, and flat interfaces. WP-4 and Q2 remain open, with no checklist change.

WP-4 MPI regression-repair continuation recorded on 2026-08-26:

- The physical two-rank fixture now records its partition quantity as `wp4_physical_flat_mpi_partition_layout_count`. A rebuilt full two-rank Application MPI run executed 25 tests per rank: the new 12-case physical equilibrium test passed, 22 other tests passed, and the existing four-rank-only case skipped as declared. The sole failure was the adjacent `ActiveCutRefreshUsesCommunicatorWideSortedBoundaryMarkerUnion` maintenance fixture, so this is not yet a clean full-executable result. The run is preserved at `/scratch/users/zsexton/wp4-application-mpi-full-20260826-r2.log`.
- The adjacent fixture has been advanced through two hardened production contracts without weakening them. It now uses the accepted endpoint stage time and weight, registers the production-style equation and mass kernels, enables backend-row ownership, primes the distributed equation sparsity through FSILS, and allocates its time history only after that distributed backend layout exists. The MPI target rebuilt successfully after these changes.
- The next focused two-rank run reaches the candidate refresh and rollback checks, then fails because its gathered accepted endpoint is rank-local while the fixture still compares and hashes it as a replicated FE vector. The resulting rank-dependent endpoint values trigger the expected collective algebraic-revision rejection. The failing transcript is `/scratch/users/zsexton/wp4-application-mpi-boundary-union-20260826-r3.log`. Resume by making this fixture's endpoint reconstruction and revision input ownership-aware, preserving its rollback and marker-union assertions, then rerun the focused test and the complete two-rank executable. WP-4 and Q2 remain open, and no checklist item changes here.

WP-4 distributed flat-equilibrium verification checkpoint recorded on 2026-08-26:

- Both rank-local gathers in the adjacent maintenance fixture now use the production owner-certified collective capture routine. The focused regression passes on both ranks while retaining its exact owner-cover, marker-union, candidate-refresh, rollback, contact-stage, and functional-publication checks.
- Review of the new physical test found that merely constructing an FSILS factory before allocating `TimeHistory` did not prove a distributed algebraic layout. The fixture now primes FSILS from the distributed `equations` sparsity, requires each rank to own a strict subset of rows, requires the ownership union to cover the global vector exactly once, and reconstructs the certified solution through the same owner-certified collective path. The strengthened physical test and repaired maintenance test both pass on both ranks; the focused transcript is `/scratch/users/zsexton/wp4-physical-flat-mpi-owned-layout-20260826-r6.log`. Separate per-rank GoogleTest documents under `/scratch/users/zsexton/wp4-physical-flat-mpi-json-20260826` agree exactly on all recorded values: two ranks, one partition layout, two coordinate directions, two wall orientations, two active sides, three cut offsets, 12 cases, maximum KKT residual `2.2204460492503131e-16`, zero relative distance, pressure-jump error `1.1638931388506733e-32`, volume error `2.2204460492503131e-16`, zero surface-energy error, and zero level-set update.
- The complete rebuilt two-rank Application MPI executable then passed 24 of 25 tests on each rank in approximately 15.4 seconds, with only the explicitly four-rank-only consensus test skipped and no failures. The durable transcript is `/scratch/users/zsexton/wp4-application-mpi-full-20260826-r7.log`. This freezes a genuine two-rank, one-partition-layout extension of the 12-case flat zero-gravity/free-gauge matrix, not WP-4 qualification. Three-dimensional cases, nonzero gravity, fixed pressure gauges, additional decompositions and numbering permutations, curved equilibria, non-right contact angles, convergence, and energy-variation checks remain open. WP-4 and Q2 remain unchecked.

WP-4 gravity and constrained-pressure implementation handoff recorded on 2026-08-26:

- The current uncommitted worktree extends the static functional from surface-plus-wall energy to physical potential energy by adding the active-volume gravitational term. The production Navier--Stokes route now declares whether its static conservative body-force description is complete and fails the static initializer closed when an undeclared prescribed body-force field or spacetime source is present. Constant gravity is retained in the same active-volume Galerkin form used by the production momentum residual.
- The pressure diagnostic is being generalized from the constant-pressure KKT special case to the full admissible pressure space. Its operator split now records pressure work, surface-plus-wall work, gravitational work, total physical-potential work, and their conservative balance. The LSQR certificate uses the total physical-potential load, records its absolute residual and relative distance, and allows a constrained pressure gauge to omit the constant multiplier only when the general pressure-space certificate and the sampled production residual pass their independent gates.
- The Application schedule, canonical configuration, declaration hashing, minimizer objective, publication checks, and telemetry have corresponding fields for pressure representability and physical-equilibrium residuals. Focused regressions have been added for gravitational energy/material-power variation, gravity-sensitive pressure representability, a constrained synthetic minimizer certificate, production constant-body-force/operator agreement, fail-closed nonconstant source declarations, and a production nonzero-gravity equilibrium with a fixed zero pressure gauge; adjacent serial and MPI fixtures have been updated to the expanded diagnostic contract.
- This is a handoff record, not validation evidence. A fresh coherent FE rebuild has started but has not completed, and the new focused tests, Physics and Application rebuilds, full regressions, and fresh independent Sherlock runs remain outstanding. Resume by completing the FE build and focused tests, repairing any findings, rebuilding and testing FE, Physics, and Application independently within the declared scheduler envelope, reviewing the complete diff and content policy, and only then freezing a commit and reassessing further WP-4 evidence. WP-4 and Q2 remain unchecked, and no formal status changes with this handoff.

Focused gravity/constrained-pressure checkpoint later on 2026-08-26:

- Review of the one-shot compatible-pressure path found that adding the complete representability solution to a nonzero entry pressure could count an already present exterior or hydrostatic component twice. The initializer now solves a second mixed-pair LSQR problem for the remaining conservative-balance residual and adds only that correction. Its pair-layout correction load is allocated and refreshed with the other representability vectors; layout mismatch and replacement-sparsity regressions cover the new workspace member. An exactly compatible hydrostatic field is required to receive a roundoff-sized zero increment.
- A coherent rebuild passed all 23 focused FE pressure-representability and compatible-initializer tests. The production two-dimensional hydrostatic/fixed-zero-gauge Application regression passed both gravity directions with maximum exact-field production residual `3.3291811399467674e-16`, maximum computed production residual `2.7715350472406518e-11`, maximum pressure-space relative distance `1.1957757577740974e-10`, maximum gravitational-energy error `1.1102230246251565e-16`, and exact-compatible initializer pressure update `0`; its machine-readable result is `/scratch/users/zsexton/wp4-gravity-pressure-20260826/application-pressure-correction-dev-r17.json`. The focused FE result is `/scratch/users/zsexton/wp4-gravity-pressure-20260826/fe-pressure-correction-dev-r20.json`.
- The rebuilt physical flat-equilibrium Application test also passed on both ranks of a two-rank run after a coherent workspace-layout rebuild. Full FE, Application, and Physics regressions and the long serial/two-rank/four-rank Physics cut-stability lanes are still required before this prerequisite checkpoint can be frozen as complete evidence. This checkpoint changes no formal status: WP-4 and Q2 remain unchecked, all other open work packages and qualification items remain open, and no qualification runner has been launched.

Required tests:

- Flat interfaces for every coordinate direction, wall orientation, phase sign, gravity direction, cut offset, pressure-gauge treatment, and MPI decomposition. At exactly representable fields, the assembled compatible pressure/external-pressure residual should balance at scaled roundoff. A computed solution is instead gated by its declared nonlinear and linear-solver tolerances.
- Closed circles/spheres and sessile caps at `30`, `60`, `90`, `120`, and `150` degrees, rotated to every wall. Report pressure jump, best pressure-space residual, capillary residual, parasitic capillary number, kinetic energy, volume, base radius, apex height, and angle.
- Positive rescaling of `phi` must not change prescribed-angle enforcement. Test dimensional consistency and independent `h`, `dt`, and reinitialization-cadence refinement in 2D and 3D.
- Add one-phase adaptations of the Groß--Reusken planar/spherical force-discretization tests and Reusken sessile/spreading cases already identified in the literature review. Their original two-phase/XFEM ingredients must not be presented as fully reproduced before WP-10.

Candidate gates to ratify before qualification:

- exact-field flat assembly balance at scaled floating-point tolerance, with solved-state residuals bounded by the declared solver tolerances;
- a stable asymptotic convergence envelope, expected rate, and GCI for best-pressure-space and force residuals over cut offsets; a mildly nonmonotone pair triggers another level rather than automatic acceptance or failure;
- on the finest qualified static-cap level, pressure-jump error below `1%`, contact-angle error below `1 degree`, base-radius/height error below `1%`, and parasitic capillary number below `1e-6`;
- finite-difference energy-variation error at the expected discretization/roundoff rate.

These values are starting proposals, not retrospective pass criteria. If feasibility studies show that a threshold is incompatible with the chosen formal order, revise it with a documented derivation before the qualification matrix, then run the entire matrix unchanged.

### WP-5: side-wall contact-line dynamics and wall-aware maintenance

**Finding addressed:** FSR-05. Final qualification depends on closing FSR-01 through FSR-04, FSR-06, FSR-07, FSR-09, FSR-15, and FSR-16 so contact parameters are not compensating for extension, transport, cut, force-balance, geometry, or boundary-domain defects.

Implementation notes:

- Express the contact model as three coupled but distinguishable pieces: momentum-side surface/wall energy, positive line and wall dissipation, and physical contact-point kinematics. Record each piece in the energy ledger.
- Integrate Navier slip and Ren--E contact-line terms on the sharp domains from WP-2/WP-3. Keep physical slip length, line mobility/friction, any numerical wall width, and mesh size independent and visible.
- Evaluate contact position, dynamic angle, line velocity, wall slip, and constitutive residual at the identical accepted time, generalized-alpha stage, state revision, and geometry revision. Store both advancing/receding convention and wall frame in each record.
- Make reinitialization wall aware. Prescribed-angle reinitialization should preserve the prescribed wall condition; dynamic-angle reinitialization should preserve the accepted stage-consistent crossing and dynamic angle rather than resetting to an unrelated equilibrium angle.
- Implement distributed wall-aware reinitialization with deterministic contact ownership, halo synchronization, global convergence/rejection, and serial/MPI equivalence. Until that path passes, moving-wetting qualification is serial-only and MPI capability claims must fail closed. Do not expose Hamilton--Jacobi, fast-marching, or other reinitialization method names in the supported schema unless that implementation actually exists and is qualified.
- Treat global volume shift as an emergency operation. Before accepting it, report the induced contact-line displacement, angle change, per-component volume transfer, and numerical work.

Required sign and dissipation tests:

- Reverse advancing/receding motion, wall normal, interface side, gravity, and wall orientation. The Ren--E residual and line speed must transform according to the declared convention.
- Verify nonnegative wall-slip and line-friction dissipation and exact agreement between sharp wall/contact measure and analytic partially wet faces.
- Run no-motion wedge/arc cases through reinitialization and correction. Contact position and angle must remain within predeclared error at every substage.

Qualification simulations:

- Repeat the current four advancing/receding side-wall cases on at least three meshes and three independent time steps. Every speed sign must be correct; the constitutive error and stage mismatch must converge.
- Run sessile-cap relaxation for five equilibrium angles on bottom and side walls, Reusken spreading/contracting drops, resolved-slip cases from Sprittles--Shikhmurzaev, and the public Gründing et al. capillary-rise benchmark.
- Resolve each fixed physical slip length on successively refined meshes corresponding to approximately `l_s/dx = 2, 4, 8`; sweep wall regularization width `delta/dx = 0, 0.5, 1, 2` only where the regularized model is deliberately tested; and limit accepted contact-line motion to approximately `0.2`, `0.1`, and `0.05` cell per step in the time study.

Candidate physical gates:

- correct speed sign in every orientation and stage;
- finest-level Ren--E line-speed/constitutive error below `5%` with a convergent trend;
- equilibrium angle within `1 degree` and equilibrium base radius or rise height within `1%` or the published uncertainty, whichever is larger;
- capillary-rise height, overshoot, and damping within the reported experimental/intercode uncertainty;
- monotone energy decay in closed relaxation and no unexplained positive line/slip work;
- negligible dependence on wall orientation, MPI partition, and an independently vanishing numerical smoothing width.

### WP-6: locally conservative interface transport and maintenance transaction

**Finding addressed:** FSR-06.

Implementation notes:

- Implement the phase-measure-conservative strategy selected in AD-4. The conserved unknown must be stated explicitly; transporting a quantity named a conservative level set is not by itself evidence that liquid volume is conserved. If continuous FE is retained, replace the present nonconservative limiter with a conservative invariant-domain/AFC/FCT construction. A DG/FV or coupled level-set/volume-fraction method must likewise expose its phase fluxes and geometry reconciliation.
- Add the flux ledger appropriate to the selected discretization: cell-face fluxes for DG/FV, or algebraic-edge/nodal-control-volume fluxes for AFC/FCT. Interior contributions must cancel exactly and the external sum must match the physical boundary flux. Track the local balance of the chosen phase indicator/volume, each connected liquid component, wall film, sheet, rim, and resolved satellite.
- Use a divergence-compatible advecting field or explicitly account for discrete divergence in the conserved phase balance.
- Make maintenance transactional: conservative transport, geometry rebuild, wall-aware reinitialization, local geometry/volume reconciliation, validation, then commit. On any failed invariant, roll back the full state and revisions rather than accepting a mixture of stages.
- Keep raw post-transport, post-limit, post-reinitialization, and post-correction quantities. Global shift is a reported emergency fallback and may not be used to declare the transport conservative.

Required tests:

- Exact constant preservation, cancellation of the method's interior discrete fluxes, local control-volume balance of the declared conserved phase quantity, conservative limiter correction, boundedness, and transport across an MPI boundary.
- Divergence-free translation and rotation, wall advection, disconnected drops, thin films, Zalesak rotation, and reversible Enright deformation with maintenance disabled, each maintenance component isolated, and the final production sequence.
- Sweep extension-band width and map fallback, because transport is qualified only together with the bounded `E` field from WP-1.

Simulation exits:

- Translate drops at `D/dx = 16, 32, 64`; use Enright grids of approximately `32^3`, `64^3`, and `128^3`, with CFL `0.5`, `0.25`, and `0.125` in independent spatial and temporal studies.
- Continue through capillary jet and filament-necking cases only after raw global and per-component errors converge without global shift.
- Close with per-control-volume/component phase-flux artifacts, a conservative-limiter derivation, convergent interface norms and geometry, and explicit pre/post-maintenance mass and displacement histories.

### WP-7: coherent small-cut stability and conditioning

**Finding addressed:** FSR-07; this replaces any conditioning role previously attributed to FSR-01.

Implementation notes:

- Select one mathematically coherent method under AD-3. Do not combine pressure-only ghost penalty, empirical equal-order stabilization, unconstrained extrapolation, aggregation, pruning, and pinning and then assume their individual motivations prove the combination.
- For CutFEM, implement and scale the velocity and pressure stabilization required by the applicable analysis for the actual transient equal-order VMS/PSPG spaces. For aggregation, define the aggregate trial/test space, conservation behavior, root selection, maximum path/extrapolation distance, and coefficient/row-norm limits.
- Centralize stabilization coefficients and nondimensional scaling. Every term should report its dimensional scale and contribution to residual and energy.
- Make sliver pruning an explicit topology/model event. Production splash runs must report or reject unresolved rootless liquid features; deleted liquid cannot be counted as conserved merely because a bookkeeping total is adjusted.

Required cut matrix:

- Volume fractions `1e-8`, `1e-6`, `1e-4`, `1e-2`, `0.1`, `0.25`, and `0.49`; axis-aligned and oblique interfaces; at least three `h` levels and every supported polynomial order; viscous-, transient-, and advection-dominated regimes; and 1, 2, and 4 or more MPI ranks.
- Measure velocity/pressure error, divergence, mixed singular values or a reproducible inf-sup surrogate after removing pressure-gauge and componentwise pressure nullspaces, canonically scaled condition number, Krylov iterations, preconditioned spectrum where practical, aggregate/extension coefficient norms, rootless features, and solver fallback/retry.
- Include connected and disconnected features and a cut moving continuously through a node so a method switch cannot create an unreported force or solution jump.

Simulation exits:

- Manufactured Stokes/Navier--Stokes errors must be cut-position independent at the expected rate. Static caps, translating drops, filaments, and D18/D38 must not show a nonconvergent numerical jump caused solely by a mesh-relative cut-topology change; real physical or interface-topology events are analyzed separately.
- Closure requires a fixed sweep demonstrating cut-independent conditioning relative to a fitted/reference system at the same `h`, the theoretically expected `h` scaling after canonical scaling, bounded constraint amplification, and bounded preconditioned-iteration spread. The accompanying derivation must match the spaces and stabilization actually assembled.

### WP-8: geometry coupling, nonlinear convergence, and a complete energy law

**Finding addressed:** FSR-09.

Implementation notes:

- Implement the strategy selected by AD-5. A complete shape tangent must cover quadrature points, measures, normals, surface projector, contact geometry, cut volume, wet-wall rules, and extension-map changes while topology is fixed. An energy-stable split must instead define the common time stage and converge all coupled geometry/state fields to its declared tolerance.
- Start qualification with backward Euler and constant surface tension. Add generalized-alpha only after the simpler scheme has a closed energy balance, then demonstrate stage consistency for every surface, wall, contact, transport, and maintenance term.
- Record kinetic, gravitational, liquid--gas surface, solid--liquid wall, and any gas/compressibility energy; viscous, Navier-slip, and line-friction dissipation; external pressure/body work; and numerical work from VMS/PSPG, ghost penalty/aggregation, extension, pruning, limiting, redistancing, and volume reconciliation. Log rejected and rolled-back attempts separately; because they do not alter the accepted state, they contribute zero to the accepted-state energy balance.
- Treat topology changes as nonsmooth events. Define detection, snapshot invalidation, nonlinear restart or step rejection, energy jump accounting, and minimum resolved-feature policy.

Required tests and simulations:

- Finite-difference directional derivatives for every geometry-dependent residual on fixed topology, and same-state refresh neutrality for geometry and extension maps.
- Verify outer fixed-point contraction or the monolithic Jacobian under `h`/`dt` refinement, cut shifts, and MPI partitions.
- Run a static cap, capillary relaxation, linear capillary wave, droplet oscillation, sloshing, and wetting relaxation. The complete energy residual must converge; a closed dissipative case may not exhibit unexplained growth.
- Report energy before and after every maintenance substage. A globally decreasing final energy cannot hide a large positive transport/geometry error canceled by redistancing.

### WP-9: fitted-ALE free-surface policies

**Findings addressed:** FSR-10 and FSR-11 for the fitted path.

Implementation notes:

- Enforce the normal kinematic relation `w_m dot n = u dot n` at the free surface. Give the tangential policies unambiguous semantics: `SmoothingOnly` leaves tangential motion to the mesh-smoothing functional; `Prescribed` imposes the projected supplied tangential mesh velocity; `Free` should either mean no tangential constraint or be renamed if it is intended to follow fluid tangentially.
- Route the selected policy through the existing mesh-motion tangential-boundary infrastructure so mesh displacement has one owner. Detect conflicting mesh-motion and Navier--Stokes policies.
- Give each boundary separate local fluid-kinematic and mesh-motion enforcement policies with explicit compatibility checks. Their Nitsche/penalty forms and scaling need not be identical and must not overwrite one another or unrelated weak velocity conditions.
- Decide the fitted-path status of `SurfaceStress` and prescribed/dynamic contact angle explicitly. Either implement and qualify them through the common WP-4/WP-5 energy, geometry, and wetting infrastructure, or keep those combinations fail-closed and exclude them from fitted capability claims.

Required tests and exits:

- Each policy must create the distinct documented boundary operator/state. Test prescribed-vector projection, rotations, multiple surfaces, reverse registration order, and conflicting-policy rejection.
- Run a flat translating ALE interface, prescribed tangential shear, and fitted sloshing while measuring mesh quality, geometric conservation, phase volume, surface work, and policy-specific mesh velocity.
- No fitted policy is qualified until its effective value appears in both configuration provenance and mesh-motion history.

### WP-10: explicit one-phase boundary and a staged two-phase extension

**Finding addressed:** FSR-08.

Implementation notes:

- Retain and label a one-phase liquid capability for cases in which gas dynamics are demonstrably negligible. Its reference pressure is imposed external pressure; it must not be described as a solved ambient gas.
- For gas-sensitive splash, add an incompressible two-phase formulation with phasewise density and viscosity, both velocity/pressure fields or an equivalent stable one-field jump formulation, interface stress and velocity conditions, pressure enrichment appropriate to the jump, stabilization on both phases, phasewise conservation, and density-ratio-robust solvers.
- Couple phase and momentum transport consistently at density/viscosity jumps so phase reconciliation or mass correction cannot create untracked momentum. Qualify the phase-flux/momentum-flux relationship at high density ratio.
- Add compressible or otherwise validated gas physics before claiming trapped-air pressure, roof impact, air cushioning, ambient-pressure splash thresholds, or late atomization. The exact need should be decided from the benchmark nondimensional regime, not visual similarity.

Required qualification progression:

- Begin with planar pressure/viscous jumps, two-fluid hydrostatics, static drop, material-side reversal, both-phase mass, and high-density-ratio conditioning.
- Add two-fluid capillary waves, Hysing case 1, and a rising bubble. Treat Hysing case 2 after breakup as an intercode range rather than a single exact shape.
- Only then add ambient-pressure and gas-property sweeps for impact/cushioning and dry-wall splash. Maintain separate one-phase, incompressible-two-phase, and compressible-gas qualification records.

### Finding-to-closure traceability matrix

| Finding | Primary work package | Must pass before closure | Required completion artifact |
|---|---|---|---|
| FSR-01 | WP-1 | Wet-block/dry-depth invariance, zero island cross-coupling, replacement conditioning from WP-7 | Residual/Jacobian difference report and absence of legacy option in supported inputs |
| FSR-02 | WP-1/Q5 | Reproduction, bounded row/amplification, refresh neutrality, reduced D38 map reproducer, then full-horizon D38 | Per-revision map statistics and guard/fallback log |
| FSR-03 | WP-4 | Flat exact balance and convergent static circle/sphere/sessile matrix | Balanced-force derivation, pressure-range distance, and static qualification report |
| FSR-04 | WP-4/WP-5 | Phi-scale invariance and angle/contact preservation through maintenance | Derived wall contract and angle/reinitialization convergence report |
| FSR-05 | WP-5 | All Ren--E sign/speed gates and capillary-rise uncertainty gates | Stage-paired contact history and complete wetting energy ledger |
| FSR-06 | WP-6 | Cancellation of the selected method's phase fluxes, bounded conservative phase transport, and raw component-mass convergence | Method-specific control-volume/component flux files and maintenance-stage ledger |
| FSR-07 | WP-7 | Cut-position-independent error, cut-relative/expected `h`-scaled conditioning, stability surrogate, and MPI behavior | Fixed cut-sweep matrix and matching mathematical derivation |
| FSR-08 | WP-10 | Separate two-phase and gas-sensitive gates for any expanded claim | Capability-specific qualification matrices and model statement |
| FSR-09 | WP-8 | Shape derivative or energy-stable split, refresh neutrality, convergent full energy residual | Directional-derivative/energy-balance artifacts |
| FSR-10 | WP-0/WP-9 | Distinct fitted policy behavior and mesh-motion plumbing | Effective policy plus mesh-velocity histories |
| FSR-11 | WP-0/WP-9 | Multi-boundary registration-order invariance | Configuration and matrix comparison |
| FSR-12 | WP-0 | Exhaustive positive/negative parser matrix with no silent keys | Versioned schema and effective-configuration snapshots |
| FSR-13 | WP-2 | Constant-one equality on affine, warped, and high-order cells | Pointwise volume/mapping comparison |
| FSR-14 | WP-2 | Rejection of all injected geometry defects, false order claims, and achieved order below the required minimum | Quantitative validator report for every retained rule |
| FSR-15 | WP-2 | Valid contact-to-source provenance, deterministic remapping, and revision equality | Stable-ID/remapping/revision/ownership ledger |
| FSR-16 | WP-3, closed in the qualified affine P1/LinearCorner envelope | Exact wet-face moments, zero dry contribution, all supported operators sharply routed | [V6 boundary-domain and assembly record](qualification_logs/free_surface_wp3_sharp_boundary_v6_20260826_a73c77f4/record.md) and [summary](qualification_logs/free_surface_wp3_sharp_boundary_v6_20260826_a73c77f4/summary.json); WP-7, Q1, higher-order support, and uniform conditioning remain open |
| FSR-17 | WP-2 | Exact `(xi,x,phi)` pairing and restored high-order curvature convergence | Sample provenance and distorted-mesh curvature study |
| FSR-18 | WP-1 | Edge/geometric graph, label/MPI invariance, bounded collision behavior | Graph/map report under deliberate relabeling |

### Qualification campaign Q0--Q7

The following campaign should be implemented as a versioned benchmark registry rather than another collection of case-specific log parsers. Existing runners such as
[`run_fs16_physical_matrix.py`](../tests/cases/fluid/open_vessel_free_surface/run_fs16_physical_matrix.py),
[`run_impermeable_wall_advection_qualification.py`](../tests/cases/fluid/open_vessel_free_surface/run_impermeable_wall_advection_qualification.py),
[`run_test05_validation_grade.py`](../tests/cases/fluid/open_vessel_free_surface/run_test05_validation_grade.py),
[`run_test05_velocity_growth_smoke.py`](../tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py), and
[`unfitted_level_set/run_validation_matrix.py`](../tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/run_validation_matrix.py)
already capture useful provenance; their common behavior should be consolidated rather than discarded.

#### Q0: harness, provenance, and negative configuration tests

Before any physical gate, record accepted step/time, nonlinear stage, state/geometry/map revisions, source and dirty-tree hashes, compiler/libraries/options, machine, MPI ranks, threads, mesh/reference-data checksums, dimensional parameters, nondimensional groups, and all acceptance thresholds.

Every accepted step should expose:

- raw and post-maintenance global, component, film, sheet, rim, and satellite volumes;
- kinetic, gravitational, surface, wall, and any gas energy plus every dissipation/work channel listed in WP-8;
- map row/amplification and fallback statistics;
- cut fractions, achieved order, deleted/pruned/rootless features, aggregate paths, and geometry validation maxima;
- nonlinear/Krylov iterations, rejected attempts, time-step reductions, fallback modes, and rollback reason; and
- wall-clock time and peak resident memory.

Run the complete WP-0 invalid-input matrix in continuous integration. A malformed physics request should fail deterministically with a diagnostic, not produce a numerical comparison.

#### Q1: exact algebra, geometry, boundary, transport, and cut tests

Q1 contains the WP-1 through WP-7 low-level matrices. It is release blocking and should run before static capillarity. Where an exact identity is expected, gate a scaled residual, not a plot. Include serial/MPI and node-number permutations.

At minimum, Q1 contains:

- dry-depth/BC/extension-coefficient wet-block invariance and two-island decoupling;
- extension reproduction, coefficient/amplification bounds, refresh projection, collision bands, and a reduced deterministic D38 map reproducer;
- volume/surface/contact/wet-wall moments on affine, warped, high-order, tiny-cut, and fault-injected rules;
- every generic boundary form over dry-to-fully-wet fraction sweeps;
- capillary energy directional derivatives and flat-interface balance;
- method-specific conservative phase flux, limiter, and maintenance transactions; and
- velocity/pressure cut stability over the full fraction, regime, orientation, and MPI matrix.

A Q1 failure blocks using a later benchmark as evidence for the affected method. For example, a dam-break trajectory cannot waive an unbounded extension row, and a sessile cap cannot waive whole-face dry Nitsche work.

#### Q2: hydrostatic, capillary-equilibrium, and contact-equilibrium tests

Run flat surfaces for every axis/sign/wall orientation; translated circles and spheres; explicitly labeled one-phase adaptations of Groß--Reusken force tests; and sessile caps at five angles. Use at least `R/dx = 8, 16, 32`, adding 64 if three levels are not asymptotic, with multiple subcell offsets and rotations. Do not claim equivalence to original two-phase/XFEM ingredients that the current model does not contain.

For each state report interface norms, volume and centroid, pressure jump and pressure-range distance, capillary residual, divergence, maximum/mean velocity, parasitic capillary number, kinetic/surface/wall/gravity energy, angle, base radius, apex height, nonlinear convergence, and every geometry intervention. Test both sampled analytic initial geometry and a discrete energy-minimized equilibrium and label them distinctly.

Q2 passes only with scaled-roundoff flat assembly balance at exactly representable fields, solved flat states consistent with declared solver tolerances, and a stable observed convergence regime for curved states. The candidate finest-level thresholds in WP-4 must be frozen before running.

#### Q3: pure transport and smooth free-surface dynamics

Separate spatial and temporal error:

1. run `h`, `h/2`, `h/4` with time error made subordinate;
2. run `dt`, `dt/2`, `dt/4` at a fixed fine mesh; and
3. run a three-level diagonal refinement, adding a fourth level when the observed order is not asymptotic.

Recommended matrices are:

| Case | Spatial levels | Temporal levels | Principal quantities |
|---|---|---|---|
| Reversible Enright deformation | approximately `32^3`, `64^3`, `128^3` | CFL `0.5`, `0.25`, `0.125` | return-shape norms, raw/component volume, surface area, topology, map guards |
| Translating drop | `D/dx = 16, 32, 64` | same CFL sequence | shape/centroid, wet-block invariance, pressure/velocity, conservation |
| Linear capillary wave | `lambda/dx = 16, 32, 64` | `dt/T = 1/50, 1/100, 1/200` | frequency, damping, phase, amplitude, full energy residual |
| Oscillating drop | `R/dx = 8, 16, 32` | `dt/T = 1/100, 1/200, 1/400` | modal frequency/damping, volume, spurious modes, energy |
| Smooth sloshing | `W/dx = 32, 64, 128` | `dt/T = 1/50, 1/100, 1/200` | elevation probes, frequency/damping, pressure, conservation |

For transport cases, compare maintenance off, reinitialization only, correction only, and the proposed production transaction. Candidate smooth-dynamics gates are frequency error near `1%`, profile/amplitude error near `2%`, a stable asymptotic rate/GCI envelope over cut offsets, and finest-level unexplained energy defect below `1%`; these must be reconciled with the published reference uncertainty and frozen before use. A mildly nonmonotone pair requires another level and analysis, not automatic failure or a favorable-point pass.

#### Q4: dynamic wetting and side-wall qualification

Run analytic wedge/arc kinematics first, then the existing Ren--E side-wall matrix, sessile relaxation/spreading, resolved-slip dynamic wetting, and capillary rise. Independently refine `h`, `dt`, slip length resolution, and any numerical wall smoothing. Rotate equivalent cases among supported axis-aligned bottom, left, and right walls. Inclined or curved walls are a later gate that begins only after general planar/curved-wall frames, geometry, and contact ownership are implemented and verified.

Primary quantities are contact position and speed, apparent and microscopic angle at declared distances, Ren--E constitutive residual, wall slip, wall/contact measure, rise height/overshoot/damping, per-component volume, and wall/line energy. The same fields must be emitted before and after reinitialization and correction.

The public capillary-rise data should be the primary integral side-wall validation; the current four Ren--E tests remain sharp local regression gates. A sign failure, stage/revision mismatch, unresolved slip length, or smoothing-width-dependent limit fails Q4 even if final rise height is close. Q4 remains serial-only until the distributed wall-aware reinitialization gate in WP-5 passes; after that, representative partition sweeps become mandatory.

#### Q5: violent but one-phase-compatible free-surface motion

Use D18/D38 as numerical stress tests only after Q0--Q4 pass. They must reach their fixed accepted horizon while keeping bounded extension, raw mass, topology, geometry validation, cut-conditioning, and energy histories. Their present zero-surface-tension, no-contact-law, no-slip configurations must never be cited as physical wetting qualification.

Then qualify against applicable dam-break/run-up/impact and nonlinear sloshing references already catalogued above, including Martin--Moyce, Lobovsky, Kleefsman, SPHERIC Tests 02 and 05, and Synolakis-style run-up where the solver's governing assumptions match the reference. Suggested starting resolution is `H/dx = 32, 64, 128` with CFL `0.25`, `0.125`, and `0.0625`, followed by an additional level when impulses or thin jets are not resolved.

Compare complete histories and uncertainty bands for front position, free-surface probes, wall force/pressure and impulse, run-up, overturning time, volume distribution, and energy. Do not shift time origins, filter peaks, or select favorable probes after viewing results. Roof-impact cases that trap air are diagnostics of the one-phase limitation, not quantitative pressure validations.

#### Q6: film, filament, jet, and pre-breakup crown motion

Proceed in increasing topology difficulty:

1. contracting filament and end pinching;
2. Rayleigh--Plateau/capillary jet growth and breakup;
3. resolved wet-film drop impact and crown formation over the smooth pre-breakup interval; and
4. only then resolved sheet/rim instability and secondary fragments within the declared one-phase envelope.

Resolve an initial or residual film with at least `h_film/dx = 4, 8, 16`, adding levels until thickness-dependent quantities are asymptotic. For crown cases record `D_Base`, `D_Mid`, `D_Rim`, `H_Rim`, rim radius/speed, sheet thickness, residual film, finger wavelength/count, resolved drop-size distribution, per-region raw volume, surface area, and energy. Fix the comparison interval before running; do not move the cutoff to omit the first disagreement.

Minimum neck/rim/sheet scales and pruning/aggregation events must be shown next to the physical comparisons. A visually reasonable crown with an under-resolved sheet or correction-dominated volume is inconclusive, not a pass.

#### Q7: two-phase and gas-sensitive phenomena

Q7 begins only after WP-10. Run static/jump and Hysing tests first, then two-fluid capillary waves and rising bubbles, followed by air-cushioning, trapped-gas, ambient-pressure, and dry-wall splash matrices appropriate to the implemented gas model.

Dry splash, entrainment, roof-impact pressure, aerodynamic sheet breakup, and late atomization remain outside the current one-phase qualification even if a qualitative experiment looks plausible.

### Benchmark registry and applicability review

Create a version-controlled registry entry for every rigorous public benchmark considered for a capability claim. “All available” should mean all identified, applicable, sufficiently specified references in the maintained registry, not an impossible claim to have discovered every experiment ever published. Review and update the literature inventory at each major release.

Each registry entry should contain:

- exact citation, persistent data link, downloaded-data checksum, license/access note, and the date the source was verified;
- model applicability: one-phase, two-phase incompressible, compressible gas, contact model, turbulence, temperature, and dimensionality;
- geometry, initial and boundary conditions, material properties, angle/slip/contact parameters, and units;
- nondimensional groups and the acceptable parameter uncertainty/range;
- measured quantities, probe definitions, time origin, sampling/filtering, experimental repeatability or intercode spread, and known ambiguities;
- mesh, time-step, cut-offset, rotation, maintenance, and MPI matrices;
- fixed comparison interval, convergence estimator, error norm, and acceptance gate; and
- an explicit disposition: `PASS`, `FAIL_METHOD`, `FAIL_MODEL`, `INCONCLUSIVE_RESOLUTION`, `INFRASTRUCTURE_FAILURE`, or `OUT_OF_SCOPE`.

Keep calibration and validation data disjoint. Slip length, mobility, contact-angle law, smoothing, stabilization, pruning, and correction parameters may be calibrated only on designated cases and then frozen for independent validation. Failed or inconclusive runs remain in the archive; they should not be replaced by a favorable rerun without preserving the original and explaining the change.

### Convergence, uncertainty, and comparison rules

- Estimate spatial and temporal order independently. Use at least three levels in an apparent asymptotic range and add a fourth for oscillatory or nonmonotone sequences.
- Report observed order and Richardson extrapolation/Grid Convergence Index with the chosen safety factor. For first-order interface quantities, an initial minimum observed order near `0.8` can be proposed, but it must be derived per quantity and frozen rather than applied blindly.
- Use the worst subcell cut offset, rotation, and representative MPI partition in the qualification conclusion, not only the best-aligned mesh.
- For an experimental scalar quantity, a defensible initial acceptance rule under a predeclared common confidence convention is
  \[
  |Q_h-Q_{exp}| \le 2\sqrt{U_{num}^2+U_{exp}^2},
  \]
  with both uncertainty terms independently defined. Independently established model-form uncertainty may be reported as context, but it may not be adjusted to enlarge the acceptance band. A known governing-model mismatch is `FAIL_MODEL` or `OUT_OF_SCOPE`, and this rule does not rescue a nonconverged numerical result.
- Compare time histories using declared integrated/RMS norms plus physical feature errors such as arrival time, peak/impulse, frequency, damping, and equilibrium. Do not use undocumented time shifts, spatial registration, smoothing, or peak filtering.
- Exact/manufactured tests require both a finest-level tolerance and the expected rate. Experimental tests require numerical convergence and agreement with uncertainty; an isolated value inside an error bar is insufficient.
- An MPI result should agree with serial within the discretization and solver uncertainty. Larger differences require a deterministic ownership/reduction investigation.

### Required run-artifact layout

Each immutable run directory should contain the equivalent of:

```text
manifest.json       case/reference IDs, model envelope, inputs, units, nondimensional groups
build.json          source and dirty-tree hashes, compiler, libraries, options, machine, MPI/threads
gates.json          predeclared metrics, intervals, tolerances, and expected convergence
run.json            outcome, resources, solver/retry/fallback/rollback summary
history.csv         accepted-stage physical and numerical histories with revision IDs
geometry.*          cut, surface, wet-wall, contact, pruning, and topology diagnostics
solver.*            nonlinear/Krylov histories and conditioning/stability summaries
comparison.json     raw errors, convergence, uncertainty, disposition, and reason
checkpoints/        fixed requested times, including states before and after maintenance
plots/              generated only from archived machine-readable quantities
checksums.txt        checksums for every input, reference, and result artifact
```

Every phase quantity must distinguish raw post-transport, post-limiter, post-reinitialization, post-correction, and retained assembly values. A result lacking source/configuration/reference provenance or predeclared gates is exploratory and cannot enter a release claim.

### CI, HPC, and compute-resource safeguards

Use resource tiers so exact regressions remain frequent without making large three-dimensional qualification runs unreliable:

- **CI-0, every relevant change:** parser negative tests, exact algebra/geometry/boundary identities, small serial/MPI smoke tests, and schema checks.
- **CI-1, nightly:** two-level transport, static cap, wetting sign, cut-fraction, and extension-map matrices.
- **CI-2, weekly:** three-level smooth dynamics, side-wall/capillary-rise subsets, reduced deterministic D18/D38 map/fixed-step regressions, and representative MPI/cut rotations.
- **CI-3, release/HPC:** the full frozen Q0--Q6 matrix, including full-horizon D18/D38, with independent space/time refinement and reference comparisons.
- **CI-4, after a major method or physics change:** breakup/splash and, when present, the full two-phase/gas-sensitive Q7 matrix.

Every runner should declare memory, wall-time, output-size, rank, and thread envelopes; sample peak resident memory throughout the run; and terminate or reschedule cleanly before node exhaustion. Geometry/map caches need explicit byte counters and bounded retention. Run independent cases through scheduler job arrays rather than holding all meshes or histories in one process. Stream or checkpoint long histories at fixed intervals, and keep plotting/postprocessing out of the solver allocation where possible. Treat a reproducible resource regression as a test failure even when physical metrics pass, but compare resource trends only on controlled hardware using normalized, statistically defined bounds.

### Failure triage after implementation

| Observed failure | First mechanism to isolate | First paired ablation or diagnostic |
|---|---|---|
| Static flat residual | Sign, active side, geometry revision, or pressure/external-pressure pair | Reverse side/orientation and inspect exact assembled force/gradient rows |
| Wet solution changes with dry depth/BC | Physical dry diffusion or an unclipped boundary form | Remove dry layers/BC values one factor at a time and compare wet blocks |
| Velocity/interface spike at refresh | Ill-conditioned/stale extension map or topology remap | Compare pre/post map row guards, projection residual, and revision IDs |
| Static curved parasitic current | Pressure representability/curvature/geometry imbalance | Best-pressure residual and discrete-energy directional derivative |
| Angle changes after maintenance | Missing wall-aware reinitialization or global shift | Compare each maintenance substage and contact-root provenance |
| Wrong contact speed/sign | Stage mismatch, wall frame, slip/mobility, or smoothed domain | Rotate/reverse paired case with identical revision-tagged telemetry |
| Total volume good but local regions wrong | Nonconservative transport hidden by global correction | Inspect raw cell/component/film/sheet ledgers with correction disabled |
| Cut-position or MPI sensitivity | Stabilization, aggregation, pruning, ownership, or graph choice | Fixed physical case with only offset/partition/numbering changed |
| Unexplained energy growth | Frozen geometry/stage mismatch or unreported numerical work | Complete substage energy ledger and fixed-topology derivative test |
| Only splash/breakup disagrees | Under-resolution, topology intervention, or absent gas physics | Resolve film/sheet/rim, disable pruning/correction when safe, then test model envelope |

Use one-factor paired ablations and rerun the failed prerequisite before the original benchmark. Do not compensate for a method failure by retuning contact, stabilization, extension, pruning, or volume-correction parameters on the validation case.

### Capability-specific definition of done

The **one-phase hydrostatic/capillary** capability is done only after Q0--Q2 pass over cut/rotation/MPI sweeps with a closed force and energy account.

The **one-phase moving-wetting** capability additionally requires Q3--Q4, all existing advancing/receding sign and speed gates, resolved physical slip, wall-aware maintenance, and public capillary-rise agreement.

The **one-phase violent-flow** capability additionally requires Q5 with raw conservation, bounded maps, cut stability, and uncertainty-qualified dam-break/run-up/sloshing histories. D18/D38 completion alone is a stress-test result, not physical validation.

The **one-phase film/jet/pre-breakup crown** capability additionally requires Q6 with independently resolved film, sheet, rim, or neck scales and no correction- or pruning-dominated result.

The current one-phase model cannot be declared validated for gas cushioning, gas inertia/viscosity/compressibility, trapped air, entrainment, aerodynamic breakup, gas-pressure-dependent dry splash, Hysing two-fluid flow, or late atomization. Those claims require WP-10 and Q7 appropriate to the implemented gas model.

The overall free-surface implementation should be called robust only when every release-blocking finding has its closure artifact, the applicable registry matrix passes without benchmark-specific retuning, failures and model exclusions remain visible, and an independent reviewer can reproduce the conclusion from the archived configuration, reference data, and machine-readable histories.

## Release-blocking gates

The implementation should not be described as robust free-surface wetting/splash capability until all of these are closed:

- flat and static curved capillary states have mesh- and cut-convergent pressure and parasitic-current errors;
- the best pressure-space residual meets a fixed physical threshold;
- wet liquid solutions are invariant to irrelevant dry-domain extension geometry and boundary values;
- algebraic extension maps satisfy fixed stability/amplification bounds;
- all advancing/receding contact-speed signs and quantitative gates pass under mesh/time refinement;
- capillary-rise height and damping converge with a physically resolved slip length;
- raw local and global interface mass errors converge without relying on a global shift;
- cut stability, constraint amplification, and preconditioned iterations satisfy fixed cut-relative and expected `h`-scaled bounds across tiny-cut sweeps and MPI partitions;
- weak/natural exterior BC work is restricted to the actual wetted boundary and converges under cut sweeps;
- the complete surface/wall/line energy residual converges;
- D18/D38 reach their fixed accepted horizons as transport tests; and
- splash claims are limited to phenomena represented by the one-phase model, or a gas phase is added.

## Final assessment

The supported sharp-P1 free-surface weak form is more internally consistent than the observed failures initially suggest: its pressure sign, external-pressure sign, capillary projector, liquid-outward normal, Young angle convention, conormal ownership, side-wall frame, wall impermeability, and dissipative line/slip signs are all correct within the declared envelope.

The failures instead expose system-level incompatibilities. The physical velocity is regularized through a fictitious dry PDE that feeds back into momentum. The separate transport extension can be arbitrarily amplifying and component-number dependent. Generic boundary forms are not sharply clipped to the wet face. Polygonal surface force and continuous pressure are not exactly well balanced. Prescribed angle is an unscaled geometric penalty, dynamic wetting remains quantitatively failed, redistancing lacks a wetting wall condition, transport is not locally conservative, and general cut stability and energy behavior remain unproved. High-order volume and curvature paths also contain definite geometry-mapping inconsistencies. These issues are directly relevant to side-wall run-up, thin-film motion, crown formation, and breakup.

The next scientifically defensible milestone is not a visually plausible splash. It is a passing hierarchy of extension-invariance, flat/static force balance, conservative transport, cut-position stability, resolved moving contact, capillary rise, and complete energy tests. Wet-film crown splash should follow only after those gates pass; gas-dependent dry splash requires a two-phase model.

## References

1. Bänsch, E. “Finite element discretization of the Navier--Stokes equations with a free capillary surface.” [DOI](https://doi.org/10.1007/PL00005443).
2. Barrett, J. W., Garcke, H., and Nürnberg, R. “Eliminating spurious velocities with a stable approximation of viscous incompressible two-phase Stokes flow.” [DOI](https://doi.org/10.1016/j.cma.2013.09.023), [preprint](https://arxiv.org/abs/1306.2192).
3. Buscaglia, G. C., and Ausas, R. F. “Variational formulations for surface tension, capillarity and wetting.” [DOI](https://doi.org/10.1016/j.cma.2011.06.002).
4. Castrejón-Pita, J. R. et al. “Plethora of transitions during breakup of liquid filaments.” [DOI](https://doi.org/10.1103/PhysRevLett.108.074506).
5. Cossali, G. E. et al. “The role of time in single drop splash on thin film.” [DOI](https://doi.org/10.1007/s00348-003-0772-0).
6. Enright, D. et al. “A hybrid particle level set method for improved interface capturing.” [DOI](https://doi.org/10.1006/jcph.2002.7166).
7. Frachon, T., and Zahedi, S. “A cut finite element method for incompressible two-phase Navier--Stokes flows.” [DOI](https://doi.org/10.1016/j.jcp.2019.01.028), [preprint](https://arxiv.org/abs/1808.02662).
8. Geppert, A. et al. “A benchmark study for crown-type splashing dynamics.” [DOI](https://doi.org/10.1007/s00348-017-2447-2).
9. Groß, S., and Reusken, A. “Finite element discretization error analysis of a surface tension force.” [DOI](https://doi.org/10.1137/060667530).
10. Gründing, D. et al. “A comparative study of transient capillary rise using direct numerical simulations” (preprint title: “Capillary Rise -- A Computational Benchmark for Wetting Processes”). [DOI](https://doi.org/10.1016/j.apm.2020.04.020), [preprint](https://arxiv.org/abs/1907.05054), [data](https://doi.org/10.25534/tudatalib-173).
11. Huh, C., and Scriven, L. E. “Hydrodynamic model of steady movement of a solid/liquid/fluid contact line.” [DOI](https://doi.org/10.1016/0021-9797(71)90188-3).
12. Hysing, S. et al. “Quantitative benchmark computations of two-dimensional bubble dynamics.” [DOI](https://doi.org/10.1002/fld.1934).
13. Kleefsman, K. M. T. et al. “A volume-of-fluid based simulation method for wave impact problems.” [DOI](https://doi.org/10.1016/j.jcp.2004.12.007).
14. Lobovský, L. et al. “Experimental investigation of dynamic pressure loads during dam break.” [DOI](https://doi.org/10.1016/j.jfluidstructs.2014.03.009), [preprint/data description](https://arxiv.org/abs/1308.0115).
15. Mani, M., Mandre, S., and Brenner, M. P. “Events before droplet splashing on a solid surface.” [DOI](https://doi.org/10.1017/S0022112009993594).
16. Massing, A. et al. “A stabilized Nitsche fictitious domain method for the Stokes problem.” [DOI](https://doi.org/10.1007/s10915-014-9838-9), [preprint](https://arxiv.org/abs/1206.1933).
17. Mundo, C., Sommerfeld, M., and Tropea, C. “Droplet-wall collisions: experimental studies of the deformation and breakup process.” [DOI](https://doi.org/10.1016/0301-9322(94)00069-V).
18. Notz, P. K., and Basaran, O. A. “Dynamics and breakup of a contracting liquid filament.” [DOI](https://doi.org/10.1017/S0022112004009759).
19. Popinet, S. “An accurate adaptive solver for surface-tension-driven interfacial flows.” [DOI](https://doi.org/10.1016/j.jcp.2009.04.042).
20. Prosperetti, A. “Viscous effects on small-amplitude surface waves.” [DOI](https://doi.org/10.1063/1.861446).
21. Ren, W., and E, W. “Boundary conditions for the moving contact line problem.” [DOI](https://doi.org/10.1063/1.2646754).
22. Reusken, A., Xu, X., and Zhang, L. “Finite element methods for a class of continuum models for immiscible flows with moving contact lines.” [DOI](https://doi.org/10.1002/fld.4349), [preprint](https://arxiv.org/abs/1510.03160).
23. Rider, W. J., and Kothe, D. B. “Reconstructing volume tracking.” [DOI](https://doi.org/10.1006/jcph.1998.5906).
24. Sprittles, J. E., and Shikhmurzaev, Y. D. “Finite element framework for describing dynamic wetting phenomena.” [DOI](https://doi.org/10.1002/fld.2603), [accepted manuscript](https://wrap.warwick.ac.uk/id/eprint/78933/).
25. Sussman, M., Smereka, P., and Osher, S. “A level set approach for computing solutions to incompressible two-phase flow.” [DOI](https://doi.org/10.1006/jcph.1994.1155).
26. Synolakis, C. E. “The runup of solitary waves.” [DOI](https://doi.org/10.1017/S002211208700329X).
27. Xu, S., and Ren, W. “Reinitialization of the level-set function in 3D simulation of moving contact lines.” [DOI](https://doi.org/10.4208/cicp.210815.180316a).
28. Xu, L., Zhang, W. W., and Nagel, S. R. “Drop splashing on a dry smooth surface.” [DOI](https://doi.org/10.1103/PhysRevLett.94.184505).
29. Yarin, A. L., and Weiss, D. A. “Impact of drops on solid surfaces: self-similar capillary waves, and splashing as a new type of kinematic discontinuity.” [DOI](https://doi.org/10.1017/S0022112095002266).
30. Zahedi, S., Kronbichler, M., and Kreiss, G. “Spurious currents in finite element based level set methods for two-phase flow.” [DOI](https://doi.org/10.1002/fld.2643).
31. Zhao, X., and Ren, W. “A finite element method for two-phase flows with moving contact lines.” [DOI](https://doi.org/10.1016/j.jcp.2020.109582), [preprint](https://arxiv.org/abs/2002.12009).
32. Bagheri, M. et al. “Insights into the Dynamics of Crown Splash Using a Phase-Field Interface-Capturing Method: Benchmark data.” [Open dataset](https://tudatalib.ulb.tu-darmstadt.de/items/c83209b0-090c-4f4c-a3dc-c53fe58534f7).
33. Badia, S., Martín, A. F., and Verdugo, F. “Mixed aggregated finite element methods for the unfitted discretization of the Stokes problem.” [DOI](https://doi.org/10.1137/18M1185624), [preprint](https://arxiv.org/abs/1805.01727).
34. Burman, E., and Hansbo, P. “Fictitious domain finite element methods using cut elements: II. A stabilized Nitsche method.” [DOI](https://doi.org/10.1051/m2an/2013123).
35. Burman, E., Hansbo, P., and Larson, M. G. “On the design of locking free ghost penalty stabilization and the relation to CutFEM with discrete extension.” [Preprint](https://arxiv.org/abs/2205.01340).
36. Della Rocca, G., and Blanquart, G. “Level set reinitialization at a contact line.” [DOI](https://doi.org/10.1016/j.jcp.2014.01.040).
37. Kuzmin, D., and Quezada de Luna, M. “Algebraic entropy fixes and convex limiting for continuous finite element discretizations of scalar hyperbolic conservation laws.” [Preprint](https://arxiv.org/abs/2003.12007).
38. Olshanskii, M. A., Reusken, A., and Schwering, P. “A narrow band finite element method for the level set equation.” [DOI](https://doi.org/10.1137/24M1674182).
39. Saye, R. I. “High-order quadrature on multi-component domains implicitly defined by multivariate polynomials.” [Preprint](https://arxiv.org/abs/2105.08857).

## Nonmutation and compute-monitoring note

Only this Markdown report was created. No solver, test, input, build, or existing documentation file was edited, and no Git commit was made. Compute and memory use were monitored during the audit. No build or solver run was launched; the work remained source/literature analysis. Although raw free RAM was sometimes small because Linux used memory for cache, available memory remained approximately 9.8--10 GiB at the final check, with no sustained CPU or I/O pressure attributable to this audit.
