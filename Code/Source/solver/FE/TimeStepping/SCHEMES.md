# Time Stepping Schemes for Transient Finite Element Solvers
> Filename: `SCHEMES.md`

This document is a **practical catalog** of time integration (time-stepping) schemes used in transient finite element (FE) analysis, with:
- **Orders of accuracy** (typical/global order for smooth solutions)
- **Stability/dissipation characteristics** (what the scheme “does” to stiff/high-frequency modes)
- **Expected outcomes** (what you will observe in typical FE runs)
- **Performance metrics** (what to measure and how the method “spends” work)
- **Ideal use cases** and **why you’d include the method** in a multiphysics FE solver
- **Implementation notes** (esp. for mass matrices, DAEs, and nonlinear solves)
- A **references** section (books/papers/manuals)

The focus is on the common semi-discrete FE forms:

- **First-order ODE/DAE form** (most multiphysics):
  \[ M(u,t)\,\dot u = R(u,t) \quad \text{(possibly with algebraic constraints)} \]
- **Second-order structural dynamics**:
  \[ M \ddot u + C \dot u + K u = f(t) \quad \text{(linear or nonlinear)} \]

---

## 1) What to measure: outcomes and performance metrics

A multiphysics solver should ideally log per time step (and optionally per nonlinear iteration):

### Accuracy & solution quality
- **Observed convergence rate**: run a temporal refinement study and estimate slope \(p\) from error \(\|e\|\sim \Delta t^p\).
- **Phase error / dispersion** (waves): time integration can shift wave speed and distort frequencies.
- **Amplitude decay / algorithmic damping**: especially important for structural dynamics and acoustics.
- **Monotonicity / positivity** (transport, diffusion with sharp fronts): avoid spurious undershoots/overshoots.
- **Invariant conservation** (Hamiltonian energy, momentum, mass): long-time quality can depend more on structure preservation than formal order.

### Robustness & stability
- **CFL/stability limit** (explicit methods): maximum stable \(\Delta t\) often scales like \(h/\lambda_{\max}\) or \(h^2/\nu\), depending on operator.
- **A-stability / L-stability** (implicit methods): ability to take large steps on stiff problems without numerical blow-up; L-stable methods strongly damp very stiff modes.
- **DAE index handling** (constraints, incompressibility, multipoint constraints): whether the method remains stable/accurate for differential-algebraic systems.

### Work / cost
- **RHS evaluations** (explicit stages, IMEX explicit part): count per step.
- **Nonlinear solves** (Newton iterations per step) and **linear solves** (Krylov iterations, preconditioner setup frequency).
- **Jacobian evaluations / factorization reuse**: whether a method can reuse preconditioners across stages/steps.
- **Memory footprint**: multistep methods store history; multi-stage methods store stage vectors.
- **Parallel efficiency**: stages can limit concurrency; multistep needs history; implicit solves can dominate wall time.

### Practical “work–precision” metric
A good solver capability is to produce **work–precision diagrams**:
- x-axis: cost (CPU-seconds, linear iterations, FLOPs, energy)
- y-axis: achieved error (or a QoI error)
  This reveals whether a higher-order method is **actually cheaper** for a target accuracy.

---

## 2) Glossary (short)
- **Order \(p\)**: global accuracy ~ \(O(\Delta t^p)\) for smooth solutions (assuming spatial error and nonlinear/linear solver tolerances are tighter than time error).
- **A-stable**: stable for the entire left half-plane on the linear test equation \(y'=\lambda y\).
- **L-stable**: A-stable and strongly damps as \(\lambda\Delta t \to -\infty\) (stiff decay).
- **SSP/TVD**: strong-stability-preserving; preserves certain nonlinear stability properties (e.g., total variation diminishing) under a timestep restriction [GST01].
- **IMEX**: implicit–explicit splitting; stiff terms treated implicitly, nonstiff terms explicitly [ARS97], [KC03].
- **Splitting**: operator splitting / fractional-step; solve sub-physics sequentially (Lie = 1st order; Strang = 2nd order) [Strang68].

---

## 3) Catalog A — First-order ODE/DAE time integrators
This section assumes a first-order semi-discrete system \(M \dot u = R(u,t)\) (possibly nonlinear).

### A1) Explicit one-step (Runge–Kutta / Euler family)
These methods typically require **mass lumping** or efficient application of \(M^{-1}\) (or using a mass-matrix solve per stage). They are limited by **CFL** constraints but are very attractive on GPUs and for strongly hyperbolic problems when stable.

#### Forward Euler (FE)
- **Order**: 1
- **Type**: explicit, 1 stage
- **Expected outcomes**
    - Very diffusive; strong numerical damping.
    - Can destroy accuracy for wave-like problems unless \(\Delta t\) is extremely small.
- **Performance notes**
    - Cheapest per step (1 RHS eval), but often **most expensive per accuracy**.
- **Ideal use cases**
    - Cheap transient “relaxation” to steady state (pseudo-time stepping)
    - Strongly damped processes where accuracy is secondary
- **Why include**
    - Baseline explicit integrator; debugging; building block for SSP methods.

#### Explicit midpoint (RK2), Heun (RK2), Ralston (RK2 variants)
- **Order**: 2
- **Type**: explicit RK (2 stages)
- **Expected outcomes**
    - Much better accuracy than FE at modest extra cost.
- **Use cases**
    - Nonstiff advection, wave propagation with moderate accuracy needs.
- **Why include**
    - Simple “step up” from FE; often good for verification tests.

#### SSPRK2 / SSPRK3 (TVD Runge–Kutta)
- **Order**: 2 / 3
- **Type**: explicit RK with SSP property
- **Expected outcomes**
    - Better control of spurious oscillations for hyperbolic problems with monotone spatial discretizations.
- **Performance notes**
    - 2–3 RHS evaluations per step.
- **Ideal use cases**
    - DG/FV/FE advection with limiters; shock capturing; positivity/TVD constraints.
- **Why include**
    - A standard option when robustness near discontinuities matters [GST01], [ShuOsher88].

#### Classical RK4
- **Order**: 4
- **Type**: explicit RK (4 stages)
- **Expected outcomes**
    - High accuracy for smooth, nonstiff problems.
- **Performance notes**
    - 4 RHS evaluations per step; CFL still limits \(\Delta t\).
- **Ideal use cases**
    - Smooth wave propagation; explicit structural dynamics with lumped mass; electromagnetics.
- **Why include**
    - Workhorse high-order explicit method; easy to implement and verify.

#### Low-storage explicit RK (LSRK)
- **Order**: typically 3–5 (family-dependent)
- **Why include**
    - Large-scale explicit FE/DG where memory bandwidth dominates.

---

### A2) Implicit one-step (θ-method, trapezoidal, implicit RK)
Implicit methods pay for stability with nonlinear/linear solves per step, but excel for diffusion-dominated, stiff reaction, and strongly coupled multiphysics.

#### Backward Euler (BE)
- **Order**: 1
- **Type**: implicit, 1 stage
- **Stability**: A-stable and strongly damped (stiff decay). Widely used for stiff problems [HW96], [AP98].
- **Expected outcomes**
    - Very robust; suppresses oscillations.
    - **Numerical diffusion** can smear sharp transients and waves.
- **Performance notes**
    - 1 nonlinear solve/step; Jacobian can often be reused.
- **Ideal use cases**
    - Highly stiff diffusion/reaction; strongly coupled multiphysics; DAEs with constraints (often as a “safe mode”).
- **Why include**
    - The “failsafe” integrator; essential for production multiphysics.

#### θ-method (generalized trapezoidal rule)
- **Order**: 1 for \(\theta\neq 1/2\); 2 for \(\theta=1/2\).
- **Type**: implicit single-step family (includes BE and CN)
- **Expected outcomes**
    - \(\theta>1/2\) adds damping (often used to tame oscillations).
- **Why include**
    - One implementation covers multiple behaviors (robust ↔ accurate).

#### Crank–Nicolson (CN) / Trapezoidal rule
- **Order**: 2
- **Type**: implicit, 1 stage (θ=1/2)
- **Stability**: A-stable but **not** strongly damping (no stiff decay), so stiff transients can “ring”.
- **Expected outcomes**
    - Excellent for smooth diffusion when \(\Delta t\) resolves transients.
    - Can exhibit **spurious oscillations** for sharp initial layers or stiff source terms.
- **Ideal use cases**
    - Diffusion/heat conduction when accuracy is needed and stiffness is moderate.
    - Linear problems where large steps are not taken “too early”.
- **Why include**
    - Common reference method; good accuracy/cost balance.

#### Implicit midpoint
- **Order**: 2
- **Type**: implicit RK (1 stage)
- **Key property**: **symplectic**; good long-time energy behavior in Hamiltonian/wave systems [HairerGNI].
- **Expected outcomes**
    - Preserves qualitative oscillatory behavior better than BE/CN in long runs.
- **Ideal use cases**
    - Conservative mechanical systems; long-time wave propagation with minimal artificial damping.
- **Why include**
    - Structure preservation option (important for long-time dynamics).

#### SDIRK / DIRK / ESDIRK families (diagonally implicit RK)
- **Order**: commonly 2–5 (scheme-dependent)
- **Type**: implicit RK with lower-triangular stage coupling
- **Performance notes**
    - Multiple implicit stages; can reuse Jacobian/preconditioner across stages.
- **Ideal use cases**
    - Moderately stiff problems requiring >2nd order.
    - DAEs and multiphysics where robust nonlinear convergence is needed.
- **Why include**
    - “High-order implicit” without fully coupled stage solves.
- **References**: classic DIRK constructions include Alexander’s methods [Alexander77] and standard stiff ODE texts [HW96].

#### Fully implicit RK collocation methods
These are very robust for DAEs and stiff problems, but expensive because stages are fully coupled.

- **Gauss–Legendre IRK**
    - **Order**: \(2s\) with \(s\) stages (high-order)
    - **Properties**: A-stable, **symplectic** (excellent long-time energy) [HairerGNI].
    - **Use cases**: high-accuracy conservative dynamics; verification/reference solutions.

- **Radau IIA**
    - **Order**: \(2s-1\)
    - **Properties**: stiffly accurate, strong stiff decay; popular for stiff ODE/DAE [HW96], [AP98].
    - **Use cases**: stiff DAEs, constraint problems, reactive flows (when very robust integration is needed).

- **Lobatto (e.g., Lobatto IIIC)**
    - **Order**: \(2s-2\)
    - **Use cases**: DAEs and constrained mechanical systems; some variants have good constraint handling [HW96].

---

### A3) Linearly implicit one-step (Rosenbrock / ROW / W-methods)
These avoid nonlinear solves by replacing them with **a sequence of linear solves** per step.

#### Rosenbrock–Wanner / Rosenbrock-type (ROW) methods
- **Order**: commonly 2–4 (scheme-dependent)
- **Type**: linearly implicit RK
- **Expected outcomes**
    - Much more robust than explicit RK on stiff problems without full Newton solves.
- **Performance notes**
    - \(s\) linear solves per step; can reuse Jacobian/preconditioner.
- **Ideal use cases**
    - Stiff multiphysics where Newton is costly or fragile; large-scale PDE where linear solves dominate.
- **Why include**
    - Great compromise: robustness close to implicit RK with simpler implementation.
- **References**: Rosenbrock methods are standard stiff integrators [HW96]; see also modern surveys [LangROW2020].

#### Rosenbrock–Krylov (concept)
- **Order**: typically 2–4
- **Why include**
    - Enables Jacobian-free Krylov with reduced subspace dimension; useful when assembling Jacobians is hard.

---

### A4) Composite one-step stiff solvers
#### TR-BDF2 (Trapezoidal + BDF2 composite)
- **Order**: 2
- **Type**: implicit, **one-step composite** (two implicit substeps)
- **Key properties**
    - Designed to combine trapezoidal accuracy with better stiff behavior; widely used in stiff ODE codes [HoseaShampine96].
    - L-stable parameter choices are common in practice [HoseaShampine96].
- **Expected outcomes**
    - Better damping of stiff/initial layers than CN, without dropping to 1st order like BE.
- **Performance notes**
    - Typically 2 implicit solves per step (but can reuse matrices/factorizations).
- **Ideal use cases**
    - Stiff diffusion–reaction, circuit-like systems, multiphysics transients with sharp initial layers.
- **Why include**
    - One of the best “default stiff 2nd order” methods for production solvers.

---

## 4) Catalog B — Linear multistep methods (LMM)
Multistep methods store history and can be very efficient (one implicit solve per step) once started.

### B1) Adams–Bashforth (AB) explicit multistep
- **Order**: \(k\) for \(k\)-step AB
- **Expected outcomes**
    - Efficient for smooth, nonstiff problems; can be fragile for stiff terms.
- **Ideal use cases**
    - Explicit time stepping when memory is cheaper than extra RHS evaluations (rare in modern GPU settings).
- **Why include**
    - Useful for specialized explicit regimes; historical relevance.

### B2) Adams–Moulton (AM) implicit multistep / predictor–corrector ABM
- **Order**: typically \(k+1\) for a \(k\)-step AM formula
- **Expected outcomes**
    - Good efficiency for mildly stiff problems; but can struggle on very stiff modes compared to BDF.
- **Why include**
    - Predictor–corrector pairs can reduce nonlinear solve cost.

### B3) Backward Differentiation Formulas (BDF1–BDF6)
- **Order**: 1–6 (method order equals step number)
- **Stiffness note**
    - BDF methods are the classic workhorse for stiff ODE/DAE integration [Gear71], [HW96], [AP98].
    - BDF1 (Backward Euler) and BDF2 are A-stable; higher-order BDF methods trade some stability for higher order (Dahlquist barrier context) [HW96].
- **Expected outcomes**
    - Excellent robustness for diffusion/reaction; controlled damping of stiff modes.
    - Can exhibit **order reduction** for DAEs or stiff boundary layers if tolerances/BC handling are not careful.
- **Performance notes**
    - 1 implicit solve per step; strong advantage for large-scale stiff PDEs.
    - Variable-step/variable-order (VSVO) BDF is widely used in general-purpose integrators [AP98].
- **Ideal use cases**
    - Large stiff multiphysics problems where each implicit solve is expensive and you want **one solve per step**.
- **Why include**
    - Production-grade stiff integrator family; essential in multiphysics.

### B4) Semi-implicit BDF (SBDF) / IMEX multistep
- **Order**: typically 1–4 (scheme-dependent)
- **Why include**
    - Good when splitting stiff linear diffusion (implicit) from explicit nonlinear advection/reaction.

---

## 5) Catalog C — IMEX / Additive methods (partitioned physics)
IMEX methods are often the best compromise for multiphysics where you have:
- stiff diffusion or implicit constraints (implicit part),
- nonstiff transport/forcing (explicit part).

### IMEX Euler
- **Order**: 1
- **Expected outcomes**
    - Robust for stiff linear terms; still CFL-limited by explicit portion.
- **Why include**
    - Baseline IMEX scheme; debugging and safe fallback.

### IMEX Runge–Kutta (ARS family)
- **Order**: typically 2–4
- **Key reference**: Ascher–Ruuth–Spiteri developed RK-based IMEX schemes with good stability regions [ARS97].
- **Ideal use cases**
    - Convection–diffusion; reacting flows; weakly compressible flow with stiff source terms.
- **Why include**
    - Avoids solving fully coupled monolithic stiff system when physics splits naturally.

### Additive Runge–Kutta (ARK / IMEX-ARK)
- **Order**: typically 3–5
- **Key reference**: Kennedy & Carpenter ARK schemes (implicit–explicit) for convection–diffusion–reaction [KC03], also in NASA report form [KC01].
- **Expected outcomes**
    - High-order accuracy with controlled stiffness “leakage” when splitting is well designed.
- **Performance notes**
    - Several explicit RHS evaluations and several implicit stage solves per step.
- **Why include**
    - Best-in-class IMEX options for high-order multiphysics.

---

## 6) Catalog D — Operator splitting and fractional-step methods
Splitting is extremely common in multiphysics (chemistry + transport, Navier–Stokes projection, etc.), but introduces **splitting error** that can dominate if operators do not commute.

### Lie / Godunov splitting
- **Order**: 1
- **Expected outcomes**
    - Robust but can be too inaccurate unless \(\Delta t\) is small.
- **Why include**
    - Simple partitioned coupling; good baseline for staged physics.

### Strang splitting
- **Order**: 2
- **Key reference**: Strang’s original construction [Strang68].
- **Expected outcomes**
    - Often excellent cost/accuracy for multiphysics if each sub-operator can be solved efficiently.
- **Why include**
    - Standard second-order splitting; easy to pair with specialized solvers per physics.

### Projection / fractional-step methods for incompressible flow
- **Order**: depends on variant; many common forms are 1st–2nd order.
- **Key reference**: Chorin’s projection method [Chorin68].
- **Expected outcomes**
    - Efficient enforcement of incompressibility via pressure Poisson solve.
- **Why include**
    - Enables scalable segregated solvers; natural for many FE/FD CFD codes.

---

## 7) Catalog E — Variational time discretization (cG/dG in time)
Space–time FE viewpoints treat time with FE-like trial/test spaces; this is attractive for adaptivity and for unified space–time discretizations.

### Continuous Galerkin (cG) time stepping
- **Order (ODE)**: formal order \(2p\) for polynomial degree \(p\) in time (superconvergent) [Walkington14].
- **Lowest order case**: cG(1) corresponds to **Crank–Nicolson** (2nd order) [Walkington08].
- **Expected outcomes**
    - Often good for problems with energy-type stability.
- **Why include**
    - Natural variational structure; good for goal-oriented adaptivity and space–time methods.

### Discontinuous Galerkin (dG) time stepping
- **Order (ODE)**: formal order \(2p+1\) for polynomial degree \(p\) in time (superconvergent) [Walkington14].
- **Lowest order case**: dG(0) corresponds to **Backward Euler** (1st order) [Walkington08].
- **Parabolic PDE reference**: classic analysis for parabolic problems [EJT85].
- **Expected outcomes**
    - Built-in dissipation from jump terms; stable for many dissipative PDEs.
- **Why include**
    - Very flexible adaptivity in time; local hp refinement; robust for diffusion-dominated systems.

---

## 8) Catalog F — Exponential / integrating factor / ETD methods
These target semilinear systems \(u' = Lu + N(u,t)\) where \(L\) is stiff but linear (or efficiently approximated).

### Integrating factor (IF) + RK
- **Order**: depends on RK used (often 2–4)
- **Expected outcomes**
    - Removes stiff linear stability restriction if \(e^{Lt}\) can be applied accurately.
- **Why include**
    - Great for spectral/FE semi-discrete systems with stiff linear diffusion and moderate nonlinearity.

### Exponential time differencing (ETD)
- **ETD-RK family**: ETDRK4 is widely used in stiff PDE contexts [CoxMatthews02].
- **Implementation note**
    - Requires computing \(\varphi\)-functions of \(L\) (matrix functions) or their action on vectors.
- **Expected outcomes**
    - High accuracy per step for problems with a dominant linear stiff part.
- **Stability/robustness note**
    - Coefficient evaluation must be handled carefully; Kassam & Trefethen discuss robust evaluation for ETDRK4 [KassamTrefethen05].
- **Why include**
    - Can outperform implicit methods when \(L\) is linear and diagonalizable/handled by Krylov.

---

## 9) Catalog G — Deferred correction / collocation families (advanced)
### Spectral deferred correction (SDC)
- **Order**: can be increased iteratively (often 2–8 in practice)
- **Expected outcomes**
    - High-order accuracy with iterative correction; flexible for IMEX.
- **Why include**
    - Building block for high-order + adaptivity; can be parallel-in-time friendly.

---

## 10) Catalog H — Second-order structural dynamics (M u¨ + C u˙ + K u = f)
Many structural dynamics problems are naturally second-order. These methods control **numerical dissipation** and **dispersion** and often allow unconditional stability with tunable high-frequency damping.

### H1) Explicit central difference (leapfrog)
- **Order**: 2
- **Type**: explicit, conditionally stable (CFL-like limit based on highest frequency)
- **Expected outcomes**
    - Good for wave propagation with lumped mass; no nonlinear solves.
- **Why include**
    - Standard explicit dynamics workhorse for large problems (contact, impact) when small \(\Delta t\) is acceptable.

### H2) Newmark-β family
- **Order**: typically 2 for common parameter choices
- **Key reference**: Newmark’s original method [Newmark59].
- **Expected outcomes**
    - Wide range of damping/dispersion behavior depending on parameters.
- **Why include**
    - Canonical structural dynamics scheme; basis for many extensions.

### H3) HHT-α (Hilber–Hughes–Taylor)
- **Order**: 2
- **Key property**: introduces controllable numerical dissipation while retaining unconditional stability in common settings.
- **Reference**: original HHT paper [HHT77].
- **Expected outcomes**
    - Better suppression of spurious high-frequency response than Newmark with similar cost.
- **Why include**
    - Essential when you want “implicit + damping” for structural transients.

### H4) Bossak-α / “α-modified Newmark” (often called WBZ-α)
- **Order**: typically 2
- **Reference**: discussion of the Bossak–Newmark extension [Wood80].
- **Expected outcomes**
    - Tunable numerical damping with minimal complexity increase.
- **Why include**
    - Common in commercial FE codes; robust in practice.

### H5) Generalized-α method (structural dynamics)
- **Order**: 2
- **Key property**: user-controlled high-frequency dissipation; popular modern default.
- **Reference**: Chung & Hulbert [CH93].
- **Expected outcomes**
    - Good low-frequency accuracy with damping of high-frequency noise.
- **Why include**
    - Often the best “default implicit structural dynamics” option in production codes.

### H6) Generalized-α for first-order systems (fluids / stabilized FE)
- **Order**: 2
- **Reference**: Jansen–Whiting–Hulbert generalized-α for filtered Navier–Stokes with stabilized FE [JWH00].
- **Why include**
    - Unifies structural-style dissipation control with first-order multiphysics systems.

### H7) Bathe / q1-Bathe family (two-substep methods)
- **Order**: commonly 2 (variant-dependent)
- **Expected outcomes**
    - Designed to improve dissipation/dispersion characteristics in structural dynamics and waves.
- **Why include**
    - Attractive when you want strong control of high-frequency response with good phase accuracy.
- **Reference example**: studies and variants appear in Bathe’s group publications [KwonBathe21].

### H8) Energy–momentum conserving (geometric) methods
- **Goal**: preserve invariants (energy/momentum) in nonlinear elastodynamics.
- **Reference**: Simo & Tarnow’s discrete energy–momentum method [SimoTarnow92].
- **Expected outcomes**
    - Excellent long-time qualitative behavior; may be more expensive per step.
- **Why include**
    - For long-time nonlinear structural dynamics where conservation is critical.

---

## 11) Selection guide (quick recommendations)
Below are **practical defaults** that cover most multiphysics FE needs.

### If you need a “safe default” for stiff multiphysics
- **Backward Euler**: maximum robustness, first-order.
- **TR-BDF2**: robust 2nd order stiff integrator (excellent production default).
- **BDF2 / VSVO-BDF**: extremely good for large stiff PDE/DAE when you want 1 solve/step.

### If you have advection-dominated hyperbolics and want explicit
- **SSPRK3**: robust with limiters/positivity (DG/FV-like).
- **RK4**: high accuracy for smooth solutions when CFL allows.

### If you want energy/phase quality for waves or mechanics
- **Implicit midpoint** or **Gauss IRK** (if cost allows): structure-preserving.
- **Generalized-α**: best default for structural dynamics (tunable damping).

### If your multiphysics naturally splits stiff/nonstiff parts
- **IMEX ARS** or **ARK (Kennedy–Carpenter)**: high-order IMEX with good stability regions.

### If you want adaptive space–time, hp in time, or goal-oriented error control
- **dG/cG in time**: variational time discretizations; good stability structure and adaptivity hooks.

---

## 12) Why these methods belong in a multiphysics FE solver
A solver that aims to be multiphysics-capable typically benefits from including:

1. **At least one extremely robust implicit method** (BE, BDF2/TR-BDF2) for “get me a solution” reliability.
2. **A second-order accurate stiff-friendly method** (TR-BDF2, SDIRK2/3, BDF2) for routine production accuracy.
3. **At least one explicit RK** (SSPRK3 or RK4) for GPU-friendly, hyperbolic-dominated regimes.
4. **IMEX options** for partitioned physics without fully monolithic solves (ARK/ARS).
5. **Structural dynamics family** (Newmark + generalized-α) because second-order systems are common and damping control matters.
6. **A structure-preserving option** (implicit midpoint / energy-momentum) for long-time dynamics.
7. **Space–time/variational options** (cG/dG) if adaptivity, error estimation, or space–time coupling is a goal.

---

## References
> Short keys are used in the text. URLs/DOIs are provided for traceability.

- **[HW96]** Hairer, E.; Wanner, G. *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*. Springer. DOI: 10.1007/978-3-662-09947-6.
- **[AP98]** Ascher, U. M.; Petzold, L. R. *Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations*. SIAM (1998). DOI: 10.1137/1.9781611971392.
- **[Gear71]** Gear, C. W. *Numerical Initial Value Problems in Ordinary Differential Equations*. Prentice-Hall (1971).

- **[Newmark59]** Newmark, N. M. “A Method of Computation for Structural Dynamics.” (1959). (Commonly distributed as a technical report / reprint.)
- **[HHT77]** Hilber, H. M.; Hughes, T. J. R.; Taylor, R. L. “Improved numerical dissipation for time integration algorithms in structural dynamics.” *Earthquake Engineering & Structural Dynamics* 5(3) (1977). DOI: 10.1002/eqe.4290050306.
- **[Wood80]** Wood, W. L. “An alpha modification of Newmark’s method.” *International Journal for Numerical Methods in Engineering* 15(10) (1980). DOI: 10.1002/nme.1620151011.
- **[CH93]** Chung, J.; Hulbert, G. M. “A time integration algorithm for structural dynamics with improved numerical dissipation: the generalized-α method.” *Journal of Applied Mechanics* 60(2) (1993). DOI: 10.1115/1.2900803.
- **[JWH00]** Jansen, K. E.; Whiting, C. H.; Hulbert, G. M. “A generalized-α method for integrating the filtered Navier–Stokes equations with a stabilized finite element method.” *Computer Methods in Applied Mechanics and Engineering* (2000).

- **[HoseaShampine96]** Hosea, M. E.; Shampine, L. F. “Analysis and implementation of TR-BDF2.” *Applied Numerical Mathematics* 20(1–2) (1996). DOI: 10.1016/0168-9274(95)00115-8.
- **[ARS97]** Ascher, U. M.; Ruuth, S. J.; Spiteri, R. J. “Implicit-explicit Runge–Kutta methods for time-dependent partial differential equations.” *Applied Numerical Mathematics* 25(2–3) (1997). DOI: 10.1016/S0168-9274(97)00056-1.
- **[KC01]** Kennedy, C. A.; Carpenter, M. H. “Additive Runge–Kutta schemes for convection–diffusion–reaction equations.” NASA technical report (2001).
- **[KC03]** Kennedy, C. A.; Carpenter, M. H. “Additive Runge–Kutta schemes for convection–diffusion–reaction equations.” *Applied Numerical Mathematics* 44 (2003). DOI: 10.1016/S0168-9274(02)00138-1.
- **[GST01]** Gottlieb, S.; Shu, C.-W.; Tadmor, E. “Strong Stability-Preserving High-Order Time Discretization Methods.” *SIAM Review* 43(1) (2001). DOI: 10.1137/S003614450036757X.
- **[ShuOsher88]** Shu, C.-W.; Osher, S. “Efficient implementation of essentially non-oscillatory shock-capturing schemes.” *J. Comput. Phys.* 77 (1988).

- **[Alexander77]** Alexander, R. “Diagonally Implicit Runge–Kutta Methods for Stiff O.D.E.’s.” *SIAM J. Numerical Analysis* 14(6) (1977). DOI: 10.1137/0714068.
- **[Strang68]** Strang, G. “On the construction and comparison of difference schemes.” *SIAM J. Numerical Analysis* 5(3) (1968). DOI: 10.1137/0705041.
- **[Chorin68]** Chorin, A. J. “Numerical solution of the Navier–Stokes equations.” (1968). (Projection method original reference.)
- **[EJT85]** Eriksson, K.; Johnson, C.; Thomée, V. “Time discretization of parabolic problems by the discontinuous Galerkin method.” *M2AN* 19(4) (1985).
- **[Walkington08]** Walkington, N. J. “Compactness properties of the DG and CG time stepping schemes.” (2008). (Notes lowest-order DG/CG correspond to implicit Euler / Crank–Nicolson.)
- **[Walkington14]** Walkington, N. J. “DG–CG time stepping for wave equations.” *SIAM J. Numerical Analysis / SINUM* (2014). (States formal ODE order \(2p\) for CG and \(2p+1\) for DG time stepping.)
- **[CoxMatthews02]** Cox, S. M.; Matthews, P. C. “Exponential time differencing for stiff systems.” *J. Comput. Phys.* 176 (2002). DOI: 10.1006/jcph.2002.6995.
- **[KassamTrefethen05]** Kassam, A.-K.; Trefethen, L. N. “Fourth-order time-stepping for stiff PDEs.” *SIAM J. Sci. Comput.* 26(4) (2005). DOI: 10.1137/S1064827502410633.
- **[SimoTarnow92]** Simo, J. C.; Tarnow, N. “The discrete energy-momentum method. Conserving algorithms for nonlinear elastodynamics.” *ZAMP* 43 (1992). DOI: 10.1007/BF00913408.
- **[LangROW2020]** Lang, J. “Rosenbrock-Wanner Methods: Construction and Mission.” arXiv:2002.12028 (2020).
- **[HairerGNI]** Hairer, E.; Lubich, C.; Wanner, G. *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*. Springer (2nd ed.). DOI: 10.1007/3-540-30666-8.
- **[KwonBathe21]** Kwon, S. B.; Bathe, K. J.; Noh, G. “q1-Bathe time integration scheme” (various publications; see MIT Bathe group papers).
