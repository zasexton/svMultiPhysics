# FE/Forms - Vocabulary Expansion Roadmap

This document is the roadmap for **future vocabulary work** in `FE/Forms`.
It is intentionally narrower and more execution-oriented than a generic
"everything a PDE DSL could someday contain" checklist.

For what is **implemented and intended for public use today**, see
`VOCABULARY.md`.

This roadmap follows a few rules:

- Prefer **cross-cutting primitives** over PDE-specific helper catalogs.
- Do not treat already-implemented features as roadmap items; promote them into
  status/docs instead.
- Every roadmap item should identify **module ownership**, **prerequisites**,
  and **acceptance criteria**.
- Avoid ambiguous names. In particular, `trace(...)` already means **tensor
  trace**, so facet/interface restriction operators must use different names.

## 1. Status Model and Documentation Contract

This file is not the capability matrix. The vocabulary story should be tracked
using four status buckets:

- `implemented/public`
- `implemented/internal`
- `planned`
- `out-of-scope`

Recommended document roles:

- `VOCABULARY.md`: single source of truth for `implemented/public`
- capability matrix (same file or adjacent table): all four status buckets
- this file: only `planned` work and sequencing

### Immediate documentation-convergence work

Before adding more roadmap surface area, the current status/docs need to be
reconciled. The following items already exist in public headers and/or tests and
should be documented as current capability rather than left implicit:

- interface-face measure: `.dI(marker)`
- time terminals: `t`, `deltat`, `deltat_eff`
- solution-history access: `previousSolution(k)`, `historyWeightedSum(...)`,
  `historyConvolution(...)`
- auxiliary/coupled-state terminals: `auxiliaryState(name)`
- matrix/spectral operators: `matrixExp`, `matrixLog`, `matrixSqrt`,
  `matrixPow`, eigenvalue/eigenvector/spectral decomposition operators
- smooth regularization operators: `smoothAbs`, `smoothSign`,
  `smoothHeaviside`, `smoothMin`, `smoothMax`
- constitutive ergonomics already supported today:
  - n-ary inputs
  - indexed multi-output access via `ConstitutiveCall::out(i)`

The following items should be classified as `implemented/internal` until a
stable ergonomic API exists:

- `materialStateOldRef(offset_bytes)`
- `materialStateWorkRef(offset_bytes)`
- other slot-/offset-based symbolic terminals meant for setup-time lowering

## 2. Recommended Sequencing

The roadmap should be executed in this order:

1. Documentation convergence and status cleanup
2. Cross-cutting foundation primitives
3. Mixed-dimensional / embedded coupling and ALE as major epics
4. Advanced math and constitutive ergonomics
5. Optional FEEC/manifold branch

Long-tail domains such as optimization, stochastic/UQ, and nonlocal PDEs should
remain deferred until the foundation tracks below are complete.

## 3. Priority 1 - Cross-Cutting Foundation Primitives

These are the vocabulary pieces that unlock many future physics modules at once.
They should be prioritized ahead of broad new domain-specific helper families.

### 3.1 Facet / Interface Trace Algebra

Owner: **Forms** + **Assembly** + **Spaces**

Why this matters:

- H(div) and H(curl) formulations need normal/tangential trace semantics
- slip walls, contact, shells, Nitsche terms, and Maxwell-type interface laws
  all become awkward without explicit trace/project/jump operators
- raw `inner(..., n.minus())` expressions do not scale as a user-facing story

Roadmap items:

- first-class trace operators distinct from tensor `trace(...)`
  - `facetTrace(expr, side)`
  - or explicit `minus(expr)` / `plus(expr)` helpers as the base restriction
- normal/tangential trace family
  - `normalTrace(u)`
  - `tangentialTrace(u)`
- projection helpers
  - `projectNormal(u)`
  - `projectTangent(u)`
- jump/average family beyond scalar DG basics
  - `normalJump(u)`
  - `tangentialJump(u)`
  - `weightedAverage(u, w_minus, w_plus)`
- orientation-aware terminals
  - side-specific normals
  - side orientation/sign conventions
  - explicit interface orientation metadata where needed

Acceptance criteria:

- common H(div)/H(curl) interface terms can be written without raw component
  manipulation against `n.minus()`
- slip and Nitsche-style boundary/interface forms are authorable with explicit
  normal/tangential vocabulary
- the orientation model is documented and validated on both interior faces and
  registered interface faces

### 3.2 State / History / Stage Access

Owner: **Forms** + **Systems** + **TimeStepping** + **Constitutive**

Why this matters:

- viscoelasticity, plasticity, phase change, rate laws, and generalized-alpha /
  BDF schemes require direct access to history and state
- low-level slot/offset terminals exist, but they are not an ergonomic public
  DSL
- coupled ODE-PDE models need named auxiliary/state access, not just
  infrastructure hooks

Roadmap items:

- field-oriented history accessors
  - `prev(u, k)` or `u_prev(k)`
  - stage accessors such as `stage(u, i)` where time integrators expose stage
    states
- named auxiliary/coupled-state access
  - `aux(name)`
  - named coupled integral references where public authoring needs them
- named material-state access driven by `StateLayout`
  - `stateOld(name)`
  - `stateWork(name)`
- explicit public documentation for current time terminals and history
  operators

Acceptance criteria:

- transient weak forms can be written using named/history-aware vocabulary
  rather than raw slot/offset terminals
- constitutive/stateful models can author rate laws against named state fields
- stage-aware schemes do not require physics modules to bypass the DSL

### 3.3 Generalized Measure Model

Owner: **Forms** + **Assembly/Quadrature**

Why this matters:

- current public measure story is too narrow for embedded physics, singular
  sources, and nonmatching couplings
- interface-heavy and mixed-dimensional workflows need more than `dx`, `ds`,
  and `dS`
- quadrature control should attach to a measure, not only to global assembly
  configuration

Roadmap items:

- keep existing core measures
  - `dx`
  - `ds(boundary_marker)`
  - `dS`
  - `dI(interface_marker)` as current implemented interface-face measure
- add generalized domain selection
  - subdomain-set measures
  - boundary-set measures
  - interface-pair / interface-set measures
- add codimension-aware measures
  - codimension-2 edge/curve measures
  - point measures
  - embedded manifold measures
- add per-measure quadrature metadata
  - quadrature rule/order override
  - optional advanced/debug quadrature terminals only where justified

Acceptance criteria:

- singular sources and embedded-source problems can be expressed without
  handwritten kernels
- interface and embedded problems can attach quadrature behavior to the measure
  itself
- the public docs reflect both current measures and planned generalized ones

## 4. Priority 2 - Mixed-Dimensional and Embedded-Domain Coupling

Owner: **Systems/Assembly** + **Spaces** + **Forms**

Why this matters:

- this is the missing bridge for 3D-1D vessel models, fractures, immersed
  membranes, fibers, and 0D/1D/3D couplings
- the current roadmap has `EmbeddedDomain`, `TraceSpace`, and `MortarSpace`
  concepts, but not the actual bulk-to-lower-dimensional operators users need

Roadmap items:

- restriction/trace from a bulk field to a lower-dimensional manifold
  - `restrictTo(manifold, u)`
  - `traceTo(manifold, u)`
- extension / lifting / prolongation back to the bulk
  - `extendFrom(manifold, lambda)`
  - `lift(...)`
- transfer operators between spaces of different topological dimension
- codimension-aware measures integrated with the generalized measure model
- same-mesh and nonmatching interface workflows as distinct sub-phases

Suggested phases:

- Phase 2a: same-mesh codimension-1 coupling on registered interfaces
- Phase 2b: nonmatching codimension-1 coupling with `MortarSpace`
- Phase 2c: codimension-2 and mixed 0D/1D/3D transfer operators

Acceptance criteria:

- a bulk field can be restricted onto a lower-dimensional space and coupled back
  without ad hoc callback code
- transfer operators are explicit vocabulary concepts, not implicit assembly
  behavior
- the roadmap distinguishes same-mesh trace coupling from genuine embedded or
  nonmatching coupling

## 5. Priority 3 - ALE / Moving Domains

Owner: **Assembly/Geometry** + **Systems/TimeStepping** + **Forms**

Why this matters:

- `ALEMapping` as metadata is not enough for physics authors
- free-surface flow, moving-boundary diffusion, FSI, and shape evolution need
  explicit moving-domain terminals and transforms

Roadmap items:

- mesh/domain motion terminals
  - mesh displacement
  - mesh velocity / domain velocity
- moving-domain differential operators
  - material derivative
  - optional convected derivatives if they become a real requirement
- reference/current configuration geometry
  - current normal
  - reference normal
  - current measure
  - reference measure
- mapping transforms
  - surface Jacobian
  - Nanson-type transforms
  - current/reference pullback/pushforward semantics where needed

Acceptance criteria:

- moving-domain residuals can be authored without manual coefficient plumbing
- free-surface and FSI formulations can distinguish reference and current
  geometry in the DSL
- ALE is treated as a first-class vocabulary branch, not just geometry metadata

## 6. Priority 4 - Advanced Math and Constitutive Ergonomics

### 6.1 Scalar Nonlinear Function Expansion

Owner: **Forms**

Why this matters:

- manufactured solutions, periodic forcing, reaction kinetics, turbulence
  closures, contact regularization, and phase-field models require a broader
  scalar math set

Roadmap items:

- trigonometric functions
  - `sin`
  - `cos`
  - `tan`
- hyperbolic functions
  - `sinh`
  - `cosh`
  - `tanh`
- inverse trig functions
  - `atan`
  - `atan2`
- special functions likely worth supporting
  - `erf`
  - `erfc`
- sign/part/complementarity helpers
  - `positivePart`
  - `negativePart`
  - smooth complementarity/projector operators where needed

Acceptance criteria:

- the roadmap clearly separates exact nonsmooth operators from smooth
  regularizations
- new scalar functions ship with AD and JIT/lowering requirements, not just AST
  node names

### 6.2 Constitutive Ergonomics and State Layout

Owner: **Constitutive** + **Forms**

Why this matters:

- the code already supports richer constitutive calls than the public story
  suggests
- multi-input/multi-output and named state layout are required for realistic
  thermo-chemo-electro-mechanical materials and closure models

Roadmap items:

- document current support explicitly
  - n-ary inputs
  - multi-output calls
  - indexed output selection
- add named-output metadata and public ergonomics
  - `call.out("stress")`
  - output descriptors for tooling/docs
- connect named material-state access to `StateLayout`
- keep explicit state-layout and output-shape metadata in the roadmap rather
  than describing constitutive models as unary black boxes

Acceptance criteria:

- multi-field constitutive models do not collapse into opaque custom callback
  code
- named outputs/state fields can be referenced from public DSL code

## 7. Optional Strategic Track - FEEC / Manifold Formulations

Owner: **Forms** + **Spaces** + **Geometry**

This branch should remain **optional** unless FEEC/manifold physics is an
explicit product goal.

If FEEC/manifold formulations are a real target, the roadmap should include a
minimal complete set:

- `DifferentialForm`
- `ExteriorDerivative`
- `Codifferential`
- `HodgeStar`
- `Wedge`
- `Pullback`
- `Pushforward`

If that branch is not a near-term goal, it should be marked
`out-of-scope-for-now` rather than left half-specified.

## 8. Deferred / Later Branches

These branches stay deferred until the foundation tracks above are complete:

- optimization and inverse-problem vocabulary
- stochastic / uncertainty-quantification vocabulary
- nonlocal / integral PDE vocabulary
- broad solver-interaction or adjoint vocabulary beyond what current workflows
  already require

These are important long-term areas, but they should not outrank the missing
cross-cutting primitives that block many core physics modules today.

## References / Inspirations

- Alnaes et al. - "Unified form language." *ACM Transactions on Mathematical
  Software* (2014). DOI: 10.1145/2566630.
- Logg, Mardal, Wells (eds.) - *Automated Solution of Differential Equations by
  the Finite Element Method* (2012).
- Rathgeber et al. - "Firedrake." *ACM Transactions on Mathematical Software*
  (2016). DOI: 10.1145/2998441.
- Prud'homme et al. - "Feel++ : A computational framework for Galerkin Methods
  and Advanced Numerical Methods." *ESAIM: Proceedings* (2012). DOI:
  10.1051/proc/201238024.
- Janssens et al. - "Finite Element Assembly Using an Embedded Domain Specific
  Language." *Scientific Programming* (2015). DOI: 10.1155/2015/797325.
- MFEM - Anderson et al. - "MFEM: A modular finite element methods library."
  *Computers & Mathematics with Applications* (2021). DOI:
  10.1016/j.camwa.2020.06.009.
- deal.II - Arndt et al. - "The deal.II finite element library: Design,
  features, and insights." *Computers & Mathematics with Applications* (2021).
  DOI: 10.1016/j.camwa.2020.02.022.
