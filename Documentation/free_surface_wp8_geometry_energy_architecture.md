# WP-8 geometry-coupling and energy architecture record

Status: selected implementation direction; qualification incomplete.

Scope: one-phase unfitted free-surface geometry coupling, nonlinear
convergence, and the work/energy account required by FSR-09.

## Decision

AD-5 selects the partitioned generated-state outer iteration, not a complete
fixed-topology shape tangent.

The generated state is denoted by

\[
G(u) =
\{\text{cut snapshot},\text{curvature projection},
  \text{state-dependent constraints},\text{extension map}\}.
\]

For outer iteration \(k\), the implementation regenerates \(G_k=G(u_k)\),
projects the algebraic state through the regenerated constraints, regenerates
the remaining state from that projected value, and solves

\[
R(u_{k+1};G_k)=0
\]

with \(G_k\) frozen inside the inner Newton solve. The next outer pass restores
the immutable entry history and rate state, regenerates \(G\) from the new
algebraic iterate, and repeats. Acceptance requires a fresh outer pass for
which the inner problem already satisfies every configured absolute residual
criterion and therefore takes zero Newton updates.

This zero-update certificate establishes a same-state algebraic fixed point
for the regenerated residual in the tested finite-dimensional fixtures. It
does not by itself establish contraction, uniqueness, or an energy-stability
theorem.

The production default is the generated-state outer fixed point with a
bounded outer-iteration count. Refreshed within-solve and per-step-only modes
remain comparison controls; they are not the selected WP-8 qualification
path.

## Production transient-scheme prerequisite

The new solver now accepts an optional general parameter:

```xml
<Transient_time_integration_scheme>BackwardEuler</Transient_time_integration_scheme>
```

The exact supported values are `GeneralizedAlpha` and `BackwardEuler`;
aliases, case changes, and all other values are rejected before the time
history is repacked or a transient solve begins. Omitting the parameter
preserves the `GeneralizedAlpha` default. The spectral-radius parameter is
optional with its existing default of `0.5` and is validated only for
generalized-alpha.

Backward Euler has no spectral-radius parameter. Production reports it as
inapplicable, disables generalized-alpha PDE-rate initialization, uses
`alpha_f=1` for generated free-surface/contact state, and leaves
bound-preserving maintenance independent of a time-integration rate factor.
The external-state fixed point therefore observes the accepted endpoint time
and backward-difference stencil directly. The one-step regression holds the
candidate endpoint uncommitted through the preaccept and commit-ready
callbacks, then verifies that history advances exactly once.

This is a production method-selection and endpoint-transaction prerequisite.
It does not establish the backward-Euler constant-surface-tension energy
balance, a refinement threshold, or any WP-8 simulation exit.

## Why the complete-tangent alternative is not selected

`RefreshedFrozenQuadrature` is a local geometry-tangent policy name, not a
statement that the complete AD-5 tangent exists. Current support includes
selected point-location, measure, normal, cut-volume, and surface-functional
first variations where they are explicitly installed. It does not cover the
complete derivative of regenerated high-order rules, curvature projection,
wet-wall/contact construction, extension-map changes, or topology
transitions.

The surface-stress registration path rejects the advertised differentiated
quadrature mode where projector, normal, point-location, measure, or curvature
coverage is incomplete. The selected outer iteration is therefore the only
honest system-level AD-5 direction in the present implementation.

The fixed-topology first-variation fixture in
`FreeSurfaceGeometrySnapshot.DiscreteFunctionalFirstVariationMatchesCentralDifference`
is retained as a capillary-functional prerequisite. It is not evidence for a
complete monolithic Jacobian.

## Outer-iteration contract

The selected path currently enforces these low-level properties:

1. The constraint-construction callback runs before the projected-state
   callback, so all other generated fields observe the constrained iterate.
2. Every inner Newton problem holds its generated state fixed.
3. Relative residual tolerances are disabled for the inner problems because a
   refresh changes the residual reference.
4. A freshly regenerated residual, not convergence of the preceding frozen
   problem, determines outer convergence.
5. A failed outer pass restores solution, history, rate, auxiliary, bordered,
   and regenerated callback state.
6. A constraint-topology change invalidates the relevant algebraic caches and
   can reallocate the Jacobian sparsity.

The current report contains the outer count, accumulated inner iterations,
last outer algebraic state-change norm, and refreshed residual. It does not
record a norm for each component of \(G\), a contraction ratio, or a
predeclared contraction gate. Consequently the required \(h\), \(dt\), cut
shift, and MPI contraction study remains open.

## Required complete energy account

For an accepted step, the target stored energy is

\[
\mathcal E =
K + E_g + \gamma A_{lg,h}
- \sum_w \gamma\cos(\theta_{e,w})A_{sl,h,w}
+ E_{\mathrm{gas/compressibility}},
\]

where the final term is present only for a model that contains it. The
physical dissipation is

\[
\mathcal D =
D_{\mathrm{viscous}} + D_{\mathrm{Navier}}
+ D_{\mathrm{line}}.
\]

With external work \(W_{\mathrm{ext}}\) and explicitly classified numerical
work \(W_{\mathrm{num}}\), the accepted-step residual must use one declared
time stage and sign convention:

\[
\mathcal R_E =
\mathcal E^{n+1}-\mathcal E^n
+ \Delta t\,\mathcal D
- W_{\mathrm{ext}}-W_{\mathrm{num}}.
\]

The numerical-work ledger must separately expose VMS/PSPG, cut stabilization
or aggregation, extension, pruning, limiting, redistancing, local
reconciliation, and global correction. A closed dissipative case with no
external work must not hide positive unexplained growth behind a later
maintenance operation. Every transport and maintenance substage therefore
needs its own before/after state and work row.

Rejected and rolled-back attempts do not change the accepted state. Their
trial ledgers must be kept separate and their contribution to the accepted
balance must be exactly zero.

## Current energy evidence and its boundary

The authoritative geometry snapshot currently supports the discrete
functional

\[
\gamma A_{lg,h}
- \sum_w \gamma\cos(\theta_{e,w})A_{sl,h,w}
+ \lambda V_h
\]

and its fixed-topology first variation. Accepted records bind that functional
to the state and snapshot revisions. Accepted dynamic-contact records also
contain sharp contact measure, line-friction dissipation, sharp wetted-wall
measure, and wall-slip dissipation. Global volume correction reports the
surface, Young-wall, and volume-potential change caused by the correction.

The level-set maintenance ledger now records the authoritative
surface/wall/volume potential before and after each implemented transport,
limiting, reinitialization, geometry-reconciliation, and global-correction
substage. Rows form one continuous algebraic and functional-state chain.
Snapshot, mesh-topology, communicator-wide cut-topology, and extension-map
revisions remain distinct. Accepted rows are published only after their
physical geometry transaction commits. Rejected rows and explicit zero-row
attempts remain diagnostic records with exactly zero accepted contribution.
The complete FE-ordered algebraic fingerprint and authoritative functional
values are required to agree across the FE communicator.

These are necessary maintenance channels, not a complete balance. The present
account does not combine kinetic, gravitational, gas/compressibility, bulk
viscous, complete Navier-slip and line-friction dissipation, external
pressure, body-force, VMS/PSPG, ghost-penalty or aggregation, extension, and
pruning terms into one accepted-step residual. The extension-map revision is
provenance, not extension numerical work. No backward-Euler
constant-surface-tension energy identity or refinement gate has been
demonstrated. Generalized-alpha stage consistency is therefore also
unqualified.

## Stage and topology policy still required

The nonlinear synchronization callback receives the solve time and time
integration context, and accepted dynamic-contact state has explicit stage
time and stage fraction. That provenance does not yet prove that geometry,
surface/wall/contact work, transport, and every maintenance contribution use
one common stage.

A cut-topology change is currently detected and labeled as a nonsmooth event.
The diagnostic correctly states that ordinary Newton consistency is not
expected across the event, but detection is not an acceptance policy. WP-8
still requires:

- snapshot and extension invalidation rules;
- nonlinear restart or time-step rejection;
- an accepted-versus-trial event ledger;
- energy-jump classification;
- a minimum resolved-feature policy; and
- serial/MPI equivalence of the chosen action.

## Frozen prerequisite evidence

The low-level matrix is
`tests/cases/fluid/free_surface_wp8_energy_qualification_matrix.json`, with
`tests/cases/fluid/run_free_surface_wp8_energy_qualification.py` as its strict
wrapper. Its exact frozen SHA-256 is
`9007fb06e64cf092d2d57e6ea49fda5fe99798e1b52b185e9d1e3fba0bb9e9b6`.
Any byte drift is rejected before structural validation or test discovery. It
freezes only:

- exact production parsing and selection of generalized-alpha or backward
  Euler, including backward-Euler spectral-radius inapplicability;
- the backward-Euler one-step endpoint fixed-point and history transaction;
- the fixed-topology surface/wall/volume first-variation fixture and exact
  ghost exclusion;
- regenerated-residual outer convergence in a scalar fixture;
- transactional outer rollback and constraint-topology refresh;
- the selected frozen-inner Jacobian provenance;
- authoritative accepted functional history;
- global-correction functional-work reporting;
- continuous maintenance row chains with separate transport, limiting,
  reinitialization, geometry-reconciliation, and global-correction substages;
- accepted/rejected transaction publication, including explicit zero-row
  outcomes and zero accepted contribution for rejected attempts;
- separate snapshot, mesh-topology, cut-topology, algebraic-state, and
  extension-map revision provenance;
- bounded extension refresh prerequisites; and
- two-rank deterministic maintenance rows, complete-vector algebraic
  provenance, and communicator-wide geometry refresh.

The only accepted claim is `low_level_prerequisite`. The wrapper rejects every
claim ending in `_closure`, explicitly rejects FSR-09, WP-8, Q3, Q4, Q5, and
complete-energy requests, and does so before build, binary discovery, test
execution, or artifact creation.

## Unqualified exits

The registry machine-records all open method and simulation exits. The method
exits cover the partitioned energy argument, backward-Euler balance, complete
accepted energy/dissipation/external/numerical-work identity, common-stage
contract, same-state full-package refresh neutrality, geometry-dependent
residual directional derivatives, contraction refinement, topology policy,
generalized-alpha consistency, and a prospectively frozen energy-residual
threshold. The maintenance-only accepted/rejected separation and five
implemented substage rows are no longer listed as missing, but they do not
close any of those broader exits.

Production availability of backward Euler removes only the method-selection
prerequisite. The backward-Euler balance exit remains
`REQUIRED_NOT_CLAIMED`; no result in this matrix supplies a free-surface
energy residual.

The six required simulations remain entirely unclaimed:

- static cap;
- capillary relaxation;
- linear capillary wave;
- droplet oscillation;
- sloshing; and
- wetting relaxation.

Each simulation must independently refine the declared spatial and temporal
scales, include cut shifts and representative MPI partitions, publish the
complete ledger, and demonstrate convergence of the unexplained energy
residual. Until those records pass, FSR-09 and WP-8 remain open.

## Source evidence map

- Outer fixed point and transactional restoration:
  `Code/Source/solver/FE/TimeStepping/NewtonSolver.{h,cpp}`.
- Generated-state selection and synchronization:
  `Code/Source/solver/Application/Core/ApplicationDriver.cpp`.
- Transient-scheme XML defaults and parsing:
  `Code/Source/solver/Parameters.{h,cpp}`.
- Backward-Euler endpoint transaction:
  `Code/Source/solver/FE/TimeStepping/TimeLoop.cpp` and
  `Code/Source/solver/FE/Tests/Unit/TimeStepping/test_TimeLoopConvergence.cpp`.
- Maintenance ledger contract:
  `Code/Source/solver/Application/Core/ApplicationDriver.{h,cpp}`.
- Maintenance transaction and communicator regressions:
  `Code/Source/solver/Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflows{,MPI}.cpp`.
- Geometry policy limits:
  `Code/Source/solver/FE/LevelSet/LevelSetInterfaceLifecycle.cpp`.
- Surface/wall/volume functional and first variation:
  `Code/Source/solver/FE/Interfaces/FreeSurfaceGeometrySnapshot.{h,cpp}`.
- Accepted functional/contact records:
  `Code/Source/solver/FE/Systems/FESystem.{h,cpp}`.
- Surface-stress tangent containment:
  `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`.
