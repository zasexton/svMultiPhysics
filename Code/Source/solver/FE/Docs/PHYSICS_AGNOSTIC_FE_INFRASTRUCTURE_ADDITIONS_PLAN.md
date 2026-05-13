# FE-Only Infrastructure Additions Plan

## Summary

Add only physics-agnostic FE infrastructure that can support bioreactor,
multiphase, species, RANS, PBM, and future physics without encoding oxygen,
gas bubbles, impellers, or cell-culture rules in FE.

The FE additions should be limited to:

1. State grouping and field-shape metadata.
2. Generic admissibility and bounded-state checking hooks.
3. Generic bounded scalar update / limiter integration points.
4. Generic paired-exchange balance helpers.
5. Generic threshold, histogram, and percentile reductions.
6. Generic accumulated-exposure tracking.
7. Generic wall-distance / nearest-boundary geometry query service.
8. Small hardening around existing moving-frame and sliding-interface
   infrastructure.

Do not add new FE APIs for `O2`, `kLa`, `bubble`, `gas holdup`, `PBM`, `RANS`,
`MRF`, or `bioreactor`. Those belong in Physics modules built on this
infrastructure.

## Key FE Additions

### 1. State Group And Field-Shape Metadata

Goal: allow Physics modules to declare that multiple FE fields form one logical
state, such as species vectors, phase fractions, PBM bins, turbulence variables,
or constrained components.

Checklist:

- [x] Add a generic `StateGroup` metadata type owned by FE.
- [x] Support group kinds:
  - [x] `IndependentFields`
  - [x] `ConservedComponents`
  - [x] `VolumeFractions`
  - [x] `MomentOrBinSet`
  - [x] `AuxiliaryCoupledState`
- [x] Add generic shape metadata:
  - [x] scalar field
  - [x] vector field
  - [x] tensor field
  - [x] indexed scalar set
  - [x] mixed field group
- [x] Connect state groups to existing `FieldRegistry` without changing current
      field registration behavior.
- [x] Allow each group to declare:
  - [x] field names
  - [x] component count
  - [x] optional conserved quantity name
  - [x] optional sum constraint
  - [x] optional lower/upper bounds
  - [x] optional analysis tags
- [x] Ensure all metadata is optional so existing solvers remain unchanged.
- [x] Add unit tests proving:
  - [x] existing field registration still works
  - [x] grouped fields resolve by name
  - [x] missing group fields are rejected
  - [x] group metadata can be queried by analysis and postprocessing code

Implementation notes:

- Reuse `FieldRegistry` instead of creating a parallel field registry.
- Keep this metadata declarative only.
- Do not enforce physics constraints here; enforcement comes through
  admissibility/update hooks.

### 2. Generic State Admissibility Hooks

Goal: give FE a reusable way to check whether a field or state group is valid
without knowing the physics.

Checklist:

- [x] Add `StateAdmissibilityDescriptor`.
- [x] Support admissibility types:
  - [x] lower bound
  - [x] upper bound
  - [x] interval bound
  - [x] sum equality
  - [x] sum inequality
  - [x] finite-value requirement
  - [x] user-provided residual callback
- [x] Support scopes:
  - [x] nodal values
  - [x] cell averages
  - [x] quadrature values
  - [x] region aggregates
  - [x] global aggregate
- [x] Allow descriptors to attach to either one field or a `StateGroup`.
- [x] Add an FE utility to evaluate admissibility residuals after assembly or
      update.
- [x] Connect results to existing `InvariantDomainSummary` analysis metadata.
- [x] Add options for warning-only versus hard failure.
- [x] Add unit tests for:
  - [x] scalar nonnegativity
  - [x] bounded scalar interval
  - [x] phase-like sum constraint
  - [x] group-level violation reporting
  - [x] analysis summary emission

Implementation notes:

- FE checks validity; Physics defines what validity means.
- Do not add oxygen saturation, gas volume fraction, or turbulence-specific
  admissibility logic.
- Use existing invariant-domain analysis machinery rather than creating a new
  analysis family.

### 3. Bounded Scalar Update And Limiter Integration Points

Goal: allow Physics modules to plug in bounded update strategies without FE
owning the limiter formulas.

Checklist:

- [x] Add a generic `BoundedUpdatePolicy` interface.
- [x] Provide built-in FE policies:
  - [x] `None`
  - [x] `CheckOnly`
  - [x] `ClampCellAverage`
  - [x] `RejectStepOnViolation`
- [x] Add extension points for Physics-owned limiters:
  - [x] pre-update callback
  - [x] post-update callback
  - [x] candidate-state filter
  - [x] admissibility residual callback
- [x] Ensure the policy can operate on:
  - [x] one scalar field
  - [x] a state group
  - [x] auxiliary state
- [x] Integrate policy checks into the relevant time-update path without
      changing default behavior.
- [x] Emit invariant-domain metadata when a bounded policy is active.
- [x] Add tests for:
  - [x] no-op default behavior
  - [x] rejection on negative scalar
  - [x] warning-only mode
  - [x] cell-average clamp mode
  - [x] group-level bound check
  - [x] compatibility with existing time history

Implementation notes:

- FE may provide simple generic safety policies.
- High-resolution FCT, positivity-preserving DG, flux limiters, PBM limiters,
  and phase-fraction limiters belong in Physics or specialized modules.

### 4. Generic Paired-Exchange Balance Helper

Goal: support conservative exchange terms between fields, phases, species, or
auxiliary states without duplicating equal-and-opposite source bookkeeping.

Checklist:

- [ ] Add a `PairedExchangeDescriptor`.
- [ ] Allow descriptor to identify:
  - [ ] donor field or group
  - [ ] receiver field or group
  - [ ] exchanged quantity name
  - [ ] sign convention
  - [ ] source callback or form contribution id
  - [ ] conservation tolerance
- [ ] Add helper utilities to register equal-and-opposite residual
      contributions.
- [ ] Connect descriptor to existing `BalanceDescriptor`, `FluxBalanceSummary`,
      and conservation analysis.
- [ ] Support local and global exchange checks.
- [ ] Support optional weighted exchange, for example density-weighted or
      volume-fraction-weighted, without hard-coding the weights.
- [ ] Add tests for:
  - [ ] equal-and-opposite scalar exchange
  - [ ] multi-field group exchange
  - [ ] failed conservation tolerance
  - [ ] conservation metadata emission
  - [ ] no false conservation claim when metadata is absent

Implementation notes:

- This should be named generically, not as interphase mass transfer.
- Henry-law, kLa, drag, heat transfer, chemical reaction, and PBM source laws
  stay outside FE.

### 5. Threshold, Histogram, And Percentile Reductions

Goal: support generic reporting such as "volume below threshold",
"95th percentile", and "histogram of a scalar field" for any FE quantity.

Checklist:

- [ ] Extend derived result / quantity infrastructure with generic reduction
      types:
  - [ ] `ThresholdMeasure`
  - [ ] `ThresholdIntegral`
  - [ ] `Histogram`
  - [ ] `Percentile`
  - [ ] `MinMaxPercentileSummary`
- [ ] Support reductions over:
  - [ ] whole domain
  - [ ] named region
  - [ ] boundary
  - [ ] cell set
- [ ] Allow reduction inputs from:
  - [ ] FE fields
  - [ ] derived fields
  - [ ] auxiliary fields
  - [ ] expression/form-evaluated quantities
- [ ] Define deterministic binning and tie handling for histograms/percentiles.
- [ ] Support volume-weighted and unweighted modes.
- [ ] Add unit tests for:
  - [ ] thresholded area/volume
  - [ ] weighted percentile
  - [ ] histogram bin counts
  - [ ] empty-region behavior
  - [ ] parallel/reduction consistency if MPI tests exist
  - [ ] compatibility with existing derived result output

Implementation notes:

- FE should expose generic names like threshold volume and percentile.
- Physics can later register bioreactor outputs such as low-DO volume or
  high-shear exposure using these generic reductions.

### 6. Accumulated Exposure Tracking

Goal: support generic accumulation of time spent above or below a threshold,
without encoding biological damage or turbulence meanings.

Checklist:

- [ ] Add an `ExposureAccumulator` utility.
- [ ] Support accumulation forms:
  - [ ] time above threshold
  - [ ] time below threshold
  - [ ] integral of positive excess
  - [ ] integral of squared positive excess
  - [ ] running maximum
  - [ ] running minimum
- [ ] Support storage scopes:
  - [ ] node
  - [ ] cell
  - [ ] quadrature point
  - [ ] region aggregate
- [ ] Reuse existing auxiliary history/state storage where possible.
- [ ] Allow the driving quantity to be:
  - [ ] field value
  - [ ] derived expression
  - [ ] auxiliary value
  - [ ] callback-evaluated quantity
- [ ] Add restart-safe serialization if auxiliary state restart support already
      exists.
- [ ] Add tests for:
  - [ ] constant signal accumulation
  - [ ] threshold crossing over multiple time steps
  - [ ] reset behavior
  - [ ] QP/cell/node storage modes
  - [ ] integration with time history

Implementation notes:

- FE tracks exposure mathematically.
- Physics defines what quantity is hazardous or meaningful.

### 7. Generic Wall-Distance / Boundary-Distance Query

Goal: provide reusable geometric support needed by wall models, turbulence
closures, immersed methods, and near-wall postprocessing.

Checklist:

- [ ] Add a generic `BoundaryDistanceService`.
- [ ] Support nearest distance to:
  - [ ] all boundaries
  - [ ] selected boundary markers
  - [ ] wall-tagged boundary sets
- [ ] Support query locations:
  - [ ] nodes
  - [ ] cells
  - [ ] quadrature points
- [ ] Store optional nearest boundary id and nearest point.
- [ ] Provide invalidation when mesh coordinates or moving mesh state changes.
- [ ] Add caching with explicit rebuild.
- [ ] Add tests for:
  - [ ] simple 1D/2D/3D meshes
  - [ ] selected marker filtering
  - [ ] moving mesh invalidation
  - [ ] deterministic nearest-boundary choice for ties

Implementation notes:

- Do not call this `RANS` or `yPlus`.
- RANS modules can consume this later to compute wall distance, wall functions,
  or nondimensional wall metrics.

### 8. Moving-Frame And Sliding-Interface Hardening

Goal: reuse existing moving-frame and sliding-interface FE systems while filling
only generic orchestration gaps.

Checklist:

- [ ] Audit existing moving-frame APIs for region-scoped frame lookup.
- [ ] Add generic region-to-frame binding metadata only if missing.
- [ ] Add validation that frame-dependent forms fail clearly when no frame is
      bound.
- [ ] Audit existing sliding-interface invalidation and transfer APIs for
      time-dependent map updates.
- [ ] Add tests for:
  - [ ] frame lookup by region
  - [ ] frame velocity at quadrature points
  - [ ] stale interface map invalidation
  - [ ] conservative transfer metadata through sliding maps
- [ ] Document the intended reuse path for MRF and rotor/stator Physics modules.

Implementation notes:

- Do not add duplicate sliding, mortar, or rotating-frame APIs.
- Do not add impeller-specific concepts to FE.

## Documentation Checklist

- [ ] Update FE developer docs to state the FE/Physics boundary for these
      additions.
- [ ] Add examples using neutral names:
  - [ ] bounded concentration-like scalar
  - [ ] generic two-field exchange
  - [ ] threshold volume of an arbitrary scalar
  - [ ] accumulated exposure of an arbitrary field
- [ ] Add a "Do not put in FE" note listing examples:
  - [ ] oxygen transfer laws
  - [ ] bubble breakup/coalescence laws
  - [ ] gas-liquid drag laws
  - [ ] RANS closures
  - [ ] MRF source terms
  - [ ] cell damage models
  - [ ] bioreactor-specific metrics

## Test Plan

- [ ] Add focused unit tests for each new FE type and helper.
- [ ] Add integration-style FE tests combining:
  - [ ] state group + admissibility
  - [ ] bounded update + invariant-domain summary
  - [ ] paired exchange + conservation summary
  - [ ] derived threshold reduction + percentile reduction
  - [ ] exposure accumulator + time stepping
- [ ] Add regression tests confirming existing Poisson, NS/VMS, level-set,
      moving-frame, and derived-result tests still pass.
- [ ] Add compile-time tests where possible to ensure APIs are optional and
      default-disabled.
- [ ] Run the FE unit test target and any existing solver smoke tests that cover
      form assembly, postprocessing, time stepping, and analysis summaries.

## Assumptions And Defaults

- Default behavior remains unchanged for all existing physics.
- All new infrastructure is opt-in.
- FE owns metadata, generic checks, generic reductions, generic accumulation,
  and generic geometry services.
- Physics owns equations, closures, source laws, limiters with physical meaning,
  and named bioreactor outputs.
- Existing sliding-interface and moving-frame infrastructure must be reused, not
  replaced.
- First implementation priority should be:
  1. State groups.
  2. Admissibility hooks.
  3. Paired exchange metadata.
  4. Threshold/percentile reductions.
  5. Exposure accumulator.
  6. Wall-distance service.
  7. Moving-frame/sliding-interface hardening.
