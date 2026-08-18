# Generalized AuxiliaryState Subsystem

The `AuxiliaryState` subsystem is FE-library infrastructure for managing
non-PDE auxiliary state variables.  It is **not** a boundary-condition
feature — boundary functionals, EP-like ionic models, metabolism models,
reduced models, and future coupled subsystems all use the same neutral API.

## Architecture Overview

```
AuxiliaryModelBuilder   ─── build() ───►  AuxiliaryStateModel (abstract)
       │                                         │
   use(model)                              evaluateResidual()
       │                                   evaluateJacobian() [optional]
       ▼                                         │
AuxiliaryDeployedInstance                         ▼
  .name(), .scope(),                    AuxiliaryDerivativeProvider
  .solveMode(), .bind(),                  Symbolic / FD / Analytic
  .stepper(), .initialize()                       │
       │                                          ▼
       ▼                                AuxiliaryStateStepper
AuxiliaryStateManager ◄─── FESystem      ForwardEuler, BackwardEuler,
  registerBlock()                         RK4, BDF2
  syncGhosts()
  commitAll()                           AuxiliaryInputRegistry
  packAll() / unpackAll()                 BoundaryReduction,
                                          FormulationCallback,
AuxiliaryOperatorRegistry                 SampledStateField, ...
  registerOperator()
  finalizeLayout()                      AuxiliaryCouplingGraph
  composeMixedLayout()                    AuxToAux, FieldToAux, ...
```

## Storage Scopes

The fixed-stride deployment path is complete for the scopes below.  Ragged
deployment remains a separate follow-on and is rejected by `FESystem` setup
when requested through `use(model)`.

| Scope | Entity identity | Ownership / ghosts | Typical use |
|-------|-----------------|--------------------|-------------|
| `Global` | Single synthetic entity `0` | Replicated state; monolithic rows use an explicit single owner or reduction/lowering path | System-wide scalar or small DAE |
| `Boundary` | Single named boundary collection | Replicated state; associated with a boundary marker | Lumped RCR model |
| `Node` | Stable mesh node ids, with restricted maps preserving owned prefix then ghosts | Owned/ghost layout; owner follows backend node/DOF ownership | Ionic gating variables |
| `Cell` | Stable cell ids after deployment-region expansion | Owned-cell authoritative; no auxiliary ghosts | Damage, plasticity |
| `QuadraturePoint` | Covered cell ids plus CSR-style `qpOffsets`; only cell-volume quadrature is in scope | Inherits owning cell; no auxiliary ghosts | Viscoplastic strain |
| `Region` | Topology-region ids from `TopologyAnalysisContext` | Deterministic owner from the globally lowest owned cell in the region | One reduced model per topology component |
| `Facet` | Boundary face/edge ids on selected boundary subsets | Stable boundary entity ordering | Per-face BC state |

Scope is orthogonal to solve mode: `Global` scope does not imply monolithic
participation, and `Monolithic` solve mode does not imply `Global` scope.

## Deployment Regions

Deployment region restricts which entities get auxiliary storage.
It is orthogonal to scope (scope = entity type, region = which subset).

| Region kind          | Example                                    |
|----------------------|--------------------------------------------|
| `WholeDomain`        | All entities (default)                     |
| `CellSet`            | Selected cells by marker                   |
| `BoundarySet`        | Selected boundary faces                    |
| `MaterialIdSet`      | Cells with specific material ID            |
| `TopologyRegion`     | Connected topology-region/component id     |
| `InterfaceSet`       | Interface entities                         |
| `FormulationDefined` | Custom entity selector                     |

Region expansion is scope-aware.  For example, a `MaterialIdSet` deployment
projects to cells for `Cell`, unique nodes for `Node`, covered cells plus
QP offsets for `QuadraturePoint`, and topology-region ids for `Region`.
`Global` and `Boundary` deployments treat the region as association metadata
instead of expanding it into multiple storage entities.

## Solve Modes

- **Partitioned**: Advanced independently via a local stepper between PDE
  steps.  Supports method selection, substepping, input-refresh policies,
  mixed ODE/algebraic rows, consistent initialization, local failure policy
  handling, and smooth event/reset hooks with per-entity context.
- **Monolithic**: Participates as first-class unknowns in the assembled
  residual/Jacobian system.  Uses auxiliary-specific unknown layouts
  composed into a mixed system layout (NOT FE field DOF maps).

Fixed-stride monolithic strategy is scope-specific:

- `Global` / `Boundary`: direct-only algebraic lowering, explicitly owned
  bordered/reduced rows, or distributed low-rank correction.  FSILS full rows
  are accepted only when the mixed layout can be represented as complete
  nodal-interleaved backend blocks.
- `Node`: owner-backed sparse/block-sparse path.  Native FSILS rows require
  complete nodal component partitions; otherwise the block is condensed,
  lowered, or routed through a backend that supports the requested layout.
- `Cell` / `QuadraturePoint`: independent local condensation.  These scopes do
  not append one auxiliary row per cell or QP to FSILS in the fixed-stride path.
  Models with cell-to-cell, QP-to-QP, or auxiliary-output cross-entity coupling
  must be represented with `AuxiliaryOperator` or another nonlocal formulation.
- `Region`: deterministic region row ownership with owner-routed
  bordered/reduced coupling for MPI.

Monolithic nonsmooth/complementarity hooks are intentionally rejected at setup
until a semismooth or active-set policy is implemented.

## Math-First Authoring (Preferred)

The math-first DSL in `AuxiliaryModelDSL.h` is the recommended way to define
auxiliary models.  Typed symbols participate directly in arithmetic, equation
insertion reads like the mathematical model, and deployment uses typed handles
instead of string concatenation.

### Defining a model

```cpp
#include "Auxiliary/AuxiliaryModelDSL.h"
using namespace svmp::FE::systems;

auto rcr = aux::model("rcr", [](ModelFacade& m) {
    auto Q = m.input("Q");
    auto X = m.state("X");
    auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");

    m << ddt(X) == (Q - (X - Pd) / Rd) / C;
    m << out("P_out") == X + Rp * Q;
});
```

Equation syntax:

- `m << ddt(X) == rhs`        — ODE row: dX/dt = rhs
- `m << alg(Z) == constraint`  — algebraic row: constraint = 0
- `m << out("name") == expr`   — named output

Grouped declarations for larger models:

```cpp
auto [O2, Glc, Lac] = m.inputs("O2", "Glc", "Lac");
auto [ATP, ADP, NADH] = m.states("ATP", "ADP", "NADH");
auto [k1, k2, k3] = m.params("k1", "k2", "k3");
```

Named intermediates for readability:

```cpp
auto v_rate = m.let("v_rate", k * S / (Km + S));
m.expose(v_rate, "observed_rate");  // also make it an output
```

### Registering FE-coupled inputs

Handle-returning APIs avoid string plumbing:

```cpp
auto Q = system.boundaryIntegral(inner(u, n), marker);
auto P_out = system.derivedInput("P_out", Pd + Rsum * Q.expr());
```

### Deploying a model

```cpp
auto rcr = system.deploy(
    use(rcr_model)
        .name("ns_rcr_10")
        .boundary(10)
        .partitioned("BackwardEuler")
        .params({{"Rp", 100.0}, {"C", 0.001}, {"Rd", 1000.0}, {"Pd", 0.0}})
        .bind(Q)                        // auto-bind by matching name
        .initialState({{"X", 0.0}})     // named, must cover all states
);

auto p_out = rcr.output("P_out");      // typed handle, no string concat
```

### Convenience binding helpers

Use the explicit helpers when you want the formulation to read like the
coupling contract instead of relying on implied name-matching:

```cpp
auto Q = system.boundaryIntegral(inner(u, n), marker);
auto P = system.derivedInput("P_out", Pd + R * Q.expr());

auto inst = system.deploy(
    use(model)
        .name("outlet_aux")
        .boundary(marker)
        .partitioned("BackwardEuler")
        .bindBoundaryReduction("Q", Q)  // FE boundary reduction + marker checks
        .bindByName(P));                // binds model input "P_out" to handle "P_out"
```

`bind(handle)` remains valid shorthand when the model input name already
matches the handle name. `bindBoundaryReduction(...)` is preferred for true
boundary-integral / boundary-average handles because it validates that the
handle is actually a boundary reduction before deployment.

### Auxiliary-driven FE constraints

Auxiliary outputs can drive time-varying FE constraints through the standard
constraint lifecycle:

```cpp
auto Q = system.boundaryIntegral(inner(u, n), marker);

auto bc_model = aux::model("dirichlet_driver", [](ModelFacade& m) {
    auto Q = m.input("Q");
    auto X = m.state("X");
    auto gain = m.param("gain", 1.0);

    m << ddt(X) == FormExpr::constant(0.0);
    m << out("bc") == X + gain * Q;
});

system.deploy(
    use(bc_model)
        .name("outlet_bc")
        .boundary(marker)
        .partitioned("ForwardEuler")
        .bindBoundaryReduction("Q", Q)
        .initialState({{"X", 0.25}})
        .drivesStrongDirichlet(u_field, marker, "bc"));

// During the solve / timestep loop:
system.prepareAuxiliaryForAssembly(state, false);
system.updateConstraints(state.time, state.dt);
```

Constraint structure is built once at setup. `updateConstraints()` refreshes
the inhomogeneity values from the current auxiliary state or output without
rebuilding the constrained-DOF set.

### Validation

The builder and deployment API enforce:

- Every declared state must have exactly one governing equation.
- Duplicate equations for the same state are rejected at insertion time.
- Equations targeting undeclared states are rejected with named diagnostics.
- `initialState({...})` requires all states to be listed (no silent zero-fill).
- Unknown parameter names are rejected by `validate()`.
- Self-referential `derivedInput()` expressions are rejected before registration.
- Forward-referenced dependencies in `derivedInput()` are resolved at finalization;
  unresolved names throw with diagnostics naming both inputs.

### Optional parameters and inputs

Parameters with default values are optional at deployment time:

```cpp
auto k = m.param("k", 1.0);        // optional, defaults to 1.0
auto O2 = m.optionalInput("O2", 0.0); // optional input
```

### Bounds and scaling metadata

```cpp
m.nonnegative("x");                 // lower_bound = 0
m.bounded("x", 0.0, 1.0);          // explicit bounds
m.scale("x", 1e-3);                // solver scaling hint
m.initialGuess("z", 0.5);          // algebraic initial guess
```

### Symbol grouping

```cpp
auto mito = m.group("mito");
auto ATP = mito.state("ATP");       // declares "mito.ATP"
```

### Conservation helpers

```cpp
auto ADP = m.state("ADP", Algebraic);
m.conservation(ADP, ATP + ADP - A_tot);  // lowers to alg(ADP) == ...
```

### Model summary / introspection

```cpp
AuxiliaryModelBuilder builder("my_model");
// ... add states, params, equations ...
std::cout << builder.summary();
```

### Many-input / many-output example

```cpp
auto model = aux::model("metabolism", [](auto& m) {
    auto [O2, Glc, Lac] = m.inputs("O2", "Glc", "Lac");
    auto states = m.stateVec({"ATP", "ADP", "NADH", "NAD", "PYR", "LAC_i", "G6P", "F6P"});
    auto params = m.paramVec({"k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"});

    m << ddt(states[0]) == -params[0] * states[0] + params[1] * states[1];
    // ... 7 more equations ...

    m << out("energy_charge") == states[0] / (states[0] + states[1]);
    m << out("redox_ratio") == states[2] / (states[2] + states[3]);
});
```

### Algebraic / DAE example

```cpp
auto model = aux::model("dae", [](auto& m) {
    auto x = m.state("x");
    auto z = m.state("z", AuxiliaryVariableKind::Algebraic);
    m.initialGuess("z", 0.5);

    m << ddt(x) == -x + z;
    m << alg(z) == x + z - FormExpr::constant(1.0);
});
// initialState({{"x", 1.0}}) auto-fills z from guess.
```

### Positional / advanced paths

The DSL is the recommended authoring surface.  Users leave the safe
high-level path when they:

- Use `AuxiliaryModelBuilder` directly instead of `aux::model()`
- Use `.initialize(vector)` instead of `.initialState({...})`
- Implement `AuxiliaryStateModel` manually instead of using the builder
- Use raw `AuxiliaryOperator` registrations for nonlocal couplings

These advanced paths are fully supported but bypass the DSL's validation,
named initialization, and unused-symbol diagnostics.

### Unused-symbol diagnostics

```cpp
builder.unusedSymbolPolicy("error");  // or "warn" (default), "silent"
```

## Other Authoring Surfaces

### Fluent builder (compatible, verbose)

```cpp
auto model = AuxiliaryModelBuilder("decay")
    .state("x").param("k")
    .ode("x", -modelParam("k") * modelState("x"))
    .output("y", modelState("x"))
    .build();
```

### Lower-level residual/Jacobian interface (advanced)

```cpp
class MyModel : public AuxiliaryStateModel {
    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& req) const override {
        req.residual[0] = ctx.xdot[0] + k * ctx.x[0];
    }
    bool hasAnalyticJacobian() const override { return true; }
    void evaluateJacobian(const AuxiliaryLocalContext& ctx,
                          AuxiliaryJacobianRequest& req) const override {
        req.dF_dx[0] = k;
        if (req.want_dF_dxdot) req.dF_dxdot[0] = 1.0;
    }
};
```

All three surfaces lower to the same `AuxiliaryStateModel` representation.

## Deployment via `use(model)`

The fluent deployment API works with all model types:

```cpp
auto instance = use(model)
    .name("outlet_rcr")
    .boundary(marker)                       // scope sugar
    .partitioned("BackwardEuler")           // solve mode + stepper
    .params({{"R_p", 100.0}, {"C", 0.001}}) // bulk parameter assignment
    .bind(Q_handle)                         // handle-based auto-bind
    .initialState({{"X", 0.0}});            // named initial values
```

### QuadraturePoint auto-layout

`QuadraturePoint` deployments now infer their per-cell QP layout from the
active cell-volume consumers of their outputs. The common path does not
require `.entityCount(...)` or `.qpOffsets(...)`:

```cpp
auto damage = system.deploy(
    use(damage_model)
        .name("damage_qp")
        .quadraturePoint()
        .monolithic()
        .initialize({0.0}));

auto residual = inner(damage.output("P_out") * sigma(u), grad(v)).dx();
```

Region-restricted QP deployments infer offsets over the covered cells only:

```cpp
AuxiliaryDeploymentRegion region;
region.explicit_entities = {7, 8, 9};

auto wall_damage = system.deploy(
    use(damage_model)
        .name("damage_wall")
        .quadraturePoint()
        .region(region)
        .monolithic()
        .initialize({0.0}));
```

Dormant alternative QP models can be deployed together and selected before
`setup()`:

```cpp
system.deploy(use(tt06).name("ep_tt06").quadraturePoint().variant("ep_model", "tt06"));
system.deploy(use(fhn).name("ep_fhn").quadraturePoint().variant("ep_model", "fhn"));

system.selectAuxiliaryVariant("ep_model", "tt06");
```

Selection is frozen after `setup()`. Unselected QP variants remain dormant and
are not materialized. `qpOffsets(...)` remains available as an advanced escape
hatch for explicit layouts; QP-scoped `.entityCount(...)` is no longer part of
the common path.

Advanced layout hints are also available when a QP deployment needs an explicit
quadrature source rather than consumer-driven inference:

```cpp
use(model).quadraturePoint().quadratureLike(u_field)
use(model).quadraturePoint().quadratureFromOperator("solid").alwaysActive()
```

Use these only for nonstandard workflows. `quadratureLike(...)` /
`quadratureFromOperator(...)` still validate against active cell consumers when
those consumers exist.

Migration guidance for older QP deployments:

```cpp
// Old common path
use(model).quadraturePoint().entityCount(n_qp_total).qpOffsets(offsets)

// New common path
use(model).quadraturePoint()

// Advanced explicit override remains valid when you intentionally need it
use(model).quadraturePoint().qpOffsets(offsets)
```

If an older QP deployment still passes `.entityCount(...)`, update it to rely
on inferred layout or keep only `.qpOffsets(...)` when an explicit advanced
layout is genuinely required.

## Derivative Policy

Precedence: analytic override > symbolic > finite differences.

- **Analytic** (if model provides `hasAnalyticJacobian()`): The model's
  own `evaluateJacobian()` is used directly.  This is the fastest path.
  Reported as `AuxiliaryDerivativeSource::Analytic`.
- **Symbolic** (default for expression-defined models): Differentiates
  residual `FormExpr` trees at setup time with respect to each
  `AuxiliaryStateRef(slot)` terminal.  Cached derivative expressions
  are evaluated at runtime via `PointEvaluator`.  Supports:
  - Arithmetic: `+`, `-`, `*`, `/`, negate.
  - Transcendental: `pow`, `sqrt`, `exp`, `log`.
  - Piecewise: `abs`, `sign`, `min`, `max`, `conditional`, comparisons.
  - Smooth approximations: `smoothAbs`, `smoothSign`, `smoothHeaviside`,
    `smoothMin`, `smoothMax`.
  - Scalar tensor ops: `inner`, `trace`, `det`, `inv`, `norm`.
  - History operators (constant w.r.t. current state).
  - Unsupported ops (FE/matrix/spectral) trigger FD fallback with a
    diagnostic in `artifact().fallback_reason`.
- **FiniteDifference**: Forward FD of the residual callback.  Used for
  custom models without analytic or symbolic derivatives.
- Implemented derivative targets: `dF/dx` (symbolic + FD), `dF/d(xdot)`
  (synthesized from row kind), `dF/d(inputs)` (symbolic + FD),
  `dF/d(fields)` (symbolic per-component, Node-scoped models with
  scalar/vector/tensor `DiscreteField`/`StateField` references, up to
  9 components; requires C0-continuous nodal Lagrange spaces; non-Node
  scopes and non-C0 spaces rejected at setup).
- Hessians are optional/demand-driven.

## Auxiliary Inputs

Inputs are externally supplied values consumed by auxiliary models.
Boundary functionals are one provider type, not the core abstraction.

| Producer                  | Description                          |
|---------------------------|--------------------------------------|
| `BoundaryReduction`       | Boundary-integrated scalar           |
| `FormulationCallback`     | Formulation-supplied callable        |
| `ParameterDerived`        | From system parameters               |
| `DirectUserData`          | User-set explicit value              |
| `AuxiliaryOutput`         | From another model's output          |
| `SampledStateField`       | Sampled FE field                     |
| `CoupledField`            | Symbolic FE dependency (monolithic)  |
| `DomainIntegral`          | Domain-integrated quantity           |
| `SampledBoundaryTrace`    | FE field on boundary                 |
| `CoupledBoundaryReduction`| Symbolic boundary reduction          |

## Auxiliary Outputs (preferred coupling surface)

Auxiliary outputs are the recommended way to couple auxiliary models into
FE formulations.  Use `AuxiliaryOutput("name")` in forms vocabulary:

```cpp
auto residual = inner(grad(u), grad(v)).dx()
              - AuxiliaryOutput("P_out") * inner(n, v).ds(marker);
```

Raw auxiliary state access (`AuxiliaryState("X")`) remains available
as an advanced path.

## Symbolic Vocabulary

| New terminal            | Replaces              | When to use              |
|-------------------------|-----------------------|--------------------------|
| `AuxiliaryInput("Q")`  | `boundaryIntegralValue("Q")` | Any aux input      |
| `AuxiliaryOutput("P")` | (new)                 | Model output coupling    |
| `AuxiliaryState("X")`  | same                  | Raw state (advanced)     |

## AuxiliaryOperator (nonlocal couplings)

For genuinely nonlocal or custom sparse couplings that cannot be
expressed as local per-entity residuals:

```cpp
auto op = AuxiliaryOperatorBuilder("cross_coupling")
    .source("block_A").target("block_B")
    .topology(AuxiliaryCouplingTopology::Sparse)
    .residual(my_residual_fn)
    .jacobian(my_jacobian_fn)
    .build();
registry.registerOperator(op);
```

Local/per-entity monolithic models stay on `AuxiliaryModel` + `use()`.

## DAE Support

Mixed differential/algebraic systems are supported from phase 1:

```cpp
auto model = AuxiliaryModelBuilder("dae_system")
    .state("x", AuxiliaryVariableKind::Differential)
    .state("z", AuxiliaryVariableKind::Algebraic)
    .ode("x", -modelState("x") + modelState("z"))
    .algebraic("z", modelState("x") + modelState("z") - constant(1.0))
    .build();
```

Available steppers: ForwardEuler, BackwardEuler, RK4, BDF2.
Substepping is supported via `stepper.substep_count`.

## Lifecycle Integration

```
beginTimeStep()
  ├─ AuxiliaryStateManager::resetAllToCommitted()
  └─ AuxiliaryInputRegistry::invalidateAll()   ← marks all inputs dirty

prepareAuxiliaryForAssembly(state)         ← caches state for FE-coupled callbacks
  ├─ cacheSystemState(state)
  ├─ AuxiliaryInputRegistry::evaluate(...)  ← evaluates dirty inputs
  └─ output expression evaluation

advanceAuxiliaryState(state)               ← preferred: caches state then steps
  ├─ cacheSystemState(state)
  ├─ AuxiliaryInputRegistry::evaluate(...)  ← no-op for OncePerTimeStep (already clean)
  └─ per-entity stepper dispatch

[Newton iterations / assembly]

commitTimeStep()
  └─ AuxiliaryStateManager::commitAll(time)
```

### FE-coupled inputs and the SystemStateView overload

When auxiliary inputs depend on FE field state (e.g., boundary integrals
registered via `registerBoundaryIntegralInput()`), the callbacks
reconstruct a `SystemStateView` from cached spans/pointers that were
populated by `cacheSystemState()`.  Those caches borrow — they do not
copy — the caller's solution vectors.

**Always use `advanceAuxiliaryState(const SystemStateView&)` when
FE-coupled inputs are present.**  The `(time, dt)` overload does not
update the cache; it will silently use stale data unless
`prepareAuxiliaryForAssembly()` was called earlier in the same step
with a valid `SystemStateView` whose underlying data is still alive.

### Input refresh timing

`OncePerTimeStep` inputs are evaluated once per step and cached.
Whichever path runs first after `beginTimeStep()` — typically
`prepareAuxiliaryForAssembly()` — fixes their values for the remainder
of that step.  Subsequent calls within the same step are no-ops.

`EachNonlinearIteration` inputs are re-evaluated on every
`prepareAuxiliaryForAssembly(state, /*is_nonlinear_iteration=*/true)`
or `assembleMonolithicAuxiliary(t, dt, res, jac, /*is_nonlinear_iteration=*/true)`
call, so they track the current iterate within Newton loops.

`OnceAtSetup` inputs are evaluated only on first access and never
refreshed.  `Manual` inputs are only evaluated when explicitly
marked dirty via `markDirty(name)`.

## Migration from Legacy Coupled-Boundary API

| Legacy                          | New                                           |
|---------------------------------|-----------------------------------------------|
| `coupled.addAuxiliaryState(reg)`| `AuxiliaryModelBuilder` + `use(model)`        |
| `coupled.addBoundaryFunctional` | `AuxiliaryInputRegistry` + `BoundaryReduction`|
| `boundaryIntegralValue("Q")`   | `AuxiliaryInput("Q")`                         |
| `auxiliaryState("X")`          | `AuxiliaryOutput("X")` (preferred)            |
| `CoupledBCContext`             | `SystemStateView::auxiliary_state/inputs`      |

During the migration window, deprecated helpers such as
`applyCoupledNeumann(...)`, `applyCoupledRobin(...)`, `CoupledNaturalBC`,
and `CoupledRobinBC` remain functional. Compatible boundary reductions
declared through those helpers are mirrored into the modern
`AuxiliaryInputRegistry` / `FEQuantityRegistry` internally so diagnostics,
handle-based introspection, and distributed allreduce semantics stay aligned.

That mirroring is compatibility plumbing only. New code should still author
the coupling directly through `AuxiliaryModelBuilder`, `boundaryIntegral(...)`,
`bindBoundaryReduction(...)`, and `AuxiliaryOutput(...)`.

## Checkpoint / Restart

```cpp
auto data = system.checkpointAuxiliaryState();
// ... serialize data ...
system.restoreAuxiliaryState(data);
```

Restart schemas include scope-specific entity metadata for restricted
deployments.  Fixed-stride `Node`, `Cell`, `QuadraturePoint`, and `Region`
blocks validate stable original entity ids during remap.  QP blocks also
store covered cell ids and `qpOffsets`; Region blocks store topology-region
membership metadata.  A restart whose entity map no longer matches the
materialized deployment is rejected instead of silently remapping to a
different entity ordering.

## Distributed Semantics

| Scope | Ownership rule |
|-------|----------------|
| `Global` | Replicated state; monolithic rows are single-owned, reduced, or lowered before backend insertion |
| `Boundary` | Single named-boundary instance; same replicated/single-owner rule as `Global` for monolithic rows |
| `Node` | Owned by mesh/backend node owner, ghosts appended after owned prefix |
| `Cell` | Owned by cell owner, no auxiliary ghosts |
| `QuadraturePoint` | Inherits covered-cell ownership, no auxiliary ghosts |
| `Region` | Owned by deterministic lowest-owned-cell rule for each topology region |
| `Facet` | Owned by face/edge owner |

Ghost sync via `setGhostSyncHook()` + `syncGhosts()` for
`OwnedAndGhost` policy blocks.

### MPI notes

- Boundary reductions (`boundaryIntegral`, `boundaryAverage`) are global
  quantities. Their providers evaluate locally and then apply
  `MPI_Allreduce(SUM)`, so every rank sees the same reduced value.
- Domain and region quantity providers use owned-cell subsets and reduce
  scalar values across ranks when necessary.
- Auxiliary-driven strong Dirichlet values are refreshed through the normal
  constraint lifecycle; after `prepareAuxiliaryForAssembly(...)` and
  `updateConstraints(...)`, all ranks observe the same synchronized
  inhomogeneity for the affected constrained DOFs.
- Node-scoped `OwnedAndGhost` auxiliary blocks treat only the owned prefix as
  authoritative state. Ghost entries are communication-populated and are
  refreshed after runtime update points that invalidate them
  (advance/reset/rollback/restore plus explicit `syncGhosts()` calls).
- FSILS matrix and vector participation is owned-row authoritative.  Ordinary
  matrix/RHS assembly routes through `GlobalSystemView` and explicit owner maps;
  reverse accumulation is reserved for assembler paths that explicitly expose a
  raw overlap contribution buffer.

## Implementation Status

The fixed-stride scope-completion deliverable is implemented and tested:

- `Global`, `Boundary`, `Node`, `Cell`, `QuadraturePoint`, `Region`, and
  `Facet` have first-class scope vocabulary and indexing contracts.
- `Global`, `Node`, `Cell`, `QuadraturePoint`, and `Region` support
  fixed-stride smooth ODE and mixed ODE/algebraic DAE deployments through
  partitioned and scope-appropriate monolithic paths.
- Scope-aware deployment-region expansion is centralized in `FESystem` for
  `Node`, `Cell`, `QuadraturePoint`, `Facet`, and `Region`.
- Partitioned runtime honors consistent initialization, local solve failure
  policies, smooth event/reset hooks, rollback, commit, and restart.
- Monolithic runtime uses FE time-integrator stage coefficients for auxiliary
  `xdot` evaluation and supports smooth accepted-step event/reset lifecycle.
- FSILS-compatible monolithic paths use explicit row-owner metadata for
  distributed sparsity and numeric assembly; tests require zero off-owner writes
  and zero dropped entries.
- QP auto-layout is inferred from active cell-volume consumers, including
  dormant unused deployments, pre-setup variant selection, restricted covered
  cells, and explicit `qpOffsets()` parity.
- Region-local FE-backed quantities are available through `regionIntegral()`
  and `regionAverage()`.

### Architectural constraints

- **Entity-local inputs**: Entity-local callback inputs are fully
  supported via `registerEntityInput()`.  FE-field-sampling convenience
  helpers are available for supported FE quantity kinds.  Entity-local
  bindings are indexed by stable original entity ids after deployment-region
  projection.  Restricted deployments that cannot cover the requested stable
  entity ids fail at finalization with a diagnostic instead of falling back to
  local materialized indices.
- **Generic model input/param ordering**: For `BuiltAuxiliaryModel`, input
  and parameter order is defined by the model signature.  For custom
  `AuxiliaryStateModel` subclasses, `declaredInputNames()` with optional
  `"name:size"` suffix controls ordering; fallback is lexicographic
  `std::map` key order.
- **Layout-aware entity access**: Per-entity stepping and output
  evaluation use `gatherEntityWork`/`scatterEntityWork` which support
  FixedStride + ByEntityThenComponent and ByComponentThenEntity layouts.
  Deployment API exposes `.layoutMode()` and `.entityOrdering()` for
  FixedStride blocks.  Ragged deployment is rejected at finalization and is
  tracked by the separate ragged-layout follow-on plan.

### Runtime limitations and partial features

- **Ragged deployment**: `AuxiliaryStateManager::registerBlockRagged()` exists
  for direct manager-level storage, but `FESystem` deployment, runtime DAE
  lifecycle, restart/remap, and monolithic assembly are fixed-stride only.
- **Nonsmooth monolithic hooks**: Smooth event/reset hooks are supported.
  Nonsmooth/complementarity monolithic hooks are rejected until a semismooth or
  active-set policy is implemented.
- **Schedule modes**: `SingleRate`, `Subcycled`, and `Multirate` are
  fully supported.  `Multirate` uses `planSubsteps()` for interleaved
  cross-block time ordering via `advanceFromWork()` (does not reset
  from committed state between substeps).  Predictor/corrector and
  error control policies are defined but not yet consumed at runtime.
- **Monolithic assembly**: Two assembly paths are available:
  - `assembleMonolithicAuxiliary()`: standalone dense assembly for
    auxiliary-only solves.
  - `assembleMixedAuxiliaryIntoGlobal()`: injects auxiliary residual/
    Jacobian and `AuxiliaryOperator` contributions into the global
    `GlobalSystemView` sparse system during `assembleOperator()`.
    Auxiliary DOF indices are computed from `MixedSystemLayout`.
  Both paths support input/parameter resolution, entity-local inputs,
  active FE time-integrator `xdot` coefficients, `EachNonlinearIteration`
  input refresh, field-endpoint operators, and scoped FE-backed quantity
  gradients where the provider declares monolithic linearization support.
- **AuxiliaryInput/Output in FE forms**: Auto-resolved by `FormsInstaller`
  when input/output registries are populated before form installation.
  Use `AuxiliaryOutput(instance, name)` for multi-instance disambiguation.
  Bare `AuxiliaryOutput(name)` throws on ambiguity.
- **Custom model input sizes**: `declaredInputNames()` supports a
  "name:size" suffix convention (e.g., "velocity:3") for multi-component
  inputs.  FESystem parses this suffix during input vector building:
  the base name (before `:`) is used for binding lookup, and the size
  (after `:`) controls how many components are copied from the registry.
  Unbound inputs are zero-padded to the declared size.
- **Symbolic derivative targets**: `dF/dx` (symbolic), `dF/d(xdot)`
  (synthesized), `dF/d(inputs)` (symbolic), and `dF/d(fields)` (symbolic
  per-component, Node-scoped models with scalar/vector/tensor fields,
  C0-continuous nodal Lagrange spaces) are all implemented.  Non-Node
  scopes and non-C0 spaces are rejected at setup.  Per-component differentiation
  handles `component(u,i)`, `inner(u,u)`, and product-rule compositions.

## FE-Backed Quantity Handles

All FE-coupled auxiliary inputs return `AuxiliaryInputHandle`, which carries:

- **Registry name** — for binding and lookup
- **Quantity kind** — SampledField, BoundaryIntegral, BoundaryAverage, DomainIntegral, DomainAverage, RegionIntegral, RegionAverage, FEExpression, DerivedCallback
- **Shape metadata** — Scalar, Vector(dim), Tensor(dim)
- **Referenced FE fields** — for multi-field dependency tracking
- **Capability flags** — explicit evaluation, monolithic linearization

### Full API Family

```cpp
auto qs  = system.sampledField("u_s", "u", n_entities);
auto qb  = system.boundaryIntegral(expr, marker);
auto qab = system.boundaryAverage("Q_avg", expr, marker);
auto qd  = system.domainIntegral("M", expr);
auto qad = system.domainAverage("M_avg", expr);
auto qr  = system.regionIntegral("R", expr, region_marker);
auto qar = system.regionAverage("R_avg", expr, region_marker);
auto qe  = system.feExpression("source", expr);
auto qp  = system.derivedInput("P", algebraic_expr);
auto qns = system.boundaryNodalSum("Q_n", "u", marker);
```

### Explicit vs Monolithic Binding

Users bind the same handle API in both solve modes. The solve mode decides
whether FE-backed bindings are frozen inputs or exact-coupled chain-rule terms:

```cpp
// Partitioned: FE quantity is sampled/frozen numerically.
use(model).partitioned("BackwardEuler").bind("Q", Q_handle);

// Monolithic: exact chain-rule dF/du = dF/dI × dI/du is assembled.
use(model).monolithic().bind("Q", Q_handle);
```

Handle-backed monolithic bindings are validated:
- The deployment must use `.monolithic()` solve mode
- The handle must support monolithic linearization (`supportsMonolithicLinearization()`)

### FE Quantity Registry

`FESystem::feQuantityRegistry()` provides introspection:

```cpp
const auto& reg = system.feQuantityRegistry();
auto boundary_defs = reg.byKind(FEQuantityKind::BoundaryIntegral);
auto field_defs = reg.byField(u_field);
auto coupled_defs = reg.monolithicCapable();
```

### Multi-Field Support

Multi-field integrands are supported. When a boundary integral references
multiple FE fields, the primary (first) field provides the DOF layout and
quadrature context, while secondary fields are automatically bound through
`BoundaryReductionService::registerSecondaryField()`:

```cpp
auto flux = system.boundaryIntegral(
    p_disc * inner(u_disc, FormExpr::normal()), marker);
// Both u and p fields are bound during evaluation.
```

## Region Semantics and Measure

Region integrals (`regionIntegral`, `regionAverage`) operate over **cells**
filtered by a region/material marker (domain ID). The region marker
corresponds to the cell domain ID assigned during mesh generation.

| Quantity | Definition | Measure |
|----------|-----------|---------|
| `regionIntegral("R", f, marker)` | `∫_{cells with domain_id=marker} f dx` | Volume of filtered cells |
| `regionAverage("R_avg", f, marker)` | `∫ f dx / ∫ 1 dx` over filtered cells | Same |
| `domainIntegral("M", f)` | `∫_Ω f dx` over all cells | Total domain volume |
| `domainAverage("M_avg", f)` | `∫ f dx / ∫ 1 dx` over all cells | Same |
| `boundaryIntegral(f, marker)` | `∫_{faces with marker} f ds` | Boundary area/length |
| `boundaryAverage("Q_avg", f, marker)` | `∫ f ds / ∫ 1 ds` | Same |

Domain and region integrals use cell-based quadrature via the
`FunctionalAssembler::assembleScalar()` path (with `FunctionalFormKernel::Domain::Cell`).
Boundary integrals use face-based quadrature via `assembleBoundaryScalar()`.

## True Boundary-Integral Inputs

The `BoundaryReductionService` provides first-class, physics-agnostic
support for registering and evaluating true quadrature-weighted boundary
integrals as auxiliary inputs.

### Boundary Integral vs. Boundary Nodal Sum

| Feature                | True Boundary Integral           | Boundary Nodal Sum             |
|------------------------|----------------------------------|--------------------------------|
| Quadrature-weighted    | Yes                              | No (DOF values only)           |
| Accuracy               | Exact for polynomial integrands  | Approximate (no weights)       |
| Registration           | `registerBoundaryIntegralInput`  | `registerBoundaryNodalSumInput`|
| Reduction modes        | Sum, Average (extensible)        | Sum only                       |
| MPI support            | Built-in allreduce               | Local only                     |
| Use case               | Flow rates, boundary fluxes      | Quick diagnostic sums          |

### Registration API

```cpp
// Preferred: handle-returning API
auto Q = system.boundaryIntegral(inner(u, n), marker);

// Legacy: void API (still supported)
system.registerBoundaryIntegralInput("Q_outlet", inner(u_disc, n), marker);
```

### Derived Inputs

Simple algebraic expressions over other inputs can be registered as derived
inputs with automatic dependency discovery:

```cpp
auto P_out = system.derivedInput("P_out", Pd + Rsum * Q.expr());
// Dependencies on Q are wired automatically.
```

### Reduction Contract

- **Domain**: Integrand evaluated over all boundary faces with matching marker.
- **Marker filtering**: Only faces with the registered boundary marker.
- **Reduction**: `Sum` (raw integral), `Average` (divided by boundary measure).
- **MPI**: `MPI_Allreduce(SUM)` when MPI is initialized.
- **Time-invariance**: Determined by integrand content (no time/solution → invariant).
- **Empty boundary**: Returns 0.0 (no faces match the marker).

### Notes

- **Multi-field integrands**: Supported.  Secondary fields are automatically
  bound via `registerSecondaryField()` with correct `component_offset` and
  `dof_per_node` for the FunctionalAssembler's interleaved layout.
- **Geometry-only integrands**: Supported without registered FE fields via
  `GEOMETRY_FIELD_ID` + default P1 Lagrange space for quadrature.
- **Monolithic linearization**: Supported for field-dependent FE quantities.
  The mechanism varies by kind:
  - `SampledField`: identity at sampled DOFs (direct DOF map lookup)
  - `BoundaryIntegral`, `DomainIntegral`, `RegionIntegral`, `FEExpression`:
    symbolic gradient via `BoundaryFunctionalGradientKernel`
  - `BoundaryAverage`: symbolic gradient with service-level average reduction
  - `DomainAverage`, `RegionAverage`: gradient of `__integral` / measure
  - `DerivedCallback`: N/A (no FE field dependence)
  For multi-field integrands, gradients are computed per-field by iterating
  over all referenced fields.

### Recommended Migration Path (Navier-Stokes Example)

```cpp
// Old (legacy CoupledNaturalBC path):
auto Qsym = FormExpr::boundaryIntegral(u_disc, marker, "Q");
auto reg = auxiliaryODE("X", X0).requiresIntegral(Q).withRHS(rhs).build();
return CoupledNaturalBC(marker, flux, {reg});

// New (math-first DSL path):
auto Q = system.boundaryIntegral(inner(u, n), marker);

auto rcr = aux::model("rcr", [](ModelFacade& m) {
    auto Q = m.input("Q");
    auto X = m.state("X");
    auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");
    m << ddt(X) == (Q - (X - Pd) / Rd) / C;
    m << out("P_out") == X + Rp * Q;
});

auto inst = system.deploy(
    use(rcr).name("ns_rcr_10").boundary(10).partitioned("BackwardEuler")
        .params({{"Rp", Rp}, {"C", C}, {"Rd", Rd}, {"Pd", Pd}})
        .bind(Q).initialState({{"X", X0}}));

auto p_out = inst.output("P_out");
return NaturalBC(marker, -p_out * n - backflow);
```

## Average Quantities

`boundaryAverage()`, `domainAverage()`, and `regionAverage()` compute mean
values by dividing the integral by the region measure:

```cpp
auto u_avg = system.boundaryAverage("u_avg", u_disc, marker);
auto rho_avg = system.domainAverage("rho_avg", rho_disc);
auto T_region = system.regionAverage("T_region", T_disc, material_id);
```

Domain and region integrals compute `∫ expr dx` over cells, while averages
additionally divide by the measure (`∫ 1 dx`).  Dependency ordering between
the integral and measure inputs is handled automatically.

## Shape-Aware FE Quantities

FE-backed handles carry shape metadata (`Scalar`, `Vector(dim)`, `Tensor(dim)`):

```cpp
auto u = system.sampledField("u_s", "velocity", n_nodes);
assert(u.shape().kind == FEQuantityShapeKind::Vector);
assert(u.shape().components == 3);
```

Shape helpers extract components or compute contractions:

```cpp
auto uz = comp(u_handle, 2);           // z-component
auto speed_sq = dot(u_handle, u_handle); // |u|^2
auto tr_sig = trace(sigma_handle);      // trace of tensor
auto speed = norm(u_handle);            // ||u||
```

Shape mismatches are validated for monolithic handle-backed `bind(...)`
calls — the handle's component count must match the model input size.

## Monolithic Chain-Rule Coupling

For monolithic auxiliary models, exact chain-rule Jacobian coupling is
available for FE-backed inputs that support linearization.

### Supported dI/du paths

| Quantity Kind      | dI/du Implementation                        |
|--------------------|---------------------------------------------|
| SampledField       | Identity at sampled DOFs (proper DOF map)   |
| BoundaryIntegral   | Symbolic via `BoundaryFunctionalGradientKernel` |
| BoundaryAverage    | Symbolic gradient (service applies average reduction) |
| DomainIntegral     | Symbolic via `CellGradKernelAdapter`         |
| DomainAverage      | Symbolic gradient of `__integral` / measure  |
| RegionIntegral     | Symbolic via `CellGradKernelAdapter`         |
| RegionAverage      | Symbolic gradient of `__integral` / measure  |
| FEExpression       | Symbolic (routed through domain-functional)  |
| DerivedCallback    | N/A (no FE field dependence)                |

### Assembly path

The chain rule `dF/du = dF/dI × dI/du` is composed in
`assembleMixedAuxiliaryIntoGlobal()`.  For each monolithic auxiliary block
with coupled bindings:

1. `dF/dI` is evaluated symbolically (or via FD fallback)
2. `dI/du` is computed per-quantity-kind
3. The product is inserted into the field→auxiliary Jacobian block

### Partitioned vs Monolithic examples

```cpp
// Generic partitioned FE-expression example:
auto area = system.boundaryIntegral(FormExpr::constant(1.0), marker);
auto Q = system.boundaryIntegral(inner(u, n), marker);
auto P = system.derivedInput("P", Pd + R * Q.expr());

use(model).partitioned("BackwardEuler").bind(Q).bind("P", P);

// Generic monolithic exact-coupled example:
auto Q = system.boundaryIntegral(inner(u, n), marker);

use(model).monolithic().bind("Q", Q);
// dF/du = dF/dQ * dQ/du assembled automatically.
```

## Limitations and Future Work

- **Legacy CoupledBoundaryManager**: The legacy path remains functional but
  separate.  New monolithic coupling should use handle-backed `bind()` + the
  `FEQuantityRegistry` path.
- **Transpose Jacobian performance**: `dR_PDE/dx_aux` is computed by FD
  perturbation of auxiliary outputs + PDE residual re-assembly (one re-assembly
  per output).  For models with many outputs, a symbolic approach using
  form-level differentiation w.r.t. `AuxiliaryOutput` nodes would be faster.
