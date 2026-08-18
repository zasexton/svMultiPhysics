# Advanced H(div) Usage Guide

This guide documents the implemented advanced `H(div)` FE infrastructure:

- boundary and interface trace conditions
- trace Nitsche conditions
- orientation-aware periodic and MPC trace helpers
- mortar and hybridized facet fields
- mixed-dimensional codimension-1 interface fields
- one-sided scalar-trace inequality laws

The goal is to describe the current public FE surface, not a roadmap.

## Canonical Public Names

The following names are the canonical public surface for the advanced trace and
interface infrastructure.

### Boundary trace conditions

- `NormalTraceEssentialBC`
- `TraceLoadBC`
- `TraceRobinBC`
- `TraceNitscheBC`
- `TraceInequalityBC`

### Interface trace conditions

- `InterfaceTraceLoadBC`
- `InterfaceTraceRobinBC`
- `InterfaceTraceJumpPenaltyBC`
- `InterfaceTraceNitscheBC`

### Periodic and MPC trace helpers

- `makeHDivTracePeriodicPairs()`
- `makeHDivTracePeriodicPairsTranslation()`
- `makeHDivTracePeriodicBC()`
- `makeHDivTracePeriodicBCTranslation()`
- `makeHDivTracePeriodicMPC()`
- `makeHDivTracePeriodicMPCTranslation()`

### Mortar and mixed-dimensional field registration

- `spaces::MortarSpace`
- `FESystem::addInterfaceField()`

These names are the ones new code and new documentation should use.

## Choosing The Right Primitive

Use these rules when selecting the FE primitive:

- Use `NormalTraceEssentialBC` when the normal trace is prescribed strongly.
- Use `TraceLoadBC` or `TraceRobinBC` when the trace law is weak and equality-based.
- Use `TraceNitscheBC` when the trace law is weakly imposed through consistency, adjoint, and penalty terms.
- Use `InterfaceTrace*` classes when the law lives on `.dI(marker)` rather than `.ds(marker)`.
- Use the periodic or MPC trace helpers when the relation is algebraic and orientation-sensitive.
- Use `TraceInequalityBC` when the law is one-sided and may switch between active and inactive states during the nonlinear solve.

`TraceInequalityBC` is not a replacement for ordinary weak or strong boundary
conditions. It is the first FE-side contract for unilateral scalar-trace laws.

## 1. Boundary Trace Conditions

For standard mixed `H(div)` formulations, use the BC manager and the trace BC
wrappers from `Forms/StandardBCs.h`.

```cpp
#include "FE/Forms/StandardBCs.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/BoundaryConditionManager.h"

auto q = StateField(q_field, *Vhdiv, "q");
auto w = TestField(q_field, *Vhdiv, "w");
auto residual = inner(k_inv * q, w).dx();

systems::BoundaryConditionManager bc_manager;
bc_manager.add(forms::bc::makeNormalTraceEssentialBC(/*marker=*/4, Constant(1.0)));
bc_manager.add(forms::bc::makeTraceRobinBC(/*marker=*/7,
                                           Constant(alpha),
                                           Constant(rhs)));
bc_manager.applyAll(system, residual, q, w, q_field);
```

This is the canonical path for strong normal-flux data and weak equality-based
trace laws on `H(div)` fields.

`H(div)` fields also support analytic matrix-valued `grad(q)` in Forms on
affine cells:

```cpp
auto q = StateField(q_field, *Vhdiv, "q");
auto w = TestField(q_field, *Vhdiv, "w");
auto residual = (inner(k_inv * q, w) + beta * inner(grad(q), grad(w))).dx();
```

The FE layer computes reference RT/BDM vector-basis Jacobians analytically and
maps them with the affine contravariant Piola derivative. Curved/non-affine
Piola vector-gradient terms are not approximated; those requests throw a clear
unsupported-case diagnostic until derivatives of `J`, `J^{-1}`, and `detJ` are
implemented.

Unless noted otherwise, the later snippets assume the same surrounding setup:
`system`, the registered field ids, and the current residual/test-state symbols
already exist.

## 2. Interface Trace Conditions

Interface trace conditions use the same manager surface, but with the
`InterfaceTrace*` family and an interface marker that is assembled on
`.dI(marker)`.

```cpp
systems::BoundaryConditionManager bc_manager;
bc_manager.add(forms::bc::makeInterfaceTraceLoadBC(/*interface_marker=*/17,
                                                   Constant(0.0)));
bc_manager.add(forms::bc::makeInterfaceTraceJumpPenaltyBC(/*interface_marker=*/17,
                                                          Constant(beta),
                                                          Constant(0.0)));
bc_manager.applyAll(system, residual, q, w, q_field);
```

Use this path for transmission, interface exchange, and weak trace continuity
without introducing a physics-specific wrapper.

## 3. Trace Nitsche Conditions

Trace Nitsche conditions are explicit objects in `Forms/NitscheBC.h`. They are
appropriate when the FE law should remain weak but should include consistency,
adjoint, and penalty terms in one object.

```cpp
#include "FE/Forms/NitscheBC.h"
#include "FE/Forms/Vocabulary.h"

const auto n = normal();
auto u = StateField(u_field, *V, "u");
auto v = TestField(u_field, *V, "v");

systems::BoundaryConditionManager bc_manager;
bc_manager.add(std::make_unique<forms::bc::TraceNitscheBC>(
    /*boundary_marker=*/9,
    Constant(0.0),
    inner(grad(u), n),
    inner(grad(v), n),
    Constant(1.0) / h(),
    forms::bc::ScalarTraceOperator::Identity));
bc_manager.applyAll(system, residual, u, v, u_field);
```

Use `InterfaceTraceNitscheBC` for the interface counterpart. For `H(div)`,
switch the trace operator to `ScalarTraceOperator::NormalComponent` and supply
the formulation-specific scalar consistency and adjoint flux expressions.

## 4. H(div) Periodic And MPC Trace Relations

The periodic and MPC helpers live in `Constraints/ConstraintTools.h`. They
operate on trace-ordered facet DOFs and account for outward-normal orientation.

```cpp
#include "FE/Constraints/ConstraintTools.h"

std::vector<constraints::TraceBoundaryEntity> slave_entities = collect_slave_trace_entities();
std::vector<constraints::TraceBoundaryEntity> master_entities = collect_master_trace_entities();

constraints::TracePeriodicConstraintOptions opts;
opts.anti_periodic = false;

auto periodic = constraints::makeHDivTracePeriodicBCTranslation(
    *Vhdiv,
    slave_entities,
    master_entities,
    /*translation=*/{1.0, 0.0, 0.0},
    opts);

system.addConstraint(std::make_unique<constraints::PeriodicBC>(std::move(periodic)));
```

Use the corresponding `makeHDivTracePeriodicMPC*()` helpers when the relation
should be represented as an MPC instead of a `PeriodicBC`.

The extraction of `TraceBoundaryEntity` objects is mesh- and application-side
topology work. The FE helper owns the orientation and sign logic once those
entities are supplied.

## 5. Mortar And Hybridized Facet Fields

Mortar and hybridized facet unknowns are represented by `spaces::MortarSpace`.
The current public path is an expert `FESystem` path using explicit interface
field registration and interface-face kernels.

```cpp
#include "FE/Spaces/MortarSpace.h"

constexpr int marker = 17;

auto volume_space = std::make_shared<spaces::L2Space>(ElementType::Quad4, 0);
auto mortar_space = std::make_shared<spaces::MortarSpace>(
    std::make_shared<spaces::L2Space>(ElementType::Line2, 0),
    marker);

systems::FESystem system(mesh);
system.setInterfaceMeshFromFaceSet(marker, "middle");

const auto q = system.addField({.name = "q", .space = volume_space});
const auto lambda = system.addField({.name = "lambda", .space = mortar_space});

system.addOperator("hybrid");
system.addInterfaceFaceKernel("hybrid", marker, lambda, q, my_lambda_q_kernel);
system.addInterfaceFaceKernel("hybrid", marker, q, lambda, my_q_lambda_kernel);
system.addInterfaceFaceKernel("hybrid", marker, lambda, my_facet_kernel);
system.setup();
```

The FE library now owns the facet DOF numbering, sparsity, and interface-face
assembly path. The application still chooses the actual mortar or hybridized
weak form through the registered kernels.

## 6. Mixed-Dimensional Interface Fields

True codimension-1 unknowns are registered with `FESystem::addInterfaceField()`
rather than by restricting a volume field through `TraceSpace`.

```cpp
constexpr int marker = 23;

auto interface_space = std::make_shared<spaces::H1Space>(ElementType::Line2, 1);

systems::FESystem system(mesh);
system.setInterfaceMeshFromFaceSet(marker, "middle_vertical");

const auto q = system.addField({.name = "q", .space = volume_space});
const auto lambda = system.addInterfaceField("lambda", interface_space, marker);

system.addOperator("mixeddim");
system.addInterfaceFaceKernel("mixeddim", marker, q, lambda, my_q_lambda_kernel);
system.addInterfaceFaceKernel("mixeddim", marker, lambda, q, my_lambda_q_kernel);
system.setup();
```

This path is appropriate for fractures, membranes, wells, or any codimension-1
field that must own its own DOFs, history, and algebraic blocks.

## 7. One-Sided Trace Laws

One-sided scalar-trace laws use `TraceInequalityBC`. The current first-pass
scope is boundary-first and semismooth by default, with optional smooth
regularization.

```cpp
forms::bc::TraceInequalityOptions opts;
opts.trace_operator = forms::bc::ScalarTraceOperator::NormalComponent;
opts.sense = forms::bc::TraceInequalitySense::LessEqual;

systems::BoundaryConditionManager bc_manager;
bc_manager.add(forms::bc::makeTraceInequalityBC(/*marker=*/11,
                                                Constant(0.0),
                                                Constant(gamma),
                                                opts));
bc_manager.applyAll(system, residual, q, w, q_field);
```

Use this when the physical law is one-sided, for example:

- outflow may occur but inflow should be penalized
- a seepage-like law activates only when the trace violates a bound
- a unilateral scalar-trace relation must participate in the nonlinear solve

Do not use `TraceInequalityBC` when the intended law is an ordinary equality.
For equality-based trace laws, prefer `NormalTraceEssentialBC`,
`TraceLoadBC`, `TraceRobinBC`, or `TraceNitscheBC`.

## 8. Current Scope Boundaries

The current public FE surface is intentionally limited in a few places:

- mortar examples currently use explicit kernel registration rather than a dedicated formulation installer
- mixed-dimensional examples currently use interface-face kernels rather than a higher-level coupling DSL
- the first one-sided nonlinear contract is `TraceInequalityBC`; there is not a separate multiplier-style complementarity API

These are deliberate FE-layer boundaries, not missing documentation.
