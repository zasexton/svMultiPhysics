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
- `systems::installFormulation()` with `.dI(marker)` terms for their couplings

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
bc_manager.add(forms::bc::makeNormalTraceEssentialBC(/*marker=*/4, FormExpr::constant(1.0)));
bc_manager.add(forms::bc::makeTraceRobinBC(/*marker=*/7,
                                           FormExpr::constant(alpha),
                                           FormExpr::constant(rhs)));
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
                                                   FormExpr::constant(0.0)));
bc_manager.add(forms::bc::makeInterfaceTraceJumpPenaltyBC(/*interface_marker=*/17,
                                                          FormExpr::constant(beta),
                                                          FormExpr::constant(0.0)));
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
    FormExpr::constant(0.0),
    inner(grad(u), n),
    inner(grad(v), n),
    FormExpr::constant(1.0) / h(),
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

Mortar and hybridized facet unknowns are represented by `spaces::MortarSpace`
(an interface-owned field on the InterfaceMesh of its marker; its interface
space must be discontinuous). Their couplings are forms on `.dI(marker)`
(decision D54, LM4), installed with the volume fields like any formulation:

```cpp
#include "FE/Forms/Vocabulary.h"
#include "FE/Spaces/MortarSpace.h"
#include "FE/Systems/FormsInstaller.h"

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

const auto qq = forms::StateField(q, *volume_space, "q");
const auto w = forms::TestField(q, *volume_space, "w");
const auto lam = forms::StateField(lambda, *mortar_space, "lambda");
const auto mu = forms::TestField(lambda, *mortar_space, "mu");
const auto residual = (qq * w).dx() +
                      (lam * w.minus()).dI(marker) + (lam * w.plus()).dI(marker) +
                      (mu * qq.minus()).dI(marker) + (mu * qq.plus()).dI(marker);
systems::installFormulation(system, "hybrid", {q, lambda}, residual);
system.setup();
```

The installer evaluates each product of a `.dI(marker)` term on one side of
each interface face: the side of the volume field it holds (`.minus()` or
`.plus()`), or the minus side when it holds only the interface field. The
interface field takes no restriction (it has one value), appears in `.dI`
terms only, and enters linearly with one unknown per product; the full rules
and refusals are in `Forms/VOCABULARY.md` (Boundary Conditions and
Constraints, the interface-owned field paragraph). The FE library owns the
facet DOF numbering, sparsity, and the interface-face assembly.

Hand-written interface-face kernels (`FESystem::addInterfaceFaceKernel`)
remain the expert path for a law the forms cannot state.

## 6. Mixed-Dimensional Interface Fields

True codimension-1 unknowns are registered with `FESystem::addInterfaceField()`
rather than by restricting a volume field through `TraceSpace`, and coupled to
the volume fields by forms on `.dI(marker)` in the same way. Two subdomains
`u` (minus side) and `w` (plus side) glued by a multiplier on their interface:

```cpp
constexpr int marker = 23;

auto interface_space = std::make_shared<spaces::H1Space>(ElementType::Line2, 1);

systems::FESystem system(mesh);
system.setInterfaceMeshFromFaceSet(marker, "middle_vertical");

const auto lambda = system.addInterfaceField("lambda", interface_space, marker);
const auto lam = forms::StateField(lambda, *interface_space, "lambda");
const auto mu = forms::TestField(lambda, *interface_space, "mu");

auto residual = volume_terms +
                (lam * v.minus()).dI(marker) - (lam * z.plus()).dI(marker) +
                (mu * (u.minus() - w.plus())).dI(marker);
systems::installFormulation(system, "glue", {u_field, w_field, lambda}, residual);
```

(`v` and `z` are the test functions of `u` and `w`.) This path is appropriate
for fractures, membranes, wells, or any codimension-1 field that must own its
own DOFs, history, and algebraic blocks. A multiplier restricted to a boundary
or interface but written as a volume field is the other route (D54 T1:
`plan.restrictSupport`, `bc::LagrangeDirichletBC`).

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
                                                FormExpr::constant(0.0),
                                                FormExpr::constant(gamma),
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

- mortar and interface-field couplings are forms on `.dI(marker)` only within the limits of D54 LM4: linear in the interface field, one unknown per product, one side per product, interpreted kernels (no JIT)
- there is no public bulk-to-manifold transfer vocabulary (`restrictTo`, `traceTo`, `extendFrom`; `Forms/VOCABULARY.md`, Not Scheduled)
- the first one-sided nonlinear contract is `TraceInequalityBC`; there is not a separate multiplier-style complementarity API

These are deliberate FE-layer boundaries, not missing documentation.
