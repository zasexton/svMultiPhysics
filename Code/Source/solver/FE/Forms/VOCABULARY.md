# FE/Forms — Vocabulary Reference

This file documents what is **implemented and usable today**. For the
speculative expansion roadmap, see `VOCABULARY_ROADMAP.md`.

## Canonical Formulation Workflow

Most formulation code uses this workflow:

```cpp
#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/BoundaryConditionManager.h"

// 1. Register fields and operator
auto u_id = system.addField({.name = "u", .space = V});
system.addOperator("equations");

// 2. Build field-bound symbols
auto u = StateField(u_id, *V, "u");
auto v = TestField(u_id, *V, "v");

// 3. Write the residual
auto residual = (k * inner(grad(u), grad(v)) - f * v).dx();

// 4. Apply BCs and install
bc_manager.applyAll(system, residual, u, v, u_id);
installFormulation(system, "equations", {u_id}, residual);
```

For the full integration guide, see `Forms/SYSTEMS_INTEGRATION.md`.

---

## Implemented Vocabulary

### Terminals

| Terminal | Description |
|----------|-------------|
| `StateField(field_id, space, name)` | Unknown field bound to a registered FieldId (canonical) |
| `TestField(field_id, space, name)` | Test function bound to a registered FieldId (canonical) |
| `TrialFunction(space, name)` | Unbound trial function (expert path) |
| `TestFunction(space, name)` | Unbound test function (expert path) |
| `Coefficient(...)` | Known field (scalar, vector, matrix, rank-4 tensor) |
| `Constant(value)` | Runtime scalar parameter |
| `x`, `X` | Physical / reference coordinates |
| `J`, `Jinv`, `detJ` | Jacobian mapping quantities |
| `n` | Outward unit normal |
| `h`, `vol(K)`, `area(F)` | Cell diameter, cell volume, facet measure |
| `Constitutive(model, input)` | Constitutive model hook |

### Operators

| Operator | Description |
|----------|-------------|
| `grad(u)` | Gradient (scalar → vector, vector → matrix) |
| `div(u)`, `curl(u)` | Divergence, curl (vector fields) |
| `hessian(u)`, `laplacian(u)` | Second derivatives (scalar fields) |
| `dt(u, k)` | Time derivative (symbolic; requires transient context) |
| `inner(a, b)` | Inner product |
| `doubleContraction(A, B)` | `A:B` (rank-2 or rank-4 : rank-2) |
| `transpose`, `trace`, `det`, `inv` | Standard tensor operations |
| `cofactor`, `sym`, `skew`, `dev` | Tensor decomposition helpers |
| `norm`, `normalize`, `cross` | Vector operations |
| `normalComponent(u)` | Scalar normal trace `u·n` (for `H(div)` / trace BC authoring) |
| `component(i[,j])` | Component extraction |
| `A(i[,j])` + `einsum(expr)` | Einstein-style indexed access |
| `jump(u)`, `avg(u)` | DG jump and average |
| `u.minus()`, `u.plus()` | Explicit trace restrictions (DG) |

### Algebraic & Nonlinear

| Operator | Description |
|----------|-------------|
| `+`, `-`, `*`, `/` | Arithmetic |
| `pow`, `exp`, `log`, `sqrt` | Transcendental functions |
| `min`, `max`, `conditional` | Branching |
| `heaviside`, `clamp` | Convenience combinators (via `Vocabulary.h`) |
| `upwindValue` | DG upwind selection (via `Vocabulary.h`) |
| `interiorPenaltyCoefficient` | IP-DG penalty (via `Vocabulary.h`) |

### Measures

| Measure | Description |
|---------|-------------|
| `.dx()` | Cell (domain) integral |
| `.ds(marker)` | Boundary integral on tagged surface |
| `.dS()` | Interior facet integral (DG) |

### Installation API (canonical)

| Function | Description |
|----------|-------------|
| `installFormulation(system, op, fields, residual)` | Residual physics (auto-detects single/multi-field) |
| `installMixedBilinear(system, op, test, trial, form)` | Mixed bilinear operator |
| `installMixedLinear(system, op, test, form)` | Mixed linear operator |
| `installStrongDirichlet(system, bcs)` | Strong Dirichlet constraints |
| `BoundaryConditionManager::applyAll(...)` | One-call BC workflow |
| `FormCompiler::compile()` / `compileMixed()` | Compilation without installation |

### Boundary Condition Helpers

`Forms/StandardBCs.h` provides physics-agnostic BC wrappers for both pointwise
and trace-essential data:

- `EssentialBC` for pointwise value data
- `NormalTraceEssentialBC` for strong `H(div)` normal-trace data
- `TraceLoadBC` for weak scalar trace loads such as `-∫ g tau(v) ds`
- `TraceRobinBC` for weak scalar trace Robin terms such as `∫ alpha tau(u) tau(v) ds - ∫ rhs tau(v) ds`
- `TraceInequalityBC` for one-sided scalar trace laws such as `tau(u) <= g`
- `InterfaceTraceLoadBC` for weak interface trace loads on `.dI(marker)`
- `InterfaceTraceRobinBC` for weak interface exchange or Robin laws
- `InterfaceTraceJumpPenaltyBC` for interface trace jump penalization
- `TraceNitscheBC` for weak boundary imposition of scalar trace relations
- `InterfaceTraceNitscheBC` for weak interface imposition of scalar trace relations

For `H(div)`, the scalar trace operator is typically `tau(w) = normalComponent(w)`.

These are the canonical public class names for the trace-condition surface.
Boundary conditions and interface conditions intentionally use parallel names.

Use the ordinary strong or weak trace helpers when the intended law is an
equality such as `tau(u) = g`, Robin exchange, or weak continuity. Use
`TraceInequalityBC` only for one-sided laws where the trace may switch between
active and inactive behavior during the nonlinear solve.

### Constraint and Interface Helpers

The advanced `H(div)` infrastructure also exposes the following canonical
public helpers:

| Helper | Description |
|--------|-------------|
| `makeHDivTracePeriodicPairs*()` | Build orientation-aware periodic trace pairings |
| `makeHDivTracePeriodicBC*()` | Build an algebraic periodic trace constraint |
| `makeHDivTracePeriodicMPC*()` | Build an MPC trace relation with the same orientation rules |
| `spaces::MortarSpace` | Marker-scoped mortar or hybridized facet field space |
| `FESystem::addInterfaceField()` | Register a true codimension-1 interface field |

See `Docs/HDIV_ADVANCED_USAGE_GUIDE.md` for short usage examples covering the
trace, periodic/MPC, mortar, mixed-dimensional, and inequality paths.

### AD-backed Jacobians

Residual form → consistent Jacobian via forward-mode dual numbers
(`NonlinearFormKernel`), with coverage for `exp/log/sqrt/pow/div`.

### Auxiliary State Vocabulary

| Helper | Description |
|--------|-------------|
| `AuxiliaryInput("name")` | Reference auxiliary input by name (neutral, preferred) |
| `AuxiliaryOutput("name")` | Reference auxiliary model output (preferred coupling surface) |
| `AuxiliaryState("name")` | Raw auxiliary state access (advanced path) |
| `AuxiliaryInputSlot(n)` | Slot-indexed auxiliary input (JIT, post-resolution) |
| `AuxiliaryOutputSlot(n)` | Slot-indexed auxiliary output (JIT, post-resolution) |

**Deprecated** (use neutral alternatives above):
| Legacy | Replacement |
|--------|-------------|
| `boundaryIntegralValue("Q")` | `AuxiliaryInput("Q")` |
| `boundaryIntegralRef(slot)` | `AuxiliaryInputSlot(slot)` |

See `Docs/AuxiliaryState/README.md` for the full auxiliary state guide.

### Expert/Manual Helpers

These are available for advanced workflows but not needed for typical physics:

| Helper | Description |
|--------|-------------|
| `BlockBilinearForm` / `BlockLinearForm` | Manual block decomposition (`BlockForm.h`) |
| `ComplexScalar`, `toRealBlock2x2` | Complex-valued PDEs (`Complex.h`) |
| `installMixedFormIR()` | Pre-compiled IR installation |

## References

- Alnæs et al. — "Unified form language." *ACM TOMS* (2014). DOI: 10.1145/2566630.
- Logg, Mardal, Wells (eds.) — *Automated Solution of Differential Equations by the Finite Element Method* (2012).
- Rathgeber et al. — “Firedrake.” *ACM TOMS* (2016). DOI: 10.1145/2998441.
