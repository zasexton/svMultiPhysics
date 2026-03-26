# FE/Forms → FE/Systems Integration

How `FE/Forms` weak-form expressions become assembled operators in `FE/Systems`.

## Quick Reference: Which API to Use

| Task | API | Header |
|------|-----|--------|
| Residual physics (single or multi-field) | `installFormulation()` | `FormsInstaller.h` |
| Mixed bilinear operator | `installMixedBilinear()` | `FormsInstaller.h` |
| Mixed linear operator | `installMixedLinear()` | `FormsInstaller.h` |
| Strong Dirichlet BCs | `installStrongDirichlet()` | `FormsInstaller.h` |
| Auto-detecting compilation | `FormCompiler::compile()` | `FormCompiler.h` |
| Explicit mixed compilation | `FormCompiler::compileMixed()` | `FormCompiler.h` |
| Pre-compiled IR installation | `installMixedFormIR()` | `FormsInstaller.h` |
| Auxiliary model definition | `AuxiliaryModelBuilder("name")` | `AuxiliaryModelBuilder.h` |
| Auxiliary model deployment | `use(model).scope(...).bind(...)` | `AuxiliaryBindings.h` |
| Auxiliary inputs | `AuxiliaryInputRegistry` | `AuxiliaryInputRegistry.h` |

### Expert/Manual Paths

| Task | API | Notes |
|------|-----|-------|
| Manual block decomposition | `BlockBilinearForm` / `BlockLinearForm` | For explicit per-block control |
| Complex-valued 2x2 lifting | `ComplexScalar` / `toRealBlock2x2` | Adapter for complex PDEs |
| Direct kernel registration | `FESystem::addCellKernel()` etc. | Framework internals / handwritten kernels |

## 1. The Compilation and Execution Pipeline

`FE/Forms` is a **front-end** that builds a runtime expression tree and compiles it into an `assembly::AssemblyKernel` that `FE/Systems` can register and assemble:

1. **User vocabulary / EDSL**: build a weak form with `forms::FormExpr` (and helpers from `Forms/Vocabulary.h`).
2. **Compile**: `forms::FormCompiler` lowers the expression to `forms::FormIR` (single-field) or `forms::MixedFormIR` (multi-field):
   - splits a sum into individual integral terms,
   - records the integration domain for each term (`dx`, `ds(marker)`, `dS`, `dI`),
   - infers `assembly::RequiredData` flags for each term and the combined form.
3. **Install**: `FormsInstaller` wraps compiled IR in kernels and registers them into `FESystem`:
   - `installFormulation()` handles residual physics (auto-detects single vs mixed)
   - `installMixedBilinear()` / `installMixedLinear()` handle operator installation
4. **Assemble**: `FE/Assembly` iterates mesh entities, prepares `AssemblyContext` according to `RequiredData`, calls the kernel, and inserts local contributions into backend-neutral global views.

## 2. Residual Physics (the main path)

A physics module needs two includes:

```cpp
#include "FE/Forms/Vocabulary.h"        // FormExpr helpers (stateField, testFunction, grad, ...)
#include "FE/Systems/FormsInstaller.h"  // installFormulation, installStrongDirichlet
```

The workflow is:

```cpp
// 1. Register fields
auto u_field = system.addField({.name = "u", .space = V});
system.addOperator("equations");

// 2. Build field-bound symbols (StateField + TestField)
auto u = StateField(u_field, *V, "u");
auto v = TestField(u_field, *V, "v");

// 3. Write the residual (single expression, any number of fields)
auto residual = (inner(grad(u), grad(v)) - f * v).dx();

// 4. Apply boundary conditions (one call: validate + weak terms + strong constraints)
bc_manager.applyAll(system, residual, u, v, u_field);

// 5. Install the formulation
installFormulation(system, "equations", {u_field}, residual);
```

For multi-field:
```cpp
auto u = StateField(u_field, *V, "u");
auto p = StateField(p_field, *Q, "p");
auto v = TestField(u_field, *V, "v");
auto q = TestField(p_field, *Q, "q");

auto residual = (inner(grad(u), grad(v)) - p * div(v)).dx()
              + (div(u) * q).dx();

installFormulation(system, "equations", {u_field, p_field}, residual);
```

`installFormulation()` automatically:
- Decomposes the expression by test function
- Detects which fields each test row depends on
- Installs Jacobian blocks and residual kernels
- Creates `CoupledBlockKernel` for fused assembly

## 3. Mixed Operators (bilinear / linear)

For assembling standalone operators (mass matrices, stiffness matrices) rather than residual physics:

```cpp
auto a = (inner(grad(u), grad(v))).dx() + (p * div(v)).dx() + (div(u) * q).dx();
installMixedBilinear(system, "stiffness", test_fields, trial_fields, a);

auto L = (f * v).dx() + (g * q).dx();
installMixedLinear(system, "load", test_fields, L);
```

## 4. Modifiers

These modify the residual or operator expression, not the installation path:

- **Boundary terms**: `(h * u * v).ds(5)` — boundary integrals on marker 5
- **DG terms**: `(jump(u) * avg(grad(v))).dS()` — interior-face integrals
- **Interface terms**: `(u * v).dI(marker)` — interface-face integrals
- **Transient terms**: `(dt(u, 1) * v).dx()` — time derivative (requires `TransientSystem`)
- **Constitutive models**: `inner(constitutive(model, grad(u)), grad(v)).dx()`

## 5. Expert Paths

### Manual block decomposition

For explicit per-block control (heterogeneous compilation options, pre-split
expressions from external tooling). Compile each block individually and
install via `installMixedFormIR()`:

```cpp
FormCompiler compiler;
MixedFormIR mir(2, 2);
mir.setKind(FormKind::Bilinear);
mir.setBlock(0, 0, compiler.compileBilinear(inner(grad(u), grad(v)).dx()));
mir.setBlock(0, 1, compiler.compileBilinear((p * div(v)).dx()));
// PP block left empty (zero block)

installMixedFormIR(system, "op", test_fields, trial_fields, mir);
```

### Complex-valued PDEs

Uses `ComplexBilinearForm` and `toRealBlock2x2()` from `Forms/Complex.h`.
The result is a `BlockBilinearForm` which must be compiled per-block and
installed via `installMixedFormIR()`:

```cpp
ComplexBilinearForm a;
a.re = inner(grad(u), grad(v)).dx();  // real part
a.im = (k * u * v).dx();             // imaginary part
auto blocks = toRealBlock2x2(a);      // → 2×2 BlockBilinearForm

// Compile each block and build a MixedFormIR
FormCompiler compiler;
auto compiled = compiler.compileBilinear(blocks);  // vector<vector<optional<FormIR>>>
MixedFormIR mir(2, 2);
mir.setKind(FormKind::Bilinear);
for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
        if (compiled[i][j]) mir.setBlock(i, j, std::move(*compiled[i][j]));

installMixedFormIR(system, "op", test_fields, trial_fields, mir);
```

### Direct kernel registration

For handwritten `AssemblyKernel` implementations:

```cpp
system.addCellKernel("op", field, my_custom_kernel);
```

## 6. How "Forms Vocabulary" Maps to AssemblyContext Data

- `TrialFunction` / `TestFunction` → basis values at quadrature points
- `grad(·)` → physical gradients
- `dt(·, k)` → k-th time derivative (symbolic; needs transient context)
- `x()` → physical coordinate
- `n` → face normal
- `h()` → cell diameter
- `.dx()` → cell loop
- `.ds(marker)` → boundary-face loop
- `.dS()` → interior-face loop (DG 4-block)
- `.dI(marker)` → interface-face loop

## 7. Migrating from the Old Pattern

If your formulation module uses the previous pattern:

```cpp
// Old pattern:
auto u = FormExpr::stateField(u_id, *space, "u");        // ← verbose
auto v = FormExpr::testFunction(*space, "v");             // ← no field binding
...
auto strong = bc_manager.getStrongConstraints(u_id);      // ← manual 3-step
bc_manager.apply(system, residual, u, v, u_id);
installStrongDirichlet(system, strong);
```

Migrate to:

```cpp
// New pattern:
auto u = StateField(u_id, *space, "u");                   // ← clean helper
auto v = TestField(u_id, *space, "v");                    // ← field-bound
...
bc_manager.applyAll(system, residual, u, v, u_id);        // ← one call
```

Key changes:
- `FormExpr::stateField(...)` → `StateField(...)` (Vocabulary.h helper)
- `FormExpr::testFunction(...)` → `TestField(field_id, ...)` (field-bound, required for same-space multi-field)
- 3-step BC pattern → `applyAll()` (validate + apply + installStrongDirichlet in one call)
- `#include "Forms/Forms.h"` → `#include "Forms/Vocabulary.h"` (smaller surface)

The old APIs continue to work — this is a style migration, not a breaking change.

## References

- Alnæs et al. — "Unified form language." *ACM Trans. Math. Softw.* (2014).
- Logg, Mardal, Wells (eds.) — *Automated Solution of Differential Equations by the FEM* (2012).
- Rathgeber et al. — "Firedrake." *ACM Trans. Math. Softw.* (2016).
