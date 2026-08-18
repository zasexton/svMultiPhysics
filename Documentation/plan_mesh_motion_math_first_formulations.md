# Math-First Mesh-Motion Formulation Plan

This plan tracks the work to make mesh-motion physics formulations read as
weak forms written in Forms mathematical primitives. Field registration,
source-kind validation, mesh-motion role binding, and ALE setup live in
FE/Systems helpers; formulation modules keep the PDE algebra visible.

## Implementation Status

- [x] Mesh-motion residuals are written with primitives such as `grad`, `sym`,
  `trace`, `inner`, `dx`, and `ds`.
- [x] No Forms vocabulary was added for PDE-specific wrappers such as
  `harmonicMeshResidual(...)`, `pseudoElasticResidual(...)`, or
  `meshSmoothingOperator(...)`.
- [x] Equation-family names remain physics-module concepts:
  `HarmonicMeshMotionModule` and `PseudoElasticMeshMotionModule`.
- [x] Mesh-displacement field binding is centralized in FE/Systems.
- [x] Coupled ALE uses the same `mesh_displacement` unknown for the mesh
  equation, geometry sensitivity, and derived mesh velocity.
- [x] The larger application/restart/MPI qualification items remain tracked in
  `Documentation/plan_ale_mesh_motion_data_and_coupled_displacement.md` and
  `Documentation/plan_moving_mesh_infrastructure.md`.

## 1. Separate Binding From Math

- [x] Added `FE::systems::MeshDisplacementBinding`.
- [x] Added `FE::systems::MeshDisplacementBindingOptions`:

```cpp
struct MeshDisplacementBindingOptions {
    bool enabled{true};
    int dimension{0};
    std::string field_name{"mesh_displacement"};
    std::shared_ptr<const spaces::FunctionSpace> space{};
    bool auto_register_field{true};
    bool bind_as_mesh_displacement{true};
};
```

- [x] Added `resolveMeshDisplacementBinding(FESystem&, MeshDisplacementBindingOptions)`.
- [x] The helper finds an existing field by bound
  `MeshMotionFieldRole::Displacement` or by field name.
- [x] Auto-registration creates a true `Unknown`, not prescribed or derived
  data.
- [x] The helper requires `FieldSourceKind::Unknown`.
- [x] The helper validates vector-valued space, component count, and spatial
  dimension.
- [x] The helper binds `MeshMotionFieldRole::Displacement` when requested.
- [x] `HarmonicMeshMotionModule::registerOn()` and
  `PseudoElasticMeshMotionModule::registerOn()` use this helper instead of
  embedding field setup in the weak-form block.

## 2. Harmonic Mesh Motion

- [x] Local formulation symbols are math-oriented:
  `d_mesh`, `psi`, and `kappa`.
- [x] The residual is explicit in the module:

```cpp
const auto d_mesh = StateField(d_id, V, "d_mesh");
const auto psi = TestField(d_id, V, "psi");
const auto kappa = FE::forms::bc::toScalarExpr(effectiveKappa(options_), "mesh_motion_kappa");

auto residual = (kappa * inner(grad(d_mesh), grad(psi))).dx();
```

- [x] No `harmonicMeshResidual(...)` helper was introduced.
- [x] `HarmonicMeshMotionModule` remains the equation-family name.

## 3. Coefficients

- [x] Replaced scalar-only `stiffness` with mesh diffusivity `kappa`.
- [x] Constant `kappa` is supported.
- [x] Spatial and time scalar coefficients are supported through the existing
  Forms coefficient/callback machinery.
- [x] The residual spelling is preserved regardless of how `kappa` is supplied.
- [x] `stiffness` remains as a deprecated transition alias.
- [x] Inconsistent `stiffness` and `kappa` literals produce a diagnostic.

## 4. Boundary Terms

- [x] Strong Dirichlet mesh-displacement constraints are preserved.
- [x] Natural mesh-load boundary terms are written directly:

```cpp
residual = residual + (FormExpr::constant(-1.0) * inner(g_mesh, psi)).ds(marker);
```

- [x] Robin/interface spring terms are written directly:

```cpp
residual = residual + (alpha * inner(d_mesh - d_target, psi)).ds(marker);
```

- [x] Boundary-condition bookkeeping remains outside the Forms vocabulary.
- [x] No Forms vocabulary such as `movingWallCondition(...)` or
  `meshInterfaceSpring(...)` was added.

## 5. Pseudoelastic Mesh Motion

- [x] Added `PseudoElasticMeshMotionModule`.
- [x] Reused `resolveMeshDisplacementBinding(...)`.
- [x] The weak form is written directly:

```cpp
const auto eps_d = sym(grad(d_mesh));
const auto eps_psi = sym(grad(psi));
const auto I = FormExpr::identity(dim);

const auto sigma_mesh =
    FormExpr::constant(2.0) * mu_mesh * eps_d +
    lambda_mesh * trace(eps_d) * I;

auto residual = inner(sigma_mesh, eps_psi).dx();
```

- [x] Constant and coefficient `lambda_mesh` are supported.
- [x] Constant and coefficient `mu_mesh` are supported.
- [x] No `pseudoElasticResidual(...)` helper was introduced.
- [x] Pseudoelastic residual and tangent are checked against finite
  differences.

## 6. Geometry Frame Semantics

- [x] Documentation states that `grad(...)`, `.dx()`, and `.ds()` use the active
  FE geometry configuration.
- [x] The current modules solve on the active FE geometry. In coupled ALE, this
  is the trial current geometry when the FE geometric nonlinearity policy is
  enabled.
- [x] Reference-configuration mesh PDEs remain a future Forms primitive topic,
  for example `gradReference(u)` and `dxReference()`, not a high-level
  mesh-Laplacian wrapper.
- [x] No high-level reference-mesh wrapper was added.
- [x] The short guide documents the active/current/reference distinction.

## 7. Coupled ALE Integration

- [x] Mesh-motion modules and Navier-Stokes coupled ALE share one
  `mesh_displacement` unknown.
- [x] `resolveALEBinding(...)` accepts the displacement field already bound by
  the mesh-motion module.
- [x] Coupled ALE now diagnoses a mismatch between a bound displacement field
  and a separately named displacement field.
- [x] Coupled assembly metadata includes mesh-displacement residual rows.
- [x] Coupled assembly metadata includes mesh-displacement tangent blocks.
- [x] Fluid residual metadata includes tangent columns with respect to
  mesh displacement.
- [x] Derived mesh velocity remains
  `DerivedFromUnknown(mesh_displacement)` and is excluded from the solve block.

## 8. Tests

- [x] Unit tests cover `resolveMeshDisplacementBinding(...)`.
- [x] Harmonic mesh motion registers exactly one unknown field and binds it as
  `MeshMotionFieldRole::Displacement`.
- [x] Harmonic mesh motion does not register mesh velocity as a separate
  unknown.
- [x] Harmonic residual and tangent are checked against finite differences.
- [x] Spatially varying `kappa` is tested.
- [x] Natural boundary load assembly is tested.
- [x] Robin/interface spring assembly is tested.
- [x] Coupled ALE and harmonic mesh motion are tested to share the same
  displacement field.
- [x] Pseudoelastic residual and tangent are checked against finite
  differences.
- [x] A coupled assembly smoke test verifies shared displacement rows and fluid
  mesh-displacement columns. Backend coupled-solve qualification remains in the
  larger ALE qualification plan.

## 9. Documentation

- [x] Updated `Documentation/plan_ale_mesh_motion_data_and_coupled_displacement.md`
  with the mesh-displacement binding helper and math-first module convention.
- [x] Updated `Documentation/plan_moving_mesh_infrastructure.md` to reference
  this plan.
- [x] Added `Documentation/mesh_motion_math_first_formulation_guide.md`.
- [x] Documented the design rule: Forms expose mathematical primitives;
  physics modules express PDEs visibly.

## Verification

- [x] `cmake --build build-fe-check --target test_fe_systems -j2`
- [x] `./build-fe-check/test_fe_systems --gtest_filter='FESystem.MeshDisplacementBinding*:*ALEBinding*Displacement*'`
- [x] `./build-fe-check/test_fe_forms`
- [x] `./build-fe-check/test_fe_analysis`
- [x] `cmake --build build-physics-check --target test_physics -j2`
- [x] `./build-physics-check/test_physics --gtest_filter='MovingDomainPhysics.HarmonicMeshMotion*:MovingDomainPhysics.PseudoElasticMeshMotion*:MovingDomainPhysics.CoupledALEAndHarmonicMeshMotionShareDisplacementUnknown'`
- [x] `./build-physics-check/test_physics`

## Definition of Done

- [x] Harmonic mesh-motion formulation code reads primarily as the weak form.
- [x] Mesh-displacement setup and binding are centralized outside the weak-form
  block.
- [x] Coupled ALE and mesh-motion modules share one displacement unknown.
- [x] No new Forms vocabulary hides transport, mesh-smoothing, or PDE-specific
  residual structure.
- [x] Focused FE systems, FE assembly/forms, analysis, and physics tests pass.
