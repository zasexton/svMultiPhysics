# Mesh-Motion Math-First Formulation Guide

This guide shows the intended style for mesh-motion physics modules in the new
OOP solver. Forms should expose mathematical primitives. Physics modules should
write the weak form visibly and use FE/Systems helpers for binding fields into
the solve state.

## Design Rule

Use Forms primitives for math:

- `StateField`, `TestField`
- `grad`, `sym`, `trace`, `inner`
- `dx`, `ds`
- coefficients through `FormExpr::constant(...)` or coefficient terminals

Do not add PDE-specific Forms wrappers such as `harmonicMeshResidual(...)`,
`pseudoElasticResidual(...)`, `movingWallCondition(...)`, or
`meshInterfaceSpring(...)`.

## Mesh-Displacement Binding

Mesh-motion modules should resolve their solved displacement unknown before the
weak-form block:

```cpp
const auto binding = FE::systems::resolveMeshDisplacementBinding(
    system,
    FE::systems::MeshDisplacementBindingOptions{
        true,
        dim,
        options_.field_name,
        displacement_space_,
        options_.auto_register_field,
        options_.bind_as_mesh_displacement});
```

The helper owns field lookup, auto-registration, `FieldSourceKind::Unknown`
validation, vector dimension checks, and `MeshMotionFieldRole::Displacement`
binding.

## Harmonic Mesh Motion

The harmonic mesh-motion residual should remain readable as the PDE:

```cpp
using namespace svmp::FE::forms;

const auto d_id = binding.displacement_field;
const auto& V = *binding.space;

const auto d_mesh = StateField(d_id, V, "d_mesh");
const auto psi = TestField(d_id, V, "psi");
const auto kappa = FE::forms::bc::toScalarExpr(options_.kappa, "mesh_motion_kappa");

auto residual = (kappa * inner(grad(d_mesh), grad(psi))).dx();
```

`kappa` may be a constant, spatial scalar coefficient, time scalar coefficient,
or already-built scalar `FormExpr`.

## Pseudoelastic Mesh Motion

Pseudoelastic smoothing should make strain and stress explicit:

```cpp
using namespace svmp::FE::forms;

const auto d_mesh = StateField(d_id, V, "d_mesh");
const auto psi = TestField(d_id, V, "psi");

const auto lambda_mesh =
    FE::forms::bc::toScalarExpr(options_.lambda_mesh, "mesh_motion_lambda");
const auto mu_mesh =
    FE::forms::bc::toScalarExpr(options_.mu_mesh, "mesh_motion_mu");

const auto eps_d = sym(grad(d_mesh));
const auto eps_psi = sym(grad(psi));
const auto I = FormExpr::identity(dim);

const auto sigma_mesh =
    FormExpr::constant(2.0) * mu_mesh * eps_d +
    lambda_mesh * trace(eps_d) * I;

auto residual = inner(sigma_mesh, eps_psi).dx();
```

Keep `sigma_mesh` in the module so readers can see the constitutive choice
without knowing a solver-specific helper name.

## Boundary Terms

Write weak boundary terms directly next to the residual.

Natural mesh load:

```cpp
const auto g_mesh = FormExpr::asVector({g0, g1, g2});
residual = residual + (FormExpr::constant(-1.0) * inner(g_mesh, psi)).ds(marker);
```

Robin or interface spring:

```cpp
const auto d_target = FormExpr::asVector({d0, d1, d2});
residual = residual + (alpha * inner(d_mesh - d_target, psi)).ds(marker);
```

Strong Dirichlet data may still be lowered through FE/Systems boundary
constraint infrastructure; it should not obscure the weak residual algebra.

## Geometry Frame Semantics

`grad(...)`, `.dx()`, and `.ds()` use the active FE geometry configuration.

For ordinary static or prescribed-motion assembly, this is the FE system's
configured current/reference geometry. For coupled ALE, the FE geometric
nonlinearity policy updates current coordinates from the trial
`mesh_displacement` state before assembly, so the same primitives assemble on
the trial current geometry.

Reference-configuration mesh PDEs should be added later as true mathematical
primitives, such as `gradReference(u)` and `dxReference()`, rather than as
high-level mesh-smoothing wrappers.

## Coupled ALE Usage

The mesh-motion module owns the solved displacement rows:

```cpp
mesh_motion::HarmonicMeshMotionModule mesh_module(d_space, mesh_options);
mesh_module.registerOn(system);
```

Navier-Stokes coupled ALE then reuses the bound displacement and derives mesh
velocity from it:

```cpp
ns_options.enable_ale = true;
ns_options.mesh_velocity_source =
    navier_stokes::ALEMeshVelocitySource::CoupledDisplacement;
ns_options.mesh_displacement_field_name = "mesh_displacement";
ns_options.mesh_velocity_field_name = "mesh_velocity";
```

The resulting `mesh_velocity` field is `DerivedFromUnknown(mesh_displacement)`;
it is not an independent solve block. The fluid residual receives tangent
columns for `mesh_displacement`, and the mesh-motion residual contributes the
mesh-displacement rows.
