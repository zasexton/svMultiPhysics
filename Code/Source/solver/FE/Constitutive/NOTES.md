# Constitutive Infrastructure Notes

This folder provides physics-agnostic interfaces for defining “laws” that can be used in FE assembly:
- Pointwise (per quadrature point) models through `forms::ConstitutiveModel`
- Nonlocal/global laws through `constitutive::GlobalLaw` emitting `systems::GlobalKernel` adapters

## Current limitations / non-goals

- `forms::Value` is intentionally small and currently supports only:
  - `Scalar`, `Vector`, `Matrix` (plus symmetric/skew tags), and fixed-size `Tensor4`
  - No general arbitrary-rank tensor container, no complex-valued numbers, no automatic unit/quantity types.
- `det(...)`, `inv(...)`, and `cofactor(...)` are implemented only for square matrices up to `3×3`.
- `cross(a,b)` is implemented as a 3-component cross product (not generalized to higher component counts).
- Vector-valued FE spaces are still conceptually “1–3 components”; higher-component unknowns should be modeled as
  product/mixed fields rather than a single “wide vector” space.
- Truly global/implicit subproblems “inside a constitutive update” are not automatically expressed as pointwise models;
  they should be implemented via `constitutive::GlobalLaw` + Systems/operator infrastructure (and may require adding new
  Systems hooks over time).

## Practical guidance

- Prefer pointwise models when the law depends only on the local evaluation context (point, quadrature weights, local
  state, parameters, and optional nonlocal queries supplied through the evaluation context).
- Use `constitutive::GlobalLaw` when evaluation requires mesh-wide operators, neighbor/face coupling, search-based loops,
  or additional assembly passes.

