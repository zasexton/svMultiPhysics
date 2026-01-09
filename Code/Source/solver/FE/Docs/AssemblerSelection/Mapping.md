# FE Assembler Auto-Selection — Mapping

This document maps **FE/Forms** expression/IR characteristics and **FE/Systems** setup characteristics to recommended **FE/Assembly** strategies.

The implementation is intentionally **opt-in** initially:
- `systems::SetupOptions::assembler_name` defaults to `StandardAssembler` (backward-compatible).
- Set `assembler_name = "Auto"` to enable rule-based selection.
- DG incompatibility is treated as a hard error (no silent fallback).

## 1) FormExprType → Assembly Implications

Legend:
- **DG**: requires interior-face assembly support (`Assembler::supportsDG()`).
- **Solution**: requires current/previous solution binding.
- **Time**: requires time/time-step and/or time-integration context.
- **High-Order**: requires higher-order basis data (curl/hessian/etc).
- **Material/Constitutive**: requires parameter/user-data and optionally material state.
- **Geometry**: requires geometry/mapping data (points/J/J⁻¹/normals/measures).
- **None**: no special assembler selection implication beyond normal kernel evaluation.

| FormExprType | Implication |
|---|---|
| `TestFunction`, `TrialFunction` | None (drives test/trial DOF dimensions; rectangular possible) |
| `DiscreteField` | Solution/Field data (multi-field access) |
| `StateField` | Solution/Field data (multi-field access; commonly nonlinear coupling) |
| `Coefficient`, `Constant` | None |
| `Coordinate`, `ReferenceCoordinate` | Geometry |
| `Time`, `TimeStep` | Time |
| `Identity` | None |
| `Jacobian`, `JacobianInverse`, `JacobianDeterminant` | Geometry |
| `Normal` | Geometry (face normal); boundary/DG if used under face integral |
| `CellDiameter`, `CellVolume`, `FacetArea` | Geometry/Measures |
| `Gradient`, `Divergence` | Geometry + basis gradients |
| `Curl` | High-Order (basis curls) |
| `Hessian` | High-Order (basis Hessians) |
| `TimeDerivative` | Time (transient; requires history via `dt(·,k)` lowering) |
| `RestrictMinus`, `RestrictPlus` | DG (trace selection) |
| `Jump`, `Average` | DG (requires neighbor data and interior-face iteration) |
| `Negate`, `Add`, `Subtract`, `Multiply`, `Divide`, `InnerProduct`, `DoubleContraction`, `OuterProduct`, `CrossProduct`, `Power`, `Minimum`, `Maximum` | None |
| `Less`, `LessEqual`, `Greater`, `GreaterEqual`, `Equal`, `NotEqual` | None |
| `Conditional` | None |
| `AsVector`, `AsTensor` | None |
| `Component`, `IndexedAccess` | None |
| `Transpose`, `Trace`, `Determinant`, `Inverse`, `Cofactor`, `Deviator`, `SymmetricPart`, `SkewPart`, `Norm`, `Normalize`, `AbsoluteValue`, `Sign`, `Sqrt`, `Exp`, `Log` | None |
| `Constitutive`, `ConstitutiveOutput` | Material/Constitutive (may require parameters/user-data and possibly material state) |
| `CellIntegral` | Cell assembly loop required |
| `BoundaryIntegral` | Boundary-face loop required |
| `InteriorFaceIntegral` | DG (interior-face loop required) |

## 2) FormIR Metadata → Selection Inputs

`forms::FormIR` provides the compile-time summary used for selection/validation:
- `FormKind` (Linear/Bilinear/Residual)
- `RequiredData` bitmask (geometry/basis/solution/material-state/DG neighbor data)
- test/trial `SpaceSignature` (order/continuity/element)
- domain presence: `hasCellTerms()`, `hasBoundaryTerms()`, `hasInteriorFaceTerms()`
- transient metadata: `maxTimeDerivativeOrder()`, `isTransient()`

## 3) Characteristics → Recommended Assembly Strategy

The current implementation supports a **base assembler + composable decorators**:
- Base assemblers: `StandardAssembler`, `ParallelAssembler`, `WorkStreamAssembler` (experimental), `DeviceAssembler` (CPU fallback unless GPU available).
- Decorators: caching, scheduling, vectorization (composable and opt-in).

| Characteristic | Recommended Strategy |
|---|---|
| DG / interior-face terms present | Base must support DG (`StandardAssembler`, `ParallelAssembler`) |
| Multi-field / coupled operators | Base must support rectangular + multi-field solution access (Standard/Parallel ok); consider block backends separately |
| Residual (nonlinear) kernels | Base supports solution binding; pair with nonlinear solve driver (outside `Assembler`) |
| High polynomial order / matrix-free intent | Prefer matrix-free operator backend (Systems `OperatorBackends`) over matrix assembly |
| GPU available + large mesh | `DeviceAssembler` (when GPU backend enabled) |
| Cache-friendly repeated assembly | Caching decorator (matrix-only or linear kernels) |
| Large mesh / locality sensitive | Scheduling decorator (ordering strategy) |
| SIMD batching | Vectorization decorator (batched traversal; future kernel support) |

