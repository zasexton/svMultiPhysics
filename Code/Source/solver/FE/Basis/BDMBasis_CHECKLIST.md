# BDMBasis Completeness Checklist

This checklist is a concrete implementation plan for extending `BDMBasis` beyond the previous **2D, order-1 only** implementation.

## Definition of done

- [x] `BDMBasis` supports **order `k >= 1`** on the targeted element families (see scope below).
- [x] `BDMBasis::size()` matches the chosen dimension formulas for every supported `(element_type, k)`.
- [x] `BDMBasis::evaluate_vector_values()` and `BDMBasis::evaluate_divergence()` work for all supported cases.
- [x] `BDMBasis::dof_associations()` reports correct entity type/id and consistent moment indexing.
- [x] Orientation correction in `HDivSpace`/`OrientationManager` is compatible with the chosen BDM DOF layout.
- [x] Unit tests cover DOF Kronecker, divergence, polynomial reproduction, and 3D size/divergence.
- [x] Deferred checklist entry `BDMBasis_HexDivergenceAndSize` is implemented and checked off in `Code/Source/solver/FE/Basis/UNIT_TEST_CHECKLIST.md`.

## Scope decisions (must be decided up front)

- [x] **Decide the BDM polynomial space on tensor-product cells**:
  - Option A (matches current `BDMBasis(ElementType::Quad*, 1)` behavior): `BDM_k(quad) := [Q_k]^2`, `BDM_k(hex) := [Q_k]^3`.
    we choose option A for this scoped decision.
  - Note: the old “quad BDM dimension = `2*(k+1)*(k+2)`” is the **RT tensor-product** dimension, not `[Q_k]^2`.
- [x] Decide **supported element types**:
  - Implemented: triangles/quads (2D) and tetrahedra/hexahedra (3D). Wedge/pyramid remain out of scope.
- [x] Decide **DOF moment basis** on edges/faces:
  - Recommended: moments against the existing `LagrangeBasis` on `Line2`/`Triangle3`/`Quad4` so orientation code can reuse `OrientationManager`’s node-permutation logic.

## Implementation checklist (code + file touchpoints)

### 1) Extend `BDMBasis` data model

- [x] Update `Code/Source/solver/FE/Basis/VectorBasis.h`:
  - [x] Add nodal-generation storage to `BDMBasis` (similar to `RaviartThomasBasis`):
    - [x] `bool nodal_generated_`
    - [x] modal polynomial representation (`ModalTerm`/`ModalPolynomial`) + `std::vector<ModalPolynomial> monomials_`
    - [x] `std::vector<Real> coeffs_`
  - [x] Ensure `BDMBasis` can represent both 2D and 3D bases (`dimension_` from `element_dimension()`).

### 2) Expand `BDMBasis` constructor: supported elements, orders, and dimension formulas

- [x] Update `Code/Source/solver/FE/Basis/VectorBasis.cpp` in `BDMBasis::BDMBasis(...)`:
  - [x] Accept all canonical variants for a shape using helpers (e.g., `is_triangle(type)` not `type == Triangle3`).
  - [x] Reject `order < 1` (BDM starts at 1 on standard definitions).
  - [x] Compute `size_` for each supported shape:
    - [x] Triangle: `size = (k+1)*(k+2)` for `BDM_k := [P_k]^2`.
    - [x] Tetra: `size = (k+1)*(k+2)*(k+3)/2` for `BDM_k := [P_k]^3`.
    - [x] Quad (if supported): `size = 2*(k+1)*(k+1)` for `BDM_k := [Q_k]^2`.
    - [x] Hex (if supported): `size = 3*(k+1)*(k+1)*(k+1)` for `BDM_k := [Q_k]^3`.

### 3) Implement nodal-generated BDM(k) basis construction (recommended approach)

- [x] Update `Code/Source/solver/FE/Basis/VectorBasis.cpp`:
  - [x] Build a modal spanning set `monomials_` matching the chosen space:
    - [x] Simplex: component-wise monomials of total degree `<= k` (`[P_k]^d`).
    - [x] Tensor-product: component-wise monomials with per-axis degree `<= k` (`[Q_k]^d`).
  - [x] Assemble the square DOF matrix `A` (rows = DOFs, cols = modal polynomials):
    - [x] **Boundary DOFs**:
      - [x] 2D: edge normal flux moments on each edge, weighted by `LagrangeBasis(Line2, k)`.
      - [x] 3D: face normal flux moments on each face, weighted by:
        - [x] `LagrangeBasis(Triangle3, k)` for tri faces (tet),
        - [x] `LagrangeBasis(Quad4, k)` for quad faces (hex).
      - [x] Use element topology from `elements::ReferenceElement` (`edge_nodes()`, `face_nodes()`) and coordinates from `basis::NodeOrdering`.
      - [x] Use existing quadrature rules (`QuadratureFactory`) with order `>= 2k+2`.
    - [x] **Interior DOFs** (needed for `k >= 2`):
      - [x] Simplex (triangle/tet): `∫_K v · w_i` against `NedelecBasis(order=k-2)`.
      - [x] Tensor-product (quad/hex): component-wise moments against `Q_{k-2,k}`/`Q_{k,k-2}` (2D) and `Q_{k-2,k,k}`/`Q_{k,k-2,k}`/`Q_{k,k,k-2}` (3D).
      - [x] Ensure the interior DOFs are compatible with `dof_associations()`’s indexing and do not depend on orientation.
      - [x] Add a small “invertibility smoke test” in unit tests for several `k` values (basis functions finite + Kronecker DOFs).
  - [x] Invert `A` into `coeffs_` and set `nodal_generated_ = true`.
  - [x] Remove the legacy hand-coded `k=1` formulas (single nodal path).

### 4) Implement evaluation routines for all supported cases

- [x] Update `Code/Source/solver/FE/Basis/VectorBasis.cpp`:
  - [x] `BDMBasis::evaluate_vector_values(...)`:
    - [x] If `nodal_generated_`, evaluate the modal basis then apply `coeffs_` (copy the RT pattern).
  - [x] `BDMBasis::evaluate_divergence(...)`:
    - [x] If `nodal_generated_`, compute divergence of each modal polynomial and apply `coeffs_` (copy the RT pattern).
    - [x] Add 3D divergence support for tetra/hex.

### 5) DOF metadata for assembly/orientation

- [x] Update `Code/Source/solver/FE/Basis/VectorBasis.cpp` in `BDMBasis::dof_associations()`:
  - [x] 2D: mark edge DOFs (`DofEntity::Edge`) with:
    - [x] `entity_id = edge index (ReferenceElement ordering)`
    - [x] `moment_index = local edge basis index (0..k)`
  - [x] 3D: mark face DOFs (`DofEntity::Face`) with:
    - [x] `entity_id = face index (ReferenceElement ordering)`
    - [x] `moment_index = local face basis index (0..face_dofs-1)` in the same ordering used by `OrientationManager`’s face permutations.
  - [x] Mark remaining DOFs as `DofEntity::Interior`.

### 6) Factory and element integration

- [x] Update `Code/Source/solver/FE/Basis/BasisFactory.cpp`:
  - [x] Keep `BasisType::BDM` as an explicit opt-in for `Continuity::H_div` (BDM now supports k>=1 for supported shapes).
  - [x] Keep current historical default: `Continuity::H_div` + `BasisType::Lagrange` only defaults to BDM for `dim==2 && k==1`.
- [x] Update `Code/Source/solver/FE/Elements/ElementFactory.cpp`:
  - [x] No `k==1` validation exists (no change needed).

### 7) Orientation correctness (required for 3D faces and higher-order edges)

- [x] Validate that the chosen DOF ordering matches existing orientation utilities:
  - [x] `Code/Source/solver/FE/Spaces/HDivSpace.h` / `Code/Source/solver/FE/Spaces/HDivSpace.cpp`
  - [x] `Code/Source/solver/FE/Spaces/OrientationManager.h` / `Code/Source/solver/FE/Spaces/OrientationManager.cpp`
- [x] If BDM DOFs are **Lagrange-node based** on edges/faces:
  - [x] Ensure `OrientationManager::orient_triangle_face_dofs` and `::orient_quad_face_dofs` produce correct permutations for your face DOF layout.
  - [x] Ensure 2D edge DOF orientation matches your edge DOF layout (add `orient_hdiv_edge_dofs`).
- [ ] If BDM DOFs are **modal (Legendre)**:
  - [ ] Implement parity-based orientation for edge/face modal moments (odd modes flip under reversal/reflection).

## Unit tests to add/extend (and where)

- [x] Update/add tests in `Code/Source/solver/FE/Tests/Unit/Basis/test_VectorBases.cpp`:
  - [x] `BDMBasis_DimensionFormulas`: extend to `k=1..N` and to any new shapes enabled.
  - [x] `BDMBasis_*Edge/Face*NormalMomentKronecker`: extend to higher order (k=2) using the same DOF basis as the implementation.
  - [x] `BDMBasis_PolynomialReproduction`: reproduce representative vector polynomials of degree `k` (2D/3D).
  - [x] `BDMBasis_*Divergence*`: verify divergence matches finite differences.
  - [x] Add an **orientation test**: permute/reverse an edge DOF vector using `OrientationManager`.
- [x] Implement and check off the deferred test:
  - [x] `BDMBasis_HexDivergenceAndSize` in `Code/Source/solver/FE/Basis/UNIT_TEST_CHECKLIST.md`.

## Suggested local validation commands

- [x] Build and run basis unit tests:
  - [x] `ninja -C build-fe-debug2 test_fe_basis`
  - [x] If you hit a Conda `GLIBCXX` mismatch, run tests with:
    - [x] `LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu ./build-fe-debug2/test_fe_basis`
