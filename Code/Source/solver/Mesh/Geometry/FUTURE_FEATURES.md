# Mesh/Geometry: Production-Readiness Gaps & Follow-Ups

This file tracks known risks, missing coverage, and follow-up work for the Mesh `Geometry/` subfolder.
Items are marked **Resolved** once implemented with unit tests. Remaining items include concrete file-level
change outlines.

## Robust Face/Polygon Normals

**Status:** Resolved

- **Issue:** `MeshGeometry::compute_normal_from_vertices()` previously relied on a first-triangle cross product, which is not robust for warped polygons or collinear first vertices.
- **Fix:** Use Newell’s method via `PolyGeometry::polygon_normal()` for 3D polygon normals (and keep 2D edge-normal logic).
- **Code:** `Code/Source/solver/Mesh/Geometry/MeshGeometry.cpp`
- **Tests:** `Code/Source/solver/Mesh/Tests/Unit/Geometry/test_MeshGeometry.cpp`
- **References:**
  - Newell, M. E. (1972). *The Utilization of Computer Graphics*. (Newell normal for polygons; widely reused in graphics/FEM preprocessing code.)

## Concave Polygon Centroids + Planar Triangulation (PolyGeometry)

**Status:** Resolved

- **Issue:** Triangle-fan formulas for polygon centroid/triangulation can fail for concave polygons unless the fan root is in the kernel; polyhedron routines inherited this weakness.
- **Fix:**
  - Compute planar polygon centroids using projected shoelace / Green’s theorem centroid on a dominant-axis projection, then lift back to 3D.
  - Add planar polygon triangulation via ear clipping, rejecting self-intersecting polygons.
  - Use triangulation (not fan) when decomposing polyhedron faces for mass properties.
- **Code:** `Code/Source/solver/Mesh/Geometry/PolyGeometry.cpp`, `Code/Source/solver/Mesh/Geometry/PolyGeometry.h`
- **Tests:** `Code/Source/solver/Mesh/Tests/Unit/Geometry/test_PolyGeometry.cpp`
- **References:**
  - Meisters, G. H. (1975). “Polygons have ears.” *The American Mathematical Monthly*.
  - O’Rourke, J. (1998). *Computational Geometry in C* (2nd ed.).

## Tessellation: Concave Faces + Conforming Local Adaptivity

**Status:** Resolved

- **Issue A (concave polygon faces):** Face tessellation used triangle fans, which can produce incorrect triangles/areas on concave polygons.
- **Fix A:** Triangulate planar polygon faces using `PolyGeometry::triangulate_planar_polygon()` with fan fallback if triangulation fails.
- **Issue B (local-adaptive quads):** Local refinement could introduce hanging nodes / nonconforming edges if quads are emitted without balancing.
- **Fix B:** Balance the refinement quadtree (2:1) and emit conforming triangles using shared mid-edge points.
- **Code:** `Code/Source/solver/Mesh/Geometry/Tessellation.cpp`, `Code/Source/solver/Mesh/Geometry/Tessellation.h`
- **Tests:** `Code/Source/solver/Mesh/Tests/Unit/Geometry/test_Tessellation.cpp`
- **References:**
  - Burstedde, C. et al. (2011). “p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement…” *SIAM J. Sci. Comput.* (2:1 balancing practice).
  - deal.II library documentation: adaptive refinement + conformity strategies for quads/hexes.

## GeometryCache: Correct Stats + Invalidation + Warm Cache

**Status:** Resolved

- **Issue:** Cache hit counters were inferred from value type (`real_t` vs `std::array`), conflating cell/face/edge quantities; missing tests for bbox caching, warm-cache, and topology invalidation.
- **Fix:**
  - Track stats per cached quantity (cell/face/edge/mesh bbox, etc.).
  - Increment hits/misses at the cache access site (not by value type).
  - Add unit tests for bbox caching, warm-cache, and invalidation on geometry/topology change events.
- **Code:** `Code/Source/solver/Mesh/Geometry/GeometryCache.cpp`, `Code/Source/solver/Mesh/Geometry/GeometryCache.h`
- **Tests:** `Code/Source/solver/Mesh/Tests/Unit/Geometry/test_GeometryCache.cpp`
- **References:**
  - Typical lazy-cache + observer invalidation patterns as used in large FE/mesh codes (e.g., PETSc DMPlex, deal.II triangulation caches).

## CurvilinearEval: Shape Functions, Derivatives, Jacobians, Inverse Map

**Status:** Resolved

- **Issue:** High-order evaluation lacked derivative/Jacobian unit tests; pyramid higher-order tests were conditionally compiled out.
- **Fix:**
  - Add tests for: derivative partition-of-unity (`∑ dN/dξ = 0`), finite-difference validation of `dN/dξ`, and Jacobian finite-difference consistency.
  - Run pyramid P5/P7 tests unconditionally (no Eigen dependency).
- **Tests:** `Code/Source/solver/Mesh/Tests/Unit/Geometry/test_CurvilinearEval.cpp`
- **References:**
  - Hughes, T. J. R. (2000). *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis* (shape function and mapping identities).
  - Bergot, M., Cohen, G., & Duruflé, M. (2009). “Higher-order finite elements for hybrid meshes using new nodal pyramidal elements.” *J. Sci. Comput.* (pyramid basis construction; see also implementations in MFEM / deal.II-style pyramids).

## BoundingVolume: AABB/OBB Construction + BVH Tree Logic

**Status:** Resolved

- **Issue:** Bounding volume construction and BVH/AABB tree logic lacked unit tests; Eigen detection needed to match build configuration.
- **Fix:**
  - Gate Eigen-dependent OBB PCA paths on `MESH_HAS_EIGEN`.
  - Make `BoundingSphere::contains()` conservative with a small floating tolerance to avoid false negatives on boundary points (broad-phase containment should prefer false positives over false negatives).
  - Add tests for cell/face/edge AABBs, mesh AABB, bounding spheres, OBB-to-AABB conversion, AABB intersections, and AABB tree partitioning.
- **Code:** `Code/Source/solver/Mesh/Geometry/BoundingVolume.cpp`, `Code/Source/solver/Mesh/Geometry/BoundingVolume.h`
- **Tests:** `Code/Source/solver/Mesh/Tests/Unit/Geometry/test_BoundingVolume.cpp`
- **References:**
  - Standard BVH construction patterns appear in many geometry kernels; FE codes often reuse AABB trees for contact/search (e.g., deal.II, MFEM coupling workflows).

## MeshQuality: Wedge/Pyramid Coverage + Distortion Sensitivity

**Status:** Resolved

- **Issue:** Quality metrics existed for wedge/pyramid, but unit tests focused on tetra/hex/quad.
- **Fix:**
  - Add wedge/pyramid tests for edge ratio and Jacobian-based metrics; verify distortion reduces scaled Jacobian without inversion.
  - Exclude the singular pyramid apex (`z=1`) from Jacobian-based sampling to avoid spurious “degenerate” reports.
- **Code:** `Code/Source/solver/Mesh/Geometry/MeshQuality.cpp`
- **Tests:** `Code/Source/solver/Mesh/Tests/Unit/Geometry/test_MeshQuality.cpp`
- **References:**
  - Knupp, P. (2001). “Algebraic mesh quality metrics.” *SIAM J. Sci. Comput.*
  - Sandia “VERDICT” (widely used mesh quality metric definitions; used/ported in several FE/mesh stacks including libMesh ecosystems).

## Optional Future Work (Not Blocking Production Readiness)

**Status:** Not started (optional)

### Robust geometric predicates for triangulation/projection

- **Motivation:** Ear clipping + orientation tests can be sensitive near degeneracy; robust predicates reduce false self-intersection/ear rejection.
- **Outline:**
  - Add `Code/Source/solver/Mesh/Geometry/RobustPredicates.h/.cpp` implementing adaptive `orient2d` / `incircle` (e.g., Shewchuk-style).
  - Use predicates inside `PolyGeometry::triangulate_planar_polygon()` to improve stability on near-collinear inputs.
  - Add fuzz/degeneracy regression tests in `Code/Source/solver/Mesh/Tests/Unit/Geometry/test_PolyGeometry.cpp`.
- **References:** Shewchuk, J. R. (1997). “Adaptive Precision Floating-Point Arithmetic and Fast Robust Predicates…” *Discrete & Computational Geometry*.

### High-order BVH / curved-geometry bounds

- **Motivation:** For curved (high-order) elements, vertex AABBs can under-bound true geometry between nodes.
- **Outline:**
  - Extend `BoundingVolumeBuilder` with sampling-based bounds via `CurvilinearEvaluator::evaluate_geometry()` at reference sample points (corners + edge/face interiors).
  - Add tests that compare vertex-only bounds vs sampled bounds for curved cells.
- **Files:** `Code/Source/solver/Mesh/Geometry/BoundingVolume.h/.cpp`, new unit tests in `Code/Source/solver/Mesh/Tests/Unit/Geometry/test_BoundingVolume.cpp`.
- **References:** Common practice in high-order FE libraries (MFEM, deal.II) for curved element bounding/visualization.
