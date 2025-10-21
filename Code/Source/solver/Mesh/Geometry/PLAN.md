**Geometry Plan**

- Purpose: Provide robust, dimension‑aware geometric computations for `MeshBase` with clear separation from topology and fields, and predictable behavior for mixed/linear/high‑order elements in both reference and current configurations.
- Scope: Cell/face/edge centers and measures, normals, distances, bounding volumes, vector utilities, optional caches, and high‑order/curvilinear evaluation support.

**Responsibilities**
- Core queries (already exposed via `MeshGeometry`):
  - Centers: `cell_center`, `face_center`, `edge_center`
  - Measures: `cell_measure` (length/area/volume), `face_area`, `edge_length`, `total_volume`, `boundary_area`
  - Normals: `face_normal` (unit), `face_normal_unnormalized`, `edge_normal` (2D/3D semantics)
  - Bounding boxes: mesh/cell/face level
  - Generic utilities: cross, dot, normalize, magnitude, distance, angles
- Specialized shape formulas:
  - Tetra/Hex/Wedge/Pyramid volumes, triangle/quad/polygon area
  - Polygon and polyhedron support via triangulation/Newell’s method
- Configuration awareness:
  - All APIs accept `Configuration` (Reference/Current) and select the correct coordinate set
- Mixed elements:
  - Handle per‑cell `CellShape` and vertex spans without assuming uniform arity
- Robustness:
  - Use consistent numeric tolerances; guard degeneracies; avoid undefined normals

**Integration Points**
- MeshBase delegates geometry methods to `MeshGeometry` (already implemented)
- Observer bus:
  - If a geometry cache is added, register a `CacheInvalidator` to clear on `TopologyChanged` and `GeometryChanged`
- Quality module:
  - `MeshQuality` builds on geometry utilities; keep APIs cohesive and avoid duplication
- Search module:
  - Reuse bounding boxes and centers; provide fast helpers for AABB construction

**Planned File Layout**
- Existing
  - `Geometry/MeshGeometry.h/.cpp` – primary APIs (centers, normals, measures, bboxes, utilities)
  - `Geometry/MeshQuality.h/.cpp` – cell quality metrics
  - `Geometry/MeshOrientation.h` – permutation codes for sub‑entity orientation (tri/quad/edge), orientation manager scaffold
- New (to add incrementally)
  - `Geometry/GeometryConfig.h`
    - Numeric tolerances (e.g., area/volume eps), configuration flags, policy enums
  - `Geometry/PolyGeometry.h/.cpp`
    - Robust polygon/polyhedron area/centroid/normal using Newell’s method and triangulation helpers
    - Shared by `MeshGeometry` for non‑standard faces/cells
  - `Geometry/BoundingVolume.h/.cpp`
    - Helpers for AABB/OBB builders; batched bbox for all cells/faces; SIMD‑friendly loops
  - `Geometry/GeometryCache.h/.cpp` (optional)
    - Lazy caches for per‑cell/face measures, centers, and bounding boxes; subscribes to Mesh events
  - `Geometry/CurvilinearEval.h/.cpp` (phase 2)
    - Evaluate high‑order/curvilinear geometry at parametric points (J, detJ, mapping)
    - Integrates with future shape‑function/element mapping services
  - `Geometry/Tessellation.h/.cpp` (phase 2)
    - Linearization/tessellation of high‑order cells/faces for I/O/visualization (triangulate quads/polygons, subdivide curved)

**Design Notes**
- APIs remain stateless; any caching lives in `GeometryCache` with explicit invalidation via the Observer bus
- Prefer safe, general algorithms for polygons/polyhedra; fall back to conservative outputs on degeneracy
- Keep tolerances centralized in `GeometryConfig` and plumbed through where relevant
- Ensure 2D/3D semantics are documented (edge normals in 2D lie in xy‑plane; z=0)

**Milestones**
- M1: Solidify current `MeshGeometry` (unit tests for tet/tri/hex/wedge/pyr measures and centers; triangle/quad normals; bbox)
- M2: Add `PolyGeometry` for general polygons/polyhedra + tests; integrate into `MeshGeometry` for mixed meshes
- M3: Introduce `GeometryConfig` and optional `GeometryCache` (subscribe to Mesh events)
- M4: Curvilinear/high‑order evaluation scaffolding (`CurvilinearEval`) with basic mapping/Jacobian utilities
- M5: Tessellation helpers for high‑order output/visualization

**Testing**
- Unit tests under `Tests/Unit/Geometry/`:
  - Vector utilities, distances, angles
  - Centers and measures for common elements (2D/3D)
  - Face normals (orientation correctness on oriented vertex lists)
  - Poly/Polyhedron: area/centroid/normal (planar polygons, simple convex polyhedra)
  - BBox correctness for mixed meshes and both configurations

**Observer Wiring (if cache added)**
- Subscribe `GeometryCache` to `MeshEvent::TopologyChanged` and `MeshEvent::GeometryChanged` via `CacheInvalidator`
- For distributed setups, revalidate on `PartitionChanged` when geometry depends on owned/ghost partitions (rare, document if used)

