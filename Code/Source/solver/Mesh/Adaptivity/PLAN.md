**Adaptivity Plan**

- Purpose: Provide adaptive mesh refinement (AMR) and coarsening capabilities for MeshBase, including error estimation, element marking, refinement/coarsening operators, conformity handling, field transfer, and quality safeguards. The design must work for mixed meshes, support both reference and current configurations, and integrate with DistributedMesh.

**Scope & Responsibilities**
- Error Estimation
  - Element‑wise indicators (residual‑based, jump indicators, gradient recovery, user‑provided fields)
  - Optional multi‑criteria aggregation (e.g., weighted sum of indicators)
- Marking Strategies
  - Fixed‑fraction (bulk) marking (Doerfler)
  - Threshold/absolute/relative marking
  - Region/label‑aware marking (e.g., skip boundary tag X)
- Refinement/Coarsening Operators
  - 1D/2D/3D families: Line, Triangle, Quad, Tetra, Hex, Wedge, Pyramid
  - Conforming refinement patterns (red/green/blue) and minimal closure
  - Coarsening with safety checks (only collapse admissible refinement patterns)
- Conformity & Constraints
  - Closure to eliminate hanging entities (or explicitly support hanging nodes via constraints)
  - Orientation/Permutation awareness for sub‑entities
- Field Transfer
  - Prolongation (coarse → fine) and restriction (fine → coarse) for vertex/cell fields
  - Conservative options for cell‑integrated quantities
- Quality & Smoothing
  - Check post‑refine quality (MeshQuality); optional smoothing/redistribution for poor elements
  - Abort/rollback strategies for catastrophic quality degradation (dev/debug)
- Parallel/Distributed
  - Partition‑aware marking and refinement; ghost layer rebuild; inter‑rank consistency

**How It Integrates**
- MeshBase
  - Uses topology (cell2vertex, face2vertex) to build refined connectivity
  - Calls `build_from_arrays()` for new meshes or in‑place operators with careful updates + `finalize()`
  - Updates labels/sets (inherit or recompute on refined entities)
  - Emits events: `MeshEvent::AdaptivityApplied`, `MeshEvent::TopologyChanged`, `MeshEvent::GeometryChanged` (if coordinates change), `MeshEvent::LabelsChanged` (if labels are touched)
- Observer Bus
  - Consumers (Search/Geometry caches) invalidate on events
- Fields
  - Uses `MeshFields` to reattach/reallocate and perform prolongation/restriction
- Topology/Orientation
  - Reuse `CellTopology` for sub‑entity definitions
  - (Optional) `Geometry/MeshOrientation.h` orientation codes to keep face/edge DOF consistency
- DistributedMesh
  - Global marking synchronization (union/majority), refine/coarsen locally, then `PartitionChanged` via local mesh bus and ghost rebuild

**Proposed File Layout (to be implemented)**
- `Adaptivity/AdaptivityManager.h/.cpp`
  - High‑level façade orchestrating the adaptivity pipeline (estimate → mark → refine/coarsen → transfer → quality checks → finalize)
- `Adaptivity/ErrorEstimator.h/.cpp`
  - Interfaces + concrete estimators (gradient recovery, jump, residual proxy, user field)
  - API: `std::vector<real_t> estimate(const MeshBase&, const MeshFields::FieldHandle& /*optional*/)`
- `Adaptivity/Marker.h/.cpp`
  - Marking strategies (bulk, threshold, fixed count)
  - API: `std::vector<bool> mark(const std::vector<real_t>& indicators, const Options&)`
- `Adaptivity/RefinementRules.h/.cpp`
  - Per‑family refinement patterns
  - Returns child connectivity, vertex insertion policy (mid‑edge, face center, cell center)
- `Adaptivity/CoarseningRules.h/.cpp`
  - Detect coarsenable aggregates and build coarse connectivity
- `Adaptivity/Conformity.h/.cpp`
  - Closure logic to remove hanging entities or build hanging‑node constraints
- `Adaptivity/FieldTransfer.h/.cpp`
  - Prolongation/restriction policies (vertex‑wise copy/interp, cell‑wise aggregation/conservation)
- `Adaptivity/QualityGuards.h/.cpp`
  - After‑adapt checks with `MeshQuality`; optional smoothing/untangling hooks (future)
- `Adaptivity/PartitionUtils.h/.cpp` (phase 2)
  - Rank‑aware marking merge; ghost/owner handling; work partitioning
- `Adaptivity/Options.h`
  - Unified options (refine/coarsen toggles, limits, quality thresholds, estimator config)

**Data Flow (Refine Path)**
1) Estimate indicators → `ErrorEstimator`
2) Mark cells → `Marker`
3) Build refined connectivity (new vertices, child cells) → `RefinementRules`
4) Enforce conformity (closure / constraints) → `Conformity`
5) Create new mesh or update in place (prefer new mesh for clarity) → `MeshBase::build_from_arrays()` + `finalize()`
6) Transfer fields → `FieldTransfer`
7) Quality checks → `QualityGuards` (optionally smooth)
8) Notify events (`AdaptivityApplied`, `TopologyChanged`, `FieldsChanged`, `LabelsChanged` when applicable)

**Event Wiring**
- Emit via `mesh.event_bus().notify(MeshEvent::AdaptivityApplied)` after successful pipeline
- Also emit `TopologyChanged` and `GeometryChanged` (if coordinates added, e.g., mid‑edge points in current config) and `FieldsChanged` after transfer
- For distributed workflows, after adaptivity + ghost rebuild: `PartitionChanged`

**Distributed Considerations (Phase 2)**
- Global consistency: shared interfaces must produce compatible splits across neighboring ranks
- Strategy options:
  - Pre‑mark exchange + deterministic tie‑breaking
  - Post‑refine closure across ranks, then re‑balance (optional `DistributedMesh::rebalance()`)
- Communication patterns reuse `DistributedMesh` exchange facilities

**Milestones**
- M1 (serial): Triangle/Tetra red‑refine; bulk marker; vertex‑based field prolongation; rebuild faces/edges; unit tests
- M2 (serial): Quad/Hex refinement; threshold marker; conservative cell field transfer; quality checks
- M3 (serial): Coarsening (safe patterns); conformity closure; labels/sets propagation
- M4 (distributed): Partition‑aware marking + refinement; ghost rebuild; PartitionChanged event; tests with 2–4 ranks
- M5 (advanced): Hanging‑node constraints option; high‑order hints; smoothing/untangling hooks

**Testing Strategy**
- Unit tests per module:
  - Rules: child counts, connectivity, vertex creation uniqueness
  - Marker: bulk/threshold correctness
  - Transfer: simple fields (constants/gradients) preserved as expected
  - Conformity: no hanging entities (or constraints map correct)
  - Quality: metrics bounded by thresholds
- Integration tests:
  - Single‑step refine/coarsen on toy meshes (tri/tet/hex)
  - Multi‑step refine cycles; verify counts and measures
  - (Phase 2) Distributed consistency across ranks

**Notes & Future Work**
- High‑order/curvilinear elements: initial support assumes linear geometry; later integrate with `Geometry/CurvilinearEval` and shape libraries
- NURBS/IGA: out of initial scope
- Provenance: persist parent/child maps to enable error localization and multigrid transfer

