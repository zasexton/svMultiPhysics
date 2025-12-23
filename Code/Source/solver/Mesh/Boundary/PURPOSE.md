# Boundary Folder - Purpose and Scope

## Purpose

The **Boundary** subfolder provides **topological and geometric** operations for detecting and analyzing boundaries of finite cell complexes (meshes). It focuses exclusively on mesh structure and spatial properties, **not** on physics or simulation-specific functionality.

## Core Responsibilities

### 1. Topological Boundary Detection
- Identify (n-1) entities incident to exactly one n-cell
- Classify (n-1) entities as boundary, interior, or non-manifold
- Detect connected boundary components
- Handle mixed element meshes (tet, hex, wedge, pyramid, triangle, quad)

### 2. Geometric Boundary Analysis
- Provide outward orientation (right-hand rule convention) for boundary entities
- Return oriented vertex order so Geometry can compute normals/areas/centroids
- Support 1D (vertex boundaries), 2D (edge boundaries), and 3D (face boundaries)

### 3. Canonical Representation
- Provide orientation-independent boundary keys for topology
- Provide orientation-preserving keys for geometry
- Enable efficient incidence counting via hash maps

### 4. Component Analysis
- Extract connected components of boundary surfaces
- Provide component membership (which boundary entities belong to each component)
- Store per-component fields (e.g. `closed`, `orientable`, `area`, `centroid`) for other modules to populate

## What This Folder DOES Include

✓ **Topological operations**
  - Boundary detection via incidence counting
  - Chain complex approach (mod 2 arithmetic)
  - Connectivity analysis
  - Non-manifold detection

✓ **Geometric computations**
  - Vertex ordering conventions (right-hand rule / CCW)
  - Oriented boundary entities for Geometry computations

✓ **Data structures**
  - `BoundaryKey`: Canonical boundary representation
  - `BoundaryComponent`: Connected boundary patch representation
  - `BoundaryDetector`: Main detection and analysis engine

✓ **Mesh introspection**
  - Query which (n-1) entities are on boundary
  - Identify boundary vertices

## What This Folder DOES NOT Include

✗ **Finite Element Method (FEM) concepts**
  - Boundary conditions (Dirichlet, Neumann, Robin, etc.)
  - Basis functions or shape functions
  - Quadrature rules or integration points
  - Degrees of freedom management

✗ **Physics-specific functionality**
  - Material properties
  - Constitutive models
  - Force/flux computations
  - Solver-specific data structures

✗ **Simulation workflow**
  - Time integration
  - Linear system assembly
  - Constraint enforcement
  - Load application

✗ **Analysis-specific features**
  - Stress/strain computation
  - Post-processing of solution fields
  - Error estimation
  - Adaptive refinement criteria

## Design Philosophy

The Boundary folder follows the **separation of concerns** principle:

**Mesh Layer (this folder):**
- "Where is the boundary?"
- "What is the boundary geometry?"
- "How are boundary entities connected?"

**Solver/Analysis Layer (separate folders):**
- "What boundary conditions apply?"
- "How do I enforce constraints?"
- "What physics govern the boundary?"

This separation enables:
- Reusability across different physics (CFD, structures, electromagnetics)
- Clear dependencies (Mesh → Topology/Geometry only)
- Testability (boundary detection independent of solver)
- Maintainability (changes to BC enforcement don't affect boundary detection)

## Integration with Mesh Infrastructure

The Boundary folder operates on and returns `MeshBase` entities:

```
MeshBase (Core/MeshBase.h)
    ↓
BoundaryDetector (Boundary/BoundaryDetector.h)
    ↓
BoundaryInfo (entity indices + oriented vertices + types)
    ↓
Used by: Geometry, Validation, I/O
Not used by: Boundary folder doesn't know about FEM concepts
```

## Typical Usage Pattern

```cpp
// ===== In Mesh code (appropriate) =====
MeshBase mesh = /* ... */;
BoundaryDetector detector(mesh);
auto boundary_info = detector.detect_boundary();

// Boundary entities are returned as IDs into boundary_info.entity_keys
for (index_t ent_id : boundary_info.boundary_entities) {
    const auto& key = boundary_info.entity_keys[ent_id];
    // key.vertices() are the (corner) vertex IDs defining this boundary entity
}

// ===== In Solver code (separate layer) =====
// If your workflow uses labeled boundary faces/edges, use MeshBase's explicit codim-1 entities:
// (MeshBase::finalize() can build these for standard cell families.)
mesh.finalize();
for (index_t f : mesh.boundary_faces()) {
    label_t bc_label = mesh.boundary_label(f);
    // Apply physics-specific BC based on label (solver responsibility).
}
```

## Future Directions (Within Scope)

Potential enhancements that remain topological/geometric:

- **Boundary curve extraction**: Extract 1D feature curves from boundaries
- **Feature detection**: Identify sharp edges, corners (geometric analysis)
- **Orientability checking**: Detect non-orientable surfaces (topology)
- **Boundary mesh extraction**: Create standalone (n-1)-D mesh from boundary
- **Spatial acceleration**: BVH for boundary-specific queries (performance)
- **Boundary integration**: Numerical quadrature over boundary surfaces (geometry)

These are all mesh-level operations and would be appropriate additions.

## Out of Scope (Belongs Elsewhere)

Features that should live in solver/physics layers:

- **Boundary condition classes**: `DirichletBC`, `NeumannBC`, etc. → Solver folder
- **Constraint enforcement**: Penalty methods, Lagrange multipliers → Solver folder
- **Load application**: Traction, pressure, heat flux → Physics folder
- **Essential BC handling**: DOF elimination, master-slave → Solver folder
- **Weak form contributions**: Surface integrals in variational form → FEM folder

## Summary

The Boundary folder is a **mesh utility module** providing topological boundary detection and geometric boundary analysis. It knows about mesh structure and spatial properties but has no knowledge of:
- What physical quantities exist on the boundary
- What constraints should be enforced
- How the solver uses boundary information

This clean separation maintains the Mesh folder's role as a general-purpose mesh management library usable across different simulation domains.
