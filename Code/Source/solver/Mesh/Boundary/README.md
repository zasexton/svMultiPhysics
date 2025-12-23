# Boundary Detection and Analysis

This subfolder implements topological boundary detection for finite cell/simplicial complexes.

## Overview

The **boundary** of an _n_-dimensional mesh (treated as a finite cell complex) is the set of (_n_-1)-faces incident to exactly one _n_-cell. This is a purely topological concept that requires no geometric information—only incidence relationships.

## Mathematical Foundation

### Topological Definition

For an _n_-dimensional mesh:
- **Boundary faces**: (_n_-1)-faces with incidence count = 1
- **Interior manifold faces**: (_n_-1)-faces with incidence count = 2
- **Non-manifold seams**: (_n_-1)-faces with incidence count > 2

### Two Computation Methods

#### 1. Incidence Counting (Implemented)

**Algorithm:**
1. For each _n_-cell, enumerate its (_n_-1)-faces
   - Tetrahedra → 4 triangular faces
   - Hexahedra → 6 quadrilateral faces
   - Triangles → 3 edges
   - etc.

2. Canonicalize each face key (sorted vertex IDs)

3. Count how many _n_-cells reference each face

4. Classify:
   - count = 1 → boundary face
   - count = 2 → interior manifold face
   - count > 2 → non-manifold seam (flag for QA)

**Features:**
- Manifold-agnostic
- Integer arithmetic only
- Works with mixed element meshes
- Detects non-manifold features

#### 2. Chain Complex View (over ℤ₂)

Build the signed cell-face incidence matrix **A**ₙ ∈ {-1, 0, +1}^(Nₙ₋₁ × Nₙ).

Over ℤ₂ (mod 2), compute:
```
b = (Aₙ mod 2) · 1 mod 2
```

Boundary faces have bᵢ = 1 (odd incidence).

## Key Classes

### `BoundaryKey`
Canonical boundary representation using sorted vertex indices. Provides:
- Hash function for `unordered_map`
- Comparison operators for `map`
- Orientation-independent representation
- Works for boundaries of any dimension (edges in 2D, faces in 3D, etc.)

### `OrientedBoundaryKey`
Oriented boundary key that preserves vertex ordering and sign. Used when orientation information is needed for:
- Computing outward-pointing normals
- Chain complex with signed incidence matrices

`OrientedBoundaryKey` is currently an auxiliary type; `BoundaryDetector` returns oriented vertex lists directly.

### `BoundaryComponent`
Represents a connected component of the boundary. Each component is a maximal connected set of boundary (_n_-1)-faces, where connectivity is defined through shared (_n_-2)-faces.

Features:
- Face and vertex storage
- Fields to store topological/geometric properties (e.g. `closed`, `orientable`, `area`, `centroid`) that can be populated by other modules (e.g. `Geometry/`, `Validation/`)

### `BoundaryDetector`
Main detection engine. Provides:

**Detection Methods:**
- `detect_boundary()` - Full boundary analysis with components and oriented faces
- `detect_boundary_chain_complex()` - Chain complex approach (ℤ₂)
- `compute_boundary_incidence()` - Raw incidence counts with orientation information

**Utilities:**
- `extract_boundary_components()` - Connected component analysis via BFS
- `detect_nonmanifold_codim1()` - Find non-manifold (n-1) entities
- `is_closed_mesh()` - Check if mesh has no boundary

**Note on Geometric Computations:**
Geometric properties (normals, areas, centroids) are computed by `MeshGeometry` (in `Geometry/MeshGeometry.h`):
- `MeshGeometry::compute_normal_from_vertices()` - Normal from oriented vertices
- `MeshGeometry::compute_area_from_vertices()` - Area/length from vertices
- `MeshGeometry::compute_centroid_from_vertices()` - Centroid from vertices
- `MeshGeometry::compute_bounding_box_from_vertices()` - Bounding box from vertices

## Vertex Ordering Convention: Right-Hand Rule

The boundary detector uses a **right-hand rule convention** for oriented boundary faces to ensure consistent outward-pointing normals.

### 3D Meshes (Boundary Faces)

For each cell type, boundary faces are extracted with vertices ordered such that:
- Following the right-hand rule produces an **outward-pointing normal**
- Normal = (v₁ - v₀) × (v₂ - v₀) points away from the cell interior

**Standard Orderings:**
- **Tetrahedron**: Faces ordered counter-clockwise when viewed from outside
- **Hexahedron**: Quad faces ordered counter-clockwise from exterior
- **Wedge (Prism)**: Triangle and quad faces with outward normals
- **Pyramid**: Base and triangular faces with outward normals

### 2D Meshes (Boundary Edges)

For 2D cells, boundary edges are oriented such that:
- The outward normal is a 90° counter-clockwise rotation of the edge vector
- Normal = (-dy, dx) where (dx, dy) is the edge vector

### Usage

```cpp
#include "Geometry/MeshGeometry.h"

auto info = detector.detect_boundary();

// Access oriented boundary (n-1) entities (vertices in right-hand rule order)
for (size_t i = 0; i < info.boundary_entities.size(); ++i) {
    const auto& oriented_verts = info.oriented_boundary_entities[i];

    // Compute outward-pointing normal (via Geometry)
    auto normal = MeshGeometry::compute_normal_from_vertices(mesh, oriented_verts);

    // Compute area
    auto area = MeshGeometry::compute_area_from_vertices(mesh, oriented_verts);

    std::cout << "Boundary " << i << ": "
              << "Normal = [" << normal[0] << ", " << normal[1] << ", " << normal[2] << "], "
              << "Area = " << area << std::endl;
}
```

## Usage Example

```cpp
#include "Boundary/BoundaryDetector.h"

// Given a mesh
MeshBase mesh = /* ... */;

// Create detector
BoundaryDetector detector(mesh);

// Detect boundary
auto info = detector.detect_boundary();

// Check results
if (info.has_boundary()) {
    std::cout << "Boundary (n-1) entities: " << info.boundary_entities.size() << std::endl;
    std::cout << "Boundary vertices: " << info.boundary_vertices.size() << std::endl;
    std::cout << "Components: " << info.n_components() << std::endl;
}

// Check for non-manifold features
if (info.has_nonmanifold()) {
    std::cout << "Warning: Non-manifold boundary entities detected!" << std::endl;
}

// Analyze each component
for (const auto& comp : info.components) {
    std::cout << "Component " << comp.id() << ": "
              << comp.n_entities() << " entities, "
              << comp.n_vertices() << " vertices" << std::endl;
}

// Compute geometric properties for boundary faces
for (size_t i = 0; i < info.boundary_entities.size(); ++i) {
    const auto& oriented_verts = info.oriented_boundary_entities[i];

    // Compute normal (via MeshGeometry)
    auto normal = MeshGeometry::compute_normal_from_vertices(mesh, oriented_verts);

    // Compute area
    auto area = MeshGeometry::compute_area_from_vertices(mesh, oriented_verts);

    // Compute centroid
    auto centroid = MeshGeometry::compute_centroid_from_vertices(mesh, oriented_verts);

    // Use properties for boundary conditions, flux computation, etc.
}
```

## Edge Cases Handled

### Mixed Element Meshes
Each cell type (tet, hex, wedge, pyramid) has specialized face extraction. The algorithm works seamlessly with meshes containing multiple cell types.

### Non-Pure Complexes
Restricted to maximal _n_-cells when forming faces.

### Non-Manifold/Cracked Meshes
Faces with incidence ≠ 2 are diagnostic. The detector flags faces with count > 2 for quality assurance.

### Periodic/Topologically Closed Meshes
Every (_n_-1)-face has even incidence → empty boundary. Use `is_closed_mesh()` to check.

### Connected Boundary Components
After detecting boundary faces, the algorithm builds their adjacency graph through shared (_n_-2)-faces and runs BFS to label connected components.

## Implementation Details

### Topology-Driven Design

The boundary detector uses `CellTopology` (in `Topology/CellTopology.h`) to obtain canonical face/edge definitions for each cell type. This **eliminates switch statements** and makes adding new cell types trivial:

```cpp
// Get face topology from CellTopology (no switch statements!)
auto face_defs = CellTopology::get_boundary_faces(shape.family);
auto oriented_face_defs = CellTopology::get_oriented_boundary_faces(shape.family);

// Apply topology to this cell's actual vertices
for (size_t i = 0; i < face_defs.size(); ++i) {
    std::vector<index_t> face_vertices;
    for (index_t local_idx : face_defs[i]) {
        face_vertices.push_back(vertices_ptr[local_idx]);
    }
    // Process face...
}
```

**Adding a new cell type:**
1. Add face definitions to `CellTopology.cpp` (one place)
2. No changes needed in `BoundaryDetector` or any other code
3. All algorithms automatically work with the new cell type

### Face Canonicalization
Vertex IDs are sorted to create orientation-independent keys. This ensures that a face appears identical regardless of which cell references it or in what vertex order.

### Hash Function
Uses a simple but effective hash combination:
```cpp
hash ^= hash(vertex_id) + 0x9e3779b9 + (hash << 6) + (hash >> 2)
```

### Robustness
Uses integer vertex IDs exclusively—no coordinate comparisons that require tolerance.

### Right-Hand Rule Implementation

The boundary detector maintains **two representations** for each boundary:

1. **Canonical (sorted)**: Used for incidence counting and topology detection
   - Vertices sorted to create orientation-independent key
   - Enables fast hash-based lookup
   - Used to identify which faces are on the boundary

2. **Oriented (right-hand rule)**: Used for geometric computations
   - Vertices in original order from cell definition (from `CellTopology`)
   - Preserves orientation for outward-pointing normals
   - Stored in `BoundaryIncidence::oriented_vertices` and `BoundaryInfo::oriented_boundary_entities`

**Entity IDs**:
- `BoundaryInfo::boundary_entities`, `interior_entities`, and `nonmanifold_entities` store indices into `BoundaryInfo::entity_keys`.

This dual representation ensures:
- Efficient topological detection (via canonical keys)
- Accurate normal computation (via oriented vertices)
- No need to recompute orientation from geometry

## Performance Considerations

- **Time Complexity**: O(n_cells × faces_per_cell) for incidence counting
- **Space Complexity**: O(unique_faces) for hash map storage
- **Cache Efficiency**: Face keys are small and hash lookups are fast

## Future Extensions

- Orientation-sensitive boundary operator over ℤ (signed incidence)
- Parallel boundary detection for distributed meshes
- Boundary curve extraction in 2D
- Feature edge detection (sharp creases, ridges)
