# Mesh Constraints

This subfolder implements **topological and geometric constraints** for mesh entities.

## Purpose

The Constraints folder provides mechanisms to represent and manage special relationships between mesh entities that arise from:

1. **Adaptive mesh refinement** (hanging vertices)
2. **Periodic boundary conditions** (topologically equivalent entities)
3. **Non-conforming mesh interfaces** (mortar methods)
4. **Multi-point geometric constraints** (vertex tying)

**Key principle**: This folder handles **topology and geometry only**. It does NOT implement:
- Physics-specific constraints (contact, friction, etc.)
- FEM-specific constraints (DOF coupling, boundary conditions)
- Constraint enforcement (that belongs in the Solver)

## Scope

### ✓ What This Folder DOES Include

#### 1. Hanging Vertex Constraints
**Purpose**: Track parent-child relationships for vertices created during adaptive refinement.

**Topological nature**:
- Which vertices are "hanging" (dependent on parent entities)
- Which edges/faces contain hanging vertices
- Parent-child connectivity for constraint equations

**Use cases**:
- Adaptive mesh refinement (h-refinement)
- Mesh coarsening
- Transition between refinement levels

**Example**:
```cpp
// A hanging vertex on an edge depends on the edge's endpoint vertices
HangingVertexConstraint hnc;
hnc.constrained_vertex = 5;
hnc.parent_entity = EntityKind::Edge;
hnc.parent_id = 12;
hnc.parent_vertices = {3, 4};        // Endpoints of edge 12
hnc.weights = {0.5, 0.5};         // Linear interpolation
```

#### 2. Periodic Constraints
**Purpose**: Identify topologically equivalent entities on opposite periodic boundaries.

**Topological nature**:
- Which vertex pairs are periodic equivalents
- Which face pairs match on opposite boundaries
- Transformation/translation vectors between periodic surfaces

**Use cases**:
- Periodic domains (unit cells, turbomachinery)
- Symmetry exploitation
- Mesh topology verification

**Example**:
```cpp
// Vertices 10 and 20 are periodic equivalents
PeriodicConstraint pc;
pc.entity_kind = EntityKind::Vertex;
pc.primary_id = 10;
pc.secondary_id = 20;
pc.translation = {1.0, 0.0, 0.0};  // Offset vector
pc.rotation_matrix = identity;     // Optional rotation
```

#### 3. Mortar Interface Topology
**Purpose**: Track topology of non-conforming mesh interfaces (mortar methods).

**Topological nature**:
- Which faces/vertices belong to each side of the interface
- Interface pairing (master/slave surface identification)
- Geometric overlap detection

**Important distinction**:
- **Mesh folder**: Identifies interface topology and geometry
- **Solver folder**: Computes projection operators and enforces continuity

**Use cases**:
- Non-conforming interfaces in multi-domain problems
- Sliding interfaces
- Fluid-structure interaction surface topology

**Example**:
```cpp
// Mortar interface between two non-conforming surfaces
MortarInterface mi;
mi.master_faces = {10, 11, 12};
mi.slave_faces = {25, 26, 27, 28};
mi.overlaps = compute_surface_overlaps(master, slave);  // Geometry
// Note: Projection operators computed in Solver, not here
```

#### 4. Multi-Point Constraints (Geometric)
**Purpose**: Enforce geometric relationships between vertices (e.g., vertex tying).

**Topological nature**:
- Linear combinations of nodal coordinates
- Master-slave vertex relationships
- Coefficient matrices for geometric constraints

**Use cases**:
- Tying vertices together (merged meshes)
- Enforcing planarity or symmetry
- Coupling vertices for continuity

**Example**:
```cpp
// Tie vertex 5 to the centroid of vertices {1, 2, 3}
MultiPointConstraint mpc;
mpc.constrained_vertex = 5;
mpc.master_vertices = {1, 2, 3};
mpc.weights = {1.0/3.0, 1.0/3.0, 1.0/3.0};
```

### ✗ What This Folder DOES NOT Include

#### Physics-Specific Constraints
- ✗ **Contact mechanics**: Detecting/enforcing contact between surfaces
  - Belongs in: Physics solver (mechanics module)
  - Reason: Requires force computation, penetration checking, friction models

- ✗ **Friction constraints**: Tangential contact forces
  - Belongs in: Physics solver
  - Reason: Material-dependent, force-based

- ✗ **Incompressibility**: Volume preservation constraints
  - Belongs in: Physics solver (fluid/structure modules)
  - Reason: Field-specific (velocity, pressure)

#### FEM-Specific Constraints
- ✗ **DOF constraints**: Degree of freedom coupling
  - Belongs in: Solver DOF manager
  - Reason: Field-specific, not mesh-level

- ✗ **Boundary conditions**: Dirichlet, Neumann, Robin
  - Belongs in: Solver boundary condition module
  - Reason: Physics and field-specific

- ✗ **Constraint enforcement**: Lagrange multipliers, penalty methods
  - Belongs in: Solver constraint enforcement module
  - Reason: Numerical method, not mesh topology

#### Solver-Level Operations
- ✗ **Constraint matrix assembly**: Building constraint equations
  - Belongs in: Solver linear algebra module
  - Reason: Operates on solution vectors, not mesh

- ✗ **Projection operators**: Mortar projection, weighted residuals
  - Belongs in: Solver integration module
  - Reason: Requires basis functions, quadrature

## Key Classes (To Be Implemented)

### `HangingVertexConstraints`
Manages hanging vertex relationships for adaptive refinement.

**Responsibilities**:
- Detect hanging vertices from mesh refinement
- Store parent-child relationships
- Compute interpolation weights
- Query constraints for specific vertices/edges/faces

**Key methods**:
```cpp
class HangingVertexConstraints {
public:
    // Detect hanging vertices from refined mesh
    void detect_hanging_vertices(const MeshBase& mesh);

    // Check if vertex is hanging
    bool is_hanging(index_t vertex) const;

    // Get parent entity and interpolation weights
    HangingVertexConstraint get_constraint(index_t vertex) const;

    // Get all hanging vertices
    const std::vector<index_t>& hanging_vertices() const;
};
```

### `PeriodicConstraints`
Manages periodic vertex/face pairs for periodic domains.

**Responsibilities**:
- Identify periodic entity pairs
- Store transformation/translation between pairs
- Validate periodicity (matching geometry)
- Support rotational periodicity

**Key methods**:
```cpp
class PeriodicConstraints {
public:
    // Add periodic pair with translation
    void add_periodic_pair(EntityKind kind, index_t primary, index_t secondary,
                          const std::array<real_t,3>& translation);

    // Check if entity has periodic pair
    bool is_periodic(EntityKind kind, index_t id) const;

    // Get periodic partner
    index_t get_periodic_pair(EntityKind kind, index_t id) const;

    // Validate periodic geometry
    bool validate_periodicity(const MeshBase& mesh, real_t tolerance) const;
};
```

### `MortarInterface`
Manages topology of non-conforming mesh interfaces.

**Responsibilities**:
- Store master/slave surface pairs
- Compute geometric overlaps
- Identify interface vertices/faces
- Provide interface topology queries

**Key methods**:
```cpp
class MortarInterface {
public:
    // Define mortar interface
    void set_master_surface(const std::vector<index_t>& faces);
    void set_slave_surface(const std::vector<index_t>& faces);

    // Compute geometric overlaps (topology only)
    void compute_overlaps(const MeshBase& mesh);

    // Query interface topology
    const std::vector<index_t>& master_faces() const;
    const std::vector<index_t>& slave_faces() const;

    // Get vertices on each side
    std::unordered_set<index_t> master_vertices() const;
    std::unordered_set<index_t> slave_vertices() const;
};
```

### `MultiPointConstraints`
Manages geometric multi-point constraints between vertices.

**Responsibilities**:
- Store linear constraint relationships
- Validate constraint coefficients
- Query constraints for specific vertices
- Detect constraint conflicts

**Key methods**:
```cpp
class MultiPointConstraints {
public:
    // Add constraint: constrained_vertex = sum(weights[i] * master_vertices[i])
    void add_constraint(index_t constrained_vertex,
                       const std::vector<index_t>& master_vertices,
                       const std::vector<real_t>& weights);

    // Check if vertex is constrained
    bool is_constrained(index_t vertex) const;

    // Get constraint for vertex
    MultiPointConstraint get_constraint(index_t vertex) const;

    // Validate constraints (no circular dependencies)
    bool validate() const;
};
```

## Interaction with Other Folders

### Topology
- **Relationship**: Constraints use topological connectivity
- **Direction**: Constraints query Topology for parent entities, adjacency

### Adaptivity
- **Relationship**: Refinement creates hanging vertices
- **Direction**: Adaptivity notifies Constraints of new hanging vertices

### Geometry
- **Relationship**: Constraints use geometric queries for overlap detection
- **Direction**: Constraints query Geometry for distances, projections

### Fields
- **Relationship**: Constraints inform field interpolation
- **Direction**: Fields query Constraints for hanging vertex weights

### Boundary
- **Relationship**: Periodic constraints identified on boundary
- **Direction**: Constraints use Boundary detection for surface identification

### Solver (External)
- **Relationship**: Solver uses constraint topology to build enforcement equations
- **Direction**: Solver queries Constraints for relationships, then enforces them

## Design Principles

### 1. Separation of Identification and Enforcement
**Mesh folder**: Identifies which entities are constrained and their relationships
**Solver folder**: Enforces constraints through equations, matrices, etc.

### 2. Geometry-Agnostic Storage
Constraints store topology (vertex IDs, weights) independent of coordinate values. Geometric validation is separate.

### 3. Observer Pattern Integration
Constraint changes should trigger mesh events for dependent systems to update.

### 4. Validation Before Storage
Constraints should validate relationships (e.g., parent entities exist) before accepting them.

## Usage Examples

### Example 1: Hanging Vertex Detection After Refinement

```cpp
#include "Constraints/HangingVertexConstraints.h"
#include "Adaptivity/MeshAdaptivity.h"

// Refine mesh
MeshAdaptivity adaptivity(mesh);
adaptivity.refine_cells({10, 15, 20});

// Detect hanging vertices
HangingVertexConstraints hnc;
hnc.detect_hanging_vertices(mesh);

// Query hanging vertices
for (index_t vertex : hnc.hanging_vertices()) {
    auto constraint = hnc.get_constraint(vertex);
    std::cout << "Vertex " << vertex << " hangs on "
              << constraint.parent_entity << " " << constraint.parent_id << std::endl;
}
```

### Example 2: Periodic Boundary Setup

```cpp
#include "Constraints/PeriodicConstraints.h"
#include "Boundary/BoundaryDetector.h"

// Detect boundaries
BoundaryDetector detector(mesh);
auto info = detector.detect_boundary();

// Identify periodic pairs (user-provided or automatic detection)
PeriodicConstraints pc;

// Add periodic vertex pairs (e.g., left and right boundaries)
for (size_t i = 0; i < left_vertices.size(); ++i) {
    pc.add_periodic_pair(EntityKind::Vertex,
                        left_vertices[i], right_vertices[i],
                        {1.0, 0.0, 0.0});  // Translation vector
}

// Validate geometry matches
if (!pc.validate_periodicity(mesh, 1e-10)) {
    std::cerr << "Warning: Periodic geometry mismatch detected!" << std::endl;
}
```

### Example 3: Non-Conforming Interface

```cpp
#include "Constraints/MortarInterface.h"

// Define mortar interface between two mesh blocks
MortarInterface interface;
interface.set_master_surface(block1_boundary_faces);
interface.set_slave_surface(block2_boundary_faces);

// Compute geometric overlaps (topology only)
interface.compute_overlaps(mesh);

// Get interface topology for solver
auto master_vertices = interface.master_vertices();
auto slave_vertices = interface.slave_vertices();

// Solver uses this topology to build projection operators (not here!)
```

## Future Extensions

- **Constraint graph analysis**: Detect circular dependencies
- **Automatic periodic detection**: Match faces by geometry
- **Constraint visualization**: Export for ParaView/VisIt
- **Distributed constraints**: MPI-aware constraint handling
- **Constraint optimization**: Minimize number of constraint equations

## Testing

Unit tests should verify:
- Hanging vertex detection accuracy
- Periodic pair identification
- Geometric overlap computation
- Constraint validation (no circular dependencies)
- Edge cases (no constraints, fully constrained)

See `Tests/Unit/Constraints/` for test implementations.
