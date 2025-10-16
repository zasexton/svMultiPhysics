# Mesh Constraints

This subfolder implements **topological and geometric constraints** for mesh entities.

## Purpose

The Constraints folder provides mechanisms to represent and manage special relationships between mesh entities that arise from:

1. **Adaptive mesh refinement** (hanging nodes)
2. **Periodic boundary conditions** (topologically equivalent entities)
3. **Non-conforming mesh interfaces** (mortar methods)
4. **Multi-point geometric constraints** (node tying)

**Key principle**: This folder handles **topology and geometry only**. It does NOT implement:
- Physics-specific constraints (contact, friction, etc.)
- FEM-specific constraints (DOF coupling, boundary conditions)
- Constraint enforcement (that belongs in the Solver)

## Scope

### ✓ What This Folder DOES Include

#### 1. Hanging Node Constraints
**Purpose**: Track parent-child relationships for nodes created during adaptive refinement.

**Topological nature**:
- Which nodes are "hanging" (dependent on parent entities)
- Which edges/faces contain hanging nodes
- Parent-child connectivity for constraint equations

**Use cases**:
- Adaptive mesh refinement (h-refinement)
- Mesh coarsening
- Transition between refinement levels

**Example**:
```cpp
// A hanging node on an edge depends on the edge's endpoint nodes
HangingNodeConstraint hnc;
hnc.constrained_node = 5;
hnc.parent_entity = EntityKind::Edge;
hnc.parent_id = 12;
hnc.parent_nodes = {3, 4};        // Endpoints of edge 12
hnc.weights = {0.5, 0.5};         // Linear interpolation
```

#### 2. Periodic Constraints
**Purpose**: Identify topologically equivalent entities on opposite periodic boundaries.

**Topological nature**:
- Which node pairs are periodic equivalents
- Which face pairs match on opposite boundaries
- Transformation/translation vectors between periodic surfaces

**Use cases**:
- Periodic domains (unit cells, turbomachinery)
- Symmetry exploitation
- Mesh topology verification

**Example**:
```cpp
// Nodes 10 and 20 are periodic equivalents
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
- Which faces/nodes belong to each side of the interface
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
**Purpose**: Enforce geometric relationships between nodes (e.g., node tying).

**Topological nature**:
- Linear combinations of nodal coordinates
- Master-slave node relationships
- Coefficient matrices for geometric constraints

**Use cases**:
- Tying nodes together (merged meshes)
- Enforcing planarity or symmetry
- Coupling nodes for continuity

**Example**:
```cpp
// Tie node 5 to the centroid of nodes {1, 2, 3}
MultiPointConstraint mpc;
mpc.constrained_node = 5;
mpc.master_nodes = {1, 2, 3};
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

### `HangingNodeConstraints`
Manages hanging node relationships for adaptive refinement.

**Responsibilities**:
- Detect hanging nodes from mesh refinement
- Store parent-child relationships
- Compute interpolation weights
- Query constraints for specific nodes/edges/faces

**Key methods**:
```cpp
class HangingNodeConstraints {
public:
    // Detect hanging nodes from refined mesh
    void detect_hanging_nodes(const MeshBase& mesh);

    // Check if node is hanging
    bool is_hanging(index_t node) const;

    // Get parent entity and interpolation weights
    HangingNodeConstraint get_constraint(index_t node) const;

    // Get all hanging nodes
    const std::vector<index_t>& hanging_nodes() const;
};
```

### `PeriodicConstraints`
Manages periodic node/face pairs for periodic domains.

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
- Identify interface nodes/faces
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

    // Get nodes on each side
    std::unordered_set<index_t> master_nodes() const;
    std::unordered_set<index_t> slave_nodes() const;
};
```

### `MultiPointConstraints`
Manages geometric multi-point constraints between nodes.

**Responsibilities**:
- Store linear constraint relationships
- Validate constraint coefficients
- Query constraints for specific nodes
- Detect constraint conflicts

**Key methods**:
```cpp
class MultiPointConstraints {
public:
    // Add constraint: constrained_node = sum(weights[i] * master_nodes[i])
    void add_constraint(index_t constrained_node,
                       const std::vector<index_t>& master_nodes,
                       const std::vector<real_t>& weights);

    // Check if node is constrained
    bool is_constrained(index_t node) const;

    // Get constraint for node
    MultiPointConstraint get_constraint(index_t node) const;

    // Validate constraints (no circular dependencies)
    bool validate() const;
};
```

## Interaction with Other Folders

### Topology
- **Relationship**: Constraints use topological connectivity
- **Direction**: Constraints query Topology for parent entities, adjacency

### Adaptivity
- **Relationship**: Refinement creates hanging nodes
- **Direction**: Adaptivity notifies Constraints of new hanging nodes

### Geometry
- **Relationship**: Constraints use geometric queries for overlap detection
- **Direction**: Constraints query Geometry for distances, projections

### Fields
- **Relationship**: Constraints inform field interpolation
- **Direction**: Fields query Constraints for hanging node weights

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
Constraints store topology (node IDs, weights) independent of coordinate values. Geometric validation is separate.

### 3. Observer Pattern Integration
Constraint changes should trigger mesh events for dependent systems to update.

### 4. Validation Before Storage
Constraints should validate relationships (e.g., parent entities exist) before accepting them.

## Usage Examples

### Example 1: Hanging Node Detection After Refinement

```cpp
#include "Constraints/HangingNodeConstraints.h"
#include "Adaptivity/MeshAdaptivity.h"

// Refine mesh
MeshAdaptivity adaptivity(mesh);
adaptivity.refine_cells({10, 15, 20});

// Detect hanging nodes
HangingNodeConstraints hnc;
hnc.detect_hanging_nodes(mesh);

// Query hanging nodes
for (index_t node : hnc.hanging_nodes()) {
    auto constraint = hnc.get_constraint(node);
    std::cout << "Node " << node << " hangs on "
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

// Add periodic node pairs (e.g., left and right boundaries)
for (size_t i = 0; i < left_nodes.size(); ++i) {
    pc.add_periodic_pair(EntityKind::Vertex,
                        left_nodes[i], right_nodes[i],
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
auto master_nodes = interface.master_nodes();
auto slave_nodes = interface.slave_nodes();

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
- Hanging node detection accuracy
- Periodic pair identification
- Geometric overlap computation
- Constraint validation (no circular dependencies)
- Edge cases (no constraints, fully constrained)

See `Tests/Unit/Constraints/` for test implementations.
