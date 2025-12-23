/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_HANGING_VERTEX_CONSTRAINTS_H
#define SVMP_HANGING_VERTEX_CONSTRAINTS_H

#include "../Core/MeshTypes.h"
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace svmp {

// Forward declarations
class MeshBase;

/**
 * @brief Entity kind for constraint parent
 */
enum class ConstraintParentType {
  Edge,     ///< Constraint from edge midpoint (2D and 3D)
  Face,     ///< Constraint from face centroid (3D only)
  Invalid   ///< Invalid parent type
};

/**
 * @brief Single hanging vertex constraint information
 *
 * Stores the relationship between a hanging vertex and its parent entity.
 * The hanging vertex's value is constrained to be an interpolation of
 * the parent entity's vertices.
 */
struct HangingVertexConstraint {
  /** The constrained (hanging) vertex ID */
  index_t constrained_vertex;

  /** Type of parent entity (edge or face) */
  ConstraintParentType parent_type;

  /** Parent entity vertices (2 for edge, 3-4 for face) */
  std::vector<index_t> parent_vertices;

  /** Interpolation weights for parent vertices (sum to 1.0) */
  std::vector<real_t> weights;

  /** Refinement level of the hanging vertex */
  size_t refinement_level;

  /** Cells that share this hanging vertex */
  std::set<index_t> adjacent_cells;

  /** Constructor */
  HangingVertexConstraint() :
    constrained_vertex(-1),
    parent_type(ConstraintParentType::Invalid),
    refinement_level(0) {}

  /** Check if constraint is valid */
  bool is_valid() const {
    return constrained_vertex >= 0 &&
           parent_type != ConstraintParentType::Invalid &&
           parent_vertices.size() >= 2 &&
           weights.size() == parent_vertices.size();
  }

  /** Get dimension of constraint (2 for edge, 3 for face) */
  size_t dimension() const {
    return parent_type == ConstraintParentType::Edge ? 2 : 3;
  }
};

/**
 * @brief Container for all hanging vertex constraints in a mesh
 *
 * This class manages the detection, storage, and querying of hanging vertex
 * constraints that arise from non-conforming mesh refinement. It supports
 * both 2D (edge hanging) and 3D (edge and face hanging) constraints.
 */
class HangingVertexConstraints {
public:
  /** Default constructor */
  HangingVertexConstraints();

  /** Destructor */
  ~HangingVertexConstraints();

  /**
   * @brief Detect all hanging vertices in the mesh
   *
   * Analyzes the mesh topology to identify hanging vertices created
   * by non-conforming refinement. This includes:
   * - Edge midpoints that exist on only one side of an edge
   * - Face centroids that exist on only one side of a face (3D)
   *
   * @param mesh The mesh to analyze
   * @param refinement_levels Optional per-cell refinement levels
   */
  void detect_hanging_vertices(const MeshBase& mesh,
                               const std::vector<size_t>* refinement_levels = nullptr);

  /**
   * @brief Add a hanging vertex constraint manually
   *
   * @param constraint The constraint to add
   * @return true if added successfully, false if invalid or duplicate
   */
  bool add_constraint(const HangingVertexConstraint& constraint);

  /**
   * @brief Remove a hanging vertex constraint
   *
   * @param vertex_id The hanging vertex to unconstrain
   * @return true if removed, false if not found
   */
  bool remove_constraint(index_t vertex_id);

  /**
   * @brief Clear all constraints
   */
  void clear();

  // Query methods

  /**
   * @brief Check if a vertex is hanging
   *
   * @param vertex_id The vertex to check
   * @return true if the vertex is hanging (constrained)
   */
  bool is_hanging(index_t vertex_id) const;

  /**
   * @brief Get the constraint for a hanging vertex
   *
   * @param vertex_id The hanging vertex
   * @return The constraint, or invalid constraint if not hanging
   */
  HangingVertexConstraint get_constraint(index_t vertex_id) const;

  /**
   * @brief Get all hanging vertices
   *
   * @return Vector of hanging vertex IDs
   */
  std::vector<index_t> get_hanging_vertices() const;

  /**
   * @brief Get all constraints
   *
   * @return Map from vertex ID to constraint
   */
  const std::unordered_map<index_t, HangingVertexConstraint>& get_all_constraints() const {
    return constraints_;
  }

  /**
   * @brief Get number of hanging vertices
   */
  size_t num_hanging_vertices() const {
    return constraints_.size();
  }

  /**
   * @brief Get constraints by parent type
   *
   * @param parent_type The type of parent entity
   * @return Vector of constraints with given parent type
   */
  std::vector<HangingVertexConstraint> get_constraints_by_type(
      ConstraintParentType parent_type) const;

  /**
   * @brief Get hanging vertices on a specific edge
   *
   * @param v1 First edge vertex
   * @param v2 Second edge vertex
   * @return Vector of hanging vertex IDs on this edge
   */
  std::vector<index_t> get_edge_hanging_vertices(index_t v1, index_t v2) const;

  /**
   * @brief Get hanging vertices on a specific face
   *
   * @param face_vertices Vertices defining the face (3 or 4)
   * @return Vector of hanging vertex IDs on this face
   */
  std::vector<index_t> get_face_hanging_vertices(
      const std::vector<index_t>& face_vertices) const;

  /**
   * @brief Check if an edge has hanging vertices
   *
   * @param v1 First edge vertex
   * @param v2 Second edge vertex
   * @return true if edge has hanging vertices
   */
  bool edge_has_hanging(index_t v1, index_t v2) const;

  /**
   * @brief Check if a face has hanging vertices
   *
   * @param face_vertices Vertices defining the face
   * @return true if face has hanging vertices
   */
  bool face_has_hanging(const std::vector<index_t>& face_vertices) const;

  // Constraint generation methods

  /**
   * @brief Generate constraint matrix for FEM assembly
   *
   * Returns a map where each entry represents one constraint equation:
   * constrained_vertex = sum(weight_i * parent_vertex_i)
   *
   * @return Map from constrained vertex to (parent vertex -> weight) map
   */
  std::map<index_t, std::map<index_t, real_t>> generate_constraint_matrix() const;

  /**
   * @brief Apply constraints to a solution vector
   *
   * Enforces hanging vertex constraints by setting hanging vertex values
   * to the weighted average of their parent vertices.
   *
   * @param solution Solution vector to constrain (modified in place)
   * @param num_components Number of components per vertex (e.g., 3 for 3D vector)
   */
  void apply_constraints(std::vector<real_t>& solution, size_t num_components = 1) const;

  /**
   * @brief Validate all constraints
   *
   * Checks that:
   * - All parent vertices exist in the mesh
   * - Weights sum to 1.0 (within tolerance)
   * - No circular dependencies exist
   * - Parent entities are topologically valid
   *
   * @param mesh The mesh to validate against
   * @param tolerance Numerical tolerance for weight sum
   * @return true if all constraints are valid
   */
  bool validate(const MeshBase& mesh, real_t tolerance = 1e-10) const;

  /**
   * @brief Get statistics about constraints
   */
  struct Statistics {
    size_t num_edge_hanging;     ///< Number of edge-hanging vertices
    size_t num_face_hanging;     ///< Number of face-hanging vertices
    size_t max_refinement_level; ///< Maximum refinement level
    size_t num_affected_cells;   ///< Number of cells with hanging vertices

    Statistics() : num_edge_hanging(0), num_face_hanging(0),
                  max_refinement_level(0), num_affected_cells(0) {}
  };

  /**
   * @brief Compute constraint statistics
   */
  Statistics compute_statistics() const;

  /**
   * @brief Export constraints for visualization
   *
   * Creates fields that can be visualized to show hanging vertices
   * and their constraint relationships.
   *
   * @param mesh The mesh
   * @param field_name Base name for constraint fields
   */
  void export_to_fields(MeshBase& mesh, const std::string& field_name = "hanging_constraint") const;

private:
  // Core data storage

  /** Map from hanging vertex ID to its constraint */
  std::unordered_map<index_t, HangingVertexConstraint> constraints_;

  /** Map from edge (as sorted pair) to hanging vertices on that edge */
  std::map<std::pair<index_t, index_t>, std::vector<index_t>> edge_hanging_map_;

  /** Set of cells that have hanging vertices */
  std::unordered_set<index_t> affected_cells_;

  // Helper methods for detection

  /**
   * @brief Detect hanging vertices on edges
   */
  void detect_edge_hanging(const MeshBase& mesh,
                           const std::vector<size_t>* refinement_levels);

  /**
   * @brief Detect hanging vertices on faces (3D only)
   */
  void detect_face_hanging(const MeshBase& mesh,
                           const std::vector<size_t>* refinement_levels);

  /**
   * @brief Check if a vertex is at the midpoint of an edge
   */
  bool is_edge_midpoint(const MeshBase& mesh,
                       index_t vertex,
                       index_t v1, index_t v2,
                       real_t tolerance = 1e-10) const;

  /**
   * @brief Check if a vertex is at the centroid of a face
   */
  bool is_face_centroid(const MeshBase& mesh,
                       index_t vertex,
                       const std::vector<index_t>& face_vertices,
                       real_t tolerance = 1e-10) const;

  /**
   * @brief Compute interpolation weights for edge constraint
   */
  std::vector<real_t> compute_edge_weights(const MeshBase& mesh,
                                          index_t hanging_vertex,
                                          index_t v1, index_t v2) const;

  /**
   * @brief Compute interpolation weights for face constraint
   */
  std::vector<real_t> compute_face_weights(const MeshBase& mesh,
                                          index_t hanging_vertex,
                                          const std::vector<index_t>& face_vertices) const;

  /**
   * @brief Update internal maps after adding/removing constraints
   */
  void update_maps();

  /**
   * @brief Sort edge vertices to create canonical edge representation
   */
  std::pair<index_t, index_t> make_edge(index_t v1, index_t v2) const {
    return v1 < v2 ? std::make_pair(v1, v2) : std::make_pair(v2, v1);
  }
};

/**
 * @brief Information about a hanging vertex location
 */
struct HangingVertexInfo {
  /** The vertex ID that is hanging */
  index_t vertex_id = -1;

  /** IDs of the coarser neighbor cells */
  std::vector<index_t> coarse_neighbors;

  /** Parent entity type (edge or face) */
  ConstraintParentType parent_type = ConstraintParentType::Invalid;

  /** Parent entity vertices */
  std::vector<index_t> parent_vertices;

  /** Interpolation weights for parent vertices */
  std::vector<real_t> weights;

  /** Refinement level difference between fine and coarse cells */
  size_t level_difference = 0;
};

/**
 * @brief Statistics from hanging vertex detection
 */
struct HangingVertexStats {
  /** Number of hanging vertices detected */
  size_t num_hanging = 0;

  /** Number of hanging vertices on edges */
  size_t num_edge_hanging = 0;

  /** Number of hanging vertices on faces */
  size_t num_face_hanging = 0;

  /** Maximum refinement level difference */
  size_t max_level_difference = 0;

  /** Number of cells needing additional refinement for 2:1 balance */
  size_t cells_marked_for_balance = 0;
};

/**
 * @brief Utility functions for hanging vertex constraints
 */
class HangingVertexUtils {
public:
  /**
   * @brief Check if refinement will create hanging vertices
   *
   * @param mesh The mesh
   * @param cells_to_refine Cells marked for refinement
   * @return true if refinement will create hanging vertices
   */
  static bool will_create_hanging(const MeshBase& mesh,
                                  const std::set<index_t>& cells_to_refine);

  /**
   * @brief Find cells that need refinement to remove hanging vertices
   *
   * @param mesh The mesh
   * @param constraints Current hanging vertex constraints
   * @return Set of cell IDs that should be refined for conformity
   */
  static std::set<index_t> find_closure_cells(const MeshBase& mesh,
                                              const HangingVertexConstraints& constraints);

  /**
   * @brief Compute maximum level difference across mesh interfaces
   *
   * @param mesh The mesh
   * @param refinement_levels Per-cell refinement levels
   * @return Maximum level difference between adjacent cells
   */
  static size_t compute_max_level_difference(const MeshBase& mesh,
                                            const std::vector<size_t>& refinement_levels);

  /**
   * @brief Check if hanging vertex pattern is valid
   *
   * Valid patterns have at most 1 level difference between adjacent cells.
   *
   * @param mesh The mesh
   * @param constraints The constraints to check
   * @param refinement_levels Per-cell refinement levels
   * @return true if pattern is valid
   */
  static bool is_valid_hanging_pattern(const MeshBase& mesh,
                                       const HangingVertexConstraints& constraints,
                                       const std::vector<size_t>& refinement_levels);

  /**
   * @brief Detect all hanging vertices in the mesh
   *
   * Analyzes the mesh topology to identify vertices that lie on
   * edges or faces of coarser neighboring cells. This is the core
   * detection algorithm for non-conforming interfaces.
   *
   * @param mesh The mesh to analyze
   * @param refined_cells Set of cells that have been refined
   * @param refinement_levels Per-cell refinement levels (optional)
   * @return Map from hanging vertex ID to its information
   */
  static std::unordered_map<index_t, HangingVertexInfo>
  detect_hanging_vertices(const MeshBase& mesh,
                         const std::unordered_set<index_t>& refined_cells,
                         const std::vector<size_t>* refinement_levels = nullptr);

  /**
   * @brief Generate constraints for hanging vertices
   *
   * Creates interpolation constraints that enforce continuity
   * at non-conforming interfaces. The constraints express hanging
   * vertex values as linear combinations of parent entity vertices.
   *
   * @param mesh The mesh
   * @param hanging_vertices Map of hanging vertices from detect_hanging_vertices
   * @return List of constraints ready for HangingVertexConstraints
   */
  static std::vector<HangingVertexConstraint>
  generate_constraints(const MeshBase& mesh,
                      const std::unordered_map<index_t, HangingVertexInfo>& hanging_vertices);

  /**
   * @brief Enforce 2:1 balance in the mesh
   *
   * Identifies additional cells that must be refined to maintain
   * the 2:1 balance criterion (refinement level difference <= 1
   * between neighbors). This prevents excessive hanging vertices
   * and ensures mesh quality.
   *
   * @param mesh The mesh
   * @param marked_cells Initially marked cells for refinement
   * @param refinement_levels Current per-cell refinement levels
   * @param max_iterations Maximum propagation iterations (default 10)
   * @return Additional cells that must be refined for balance
   */
  static std::unordered_set<index_t>
  enforce_2to1_balance(const MeshBase& mesh,
                      const std::unordered_set<index_t>& marked_cells,
                      const std::vector<size_t>& refinement_levels,
                      size_t max_iterations = 10);

  /**
   * @brief Check if a vertex is hanging
   *
   * Quick check to determine if a vertex is hanging based on
   * its connectivity and neighbor refinement levels.
   *
   * @param mesh The mesh
   * @param vertex_id Vertex to check
   * @param refined_cells Set of refined cells
   * @return true if vertex is hanging
   */
  static bool is_hanging_vertex(const MeshBase& mesh,
                               index_t vertex_id,
                               const std::unordered_set<index_t>& refined_cells);

  /**
   * @brief Compute statistics about hanging vertices
   *
   * @param hanging_vertices Map of hanging vertices
   * @return Statistics structure
   */
  static HangingVertexStats
  compute_statistics(const std::unordered_map<index_t, HangingVertexInfo>& hanging_vertices);

  /**
   * @brief Validate 2:1 balance in the mesh
   *
   * Checks if the mesh satisfies 2:1 balance criterion.
   *
   * @param mesh The mesh
   * @param refinement_levels Per-cell refinement levels
   * @param violations Output: cells violating 2:1 balance
   * @return true if mesh is balanced
   */
  static bool validate_balance(const MeshBase& mesh,
                              const std::vector<size_t>& refinement_levels,
                              std::vector<index_t>* violations = nullptr);

private:
  /**
   * @brief Check if vertex lies on edge between two vertices
   *
   * @param mesh The mesh
   * @param vertex Potential hanging vertex
   * @param v1 First edge endpoint
   * @param v2 Second edge endpoint
   * @param tolerance Geometric tolerance
   * @return true if vertex is on edge
   */
  static bool vertex_on_edge(const MeshBase& mesh,
                            index_t vertex,
                            index_t v1,
                            index_t v2,
                            real_t tolerance = 1e-10);

  /**
   * @brief Check if vertex lies on face defined by vertices
   *
   * @param mesh The mesh
   * @param vertex Potential hanging vertex
   * @param face_vertices Vertices defining the face
   * @param tolerance Geometric tolerance
   * @return true if vertex is on face
   */
  static bool vertex_on_face(const MeshBase& mesh,
                            index_t vertex,
                            const std::vector<index_t>& face_vertices,
                            real_t tolerance = 1e-10);

  /**
   * @brief Compute barycentric coordinates for point in face
   *
   * @param mesh The mesh
   * @param point_id Point vertex ID
   * @param face_vertices Face vertex IDs
   * @param weights Output: barycentric coordinates
   * @return true if point is in face
   */
  static bool compute_barycentric_weights(const MeshBase& mesh,
                                         index_t point_id,
                                         const std::vector<index_t>& face_vertices,
                                         std::vector<real_t>& weights);

  /**
   * @brief Get all edges of a cell
   *
   * @param mesh The mesh
   * @param cell_id Cell ID
   * @return List of edge vertex pairs
   */
  static std::vector<std::pair<index_t, index_t>>
  get_cell_edges(const MeshBase& mesh, index_t cell_id);

  /**
   * @brief Get all faces of a cell
   *
   * @param mesh The mesh
   * @param cell_id Cell ID
   * @return List of face vertex lists
   */
  static std::vector<std::vector<index_t>>
  get_cell_faces(const MeshBase& mesh, index_t cell_id);
};

} // namespace svmp

#endif // SVMP_HANGING_VERTEX_CONSTRAINTS_H