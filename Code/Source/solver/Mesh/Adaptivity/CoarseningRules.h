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

#ifndef SVMP_COARSENING_RULES_H
#define SVMP_COARSENING_RULES_H

#include "Options.h"
#include "RefinementRules.h"
#include <array>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace svmp {

// Forward declarations
class MeshBase;

/**
 * @brief Coarsening pattern types
 */
enum class CoarseningPattern {
  VERTEX_REMOVAL,      // Remove vertex and retriangulate
  EDGE_COLLAPSE,       // Collapse edge to point
  FACE_COLLAPSE,       // Collapse face to edge/point
  CELL_COLLAPSE,       // Collapse cell (3D)
  REVERSE_RED,         // Reverse of red refinement
  REVERSE_GREEN,       // Reverse of green refinement
  REVERSE_BISECTION,   // Reverse of bisection
  AGGLOMERATION,       // Merge multiple elements
  NONE                 // No coarsening
};

/**
 * @brief Information about a coarsening operation
 */
struct CoarseningOperation {
  /** Type of coarsening pattern */
  CoarseningPattern pattern;

  /** Elements to be merged/removed */
  std::vector<size_t> source_elements;

  /** Resulting parent element */
  size_t target_element;

  /** Vertices to be removed */
  std::set<size_t> removed_vertices;

  /** Edges to be collapsed */
  std::vector<std::pair<size_t, size_t>> collapsed_edges;

  /** New connectivity after coarsening */
  std::vector<size_t> new_connectivity;

  /** Quality metric after coarsening */
  double predicted_quality;

  /** Is this operation valid */
  bool valid;

  /** Priority for this operation */
  double priority;
};

/**
 * @brief Coarsening history for undoing operations
 */
struct CoarseningHistory {
  /** Original element IDs */
  std::vector<size_t> original_elements;

  /** Original connectivity */
  std::vector<std::vector<size_t>> original_connectivity;

  /** Original vertex positions */
  std::map<size_t, std::array<double, 3>> original_positions;

  /** Pattern used for coarsening */
  CoarseningPattern pattern;

  /** Timestamp of operation */
  size_t operation_id;
};

/**
 * @brief Abstract base class for coarsening rules
 */
class CoarseningRule {
public:
  virtual ~CoarseningRule() = default;

  /**
   * @brief Check if elements can be coarsened
   *
   * @param mesh The mesh
   * @param elements Elements to check
   * @return True if coarsening is possible
   */
  virtual bool can_coarsen(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const = 0;

  /**
   * @brief Determine coarsening operation for elements
   *
   * @param mesh The mesh
   * @param elements Elements to coarsen
   * @return Coarsening operation details
   */
  virtual CoarseningOperation determine_coarsening(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const = 0;

  /**
   * @brief Apply coarsening operation
   *
   * @param mesh The mesh (modified)
   * @param operation Coarsening operation to apply
   * @return History for potential undo
   */
  virtual CoarseningHistory apply_coarsening(
      MeshBase& mesh,
      const CoarseningOperation& operation) const = 0;

  /**
   * @brief Undo coarsening operation
   *
   * @param mesh The mesh (modified)
   * @param history Coarsening history
   */
  virtual void undo_coarsening(
      MeshBase& mesh,
      const CoarseningHistory& history) const = 0;

  /**
   * @brief Get rule name
   */
  virtual std::string name() const = 0;

  /**
   * @brief Check if rule applies to element type
   */
  virtual bool supports_element_type(ElementType type) const = 0;
};

/**
 * @brief Edge collapse coarsening rule
 *
 * Collapses edges to remove vertices and simplify mesh.
 */
class EdgeCollapseRule : public CoarseningRule {
public:
  /**
   * @brief Configuration for edge collapse
   */
  struct Config {
    /** Minimum edge length ratio for collapse */
    double min_edge_ratio = 0.1;

    /** Maximum aspect ratio after collapse */
    double max_aspect_ratio = 10.0;

    /** Preserve boundary edges */
    bool preserve_boundary = true;

    /** Preserve feature edges */
    bool preserve_features = true;

    /** Feature angle threshold (degrees) */
    double feature_angle = 30.0;

    /** Check for element inversion */
    bool check_inversion = true;

    /** Volume change tolerance */
    double volume_tolerance = 0.1;

    /** Collapse to edge midpoint */
    bool collapse_to_midpoint = true;
  };

  explicit EdgeCollapseRule(const Config& config = {});

  bool can_coarsen(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const override;

  CoarseningOperation determine_coarsening(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const override;

  CoarseningHistory apply_coarsening(
      MeshBase& mesh,
      const CoarseningOperation& operation) const override;

  void undo_coarsening(
      MeshBase& mesh,
      const CoarseningHistory& history) const override;

  std::string name() const override { return "EdgeCollapse"; }

  bool supports_element_type(ElementType type) const override;

private:
  Config config_;

  /** Find collapsible edge in elements */
  std::pair<size_t, size_t> find_collapsible_edge(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const;

  /** Check if edge can be collapsed */
  bool can_collapse_edge(
      const MeshBase& mesh,
      size_t v1, size_t v2) const;

  /** Compute collapse point */
  std::array<double, 3> compute_collapse_point(
      const MeshBase& mesh,
      size_t v1, size_t v2) const;

  /** Check quality after collapse */
  bool check_collapse_quality(
      const MeshBase& mesh,
      size_t v1, size_t v2,
      const std::array<double, 3>& collapse_point) const;
};

/**
 * @brief Vertex removal coarsening rule
 *
 * Removes vertices and retriangulates cavity.
 */
class VertexRemovalRule : public CoarseningRule {
public:
  /**
   * @brief Configuration for vertex removal
   */
  struct Config {
    /** Maximum valence for removable vertex */
    size_t max_valence = 8;

    /** Preserve boundary vertices */
    bool preserve_boundary = true;

    /** Minimum quality after removal */
    double min_quality = 0.3;

    /** Retriangulation method */
    enum class RetriangulationMethod {
      DELAUNAY,      // Delaunay triangulation
      ADVANCING_FRONT, // Advancing front
      EAR_CLIPPING,  // Ear clipping algorithm
      OPTIMAL        // Optimal triangulation
    };

    RetriangulationMethod method = RetriangulationMethod::DELAUNAY;
  };

  explicit VertexRemovalRule(const Config& config = {});

  bool can_coarsen(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const override;

  CoarseningOperation determine_coarsening(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const override;

  CoarseningHistory apply_coarsening(
      MeshBase& mesh,
      const CoarseningOperation& operation) const override;

  void undo_coarsening(
      MeshBase& mesh,
      const CoarseningHistory& history) const override;

  std::string name() const override { return "VertexRemoval"; }

  bool supports_element_type(ElementType type) const override;

private:
  Config config_;

  /** Find removable vertex */
  size_t find_removable_vertex(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const;

  /** Retriangulate cavity after vertex removal */
  std::vector<std::vector<size_t>> retriangulate_cavity(
      const MeshBase& mesh,
      const std::vector<size_t>& boundary_vertices) const;

  /** Check if vertex can be removed */
  bool can_remove_vertex(
      const MeshBase& mesh,
      size_t vertex) const;
};

/**
 * @brief Reverse refinement coarsening rule
 *
 * Reverses previous refinement operations.
 */
class ReverseRefinementRule : public CoarseningRule {
public:
  /**
   * @brief Configuration for reverse refinement
   */
  struct Config {
    /** Check refinement history */
    bool require_history = true;

    /** Allow partial reversal */
    bool allow_partial = false;

    /** Minimum element group size for reversal */
    size_t min_group_size = 4;

    /** Check conformity after reversal */
    bool check_conformity = true;
  };

  explicit ReverseRefinementRule(const Config& config = {});

  bool can_coarsen(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const override;

  CoarseningOperation determine_coarsening(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const override;

  CoarseningHistory apply_coarsening(
      MeshBase& mesh,
      const CoarseningOperation& operation) const override;

  void undo_coarsening(
      MeshBase& mesh,
      const CoarseningHistory& history) const override;

  std::string name() const override { return "ReverseRefinement"; }

  bool supports_element_type(ElementType type) const override;

private:
  Config config_;

  /** Check if elements are siblings from refinement */
  bool are_refinement_siblings(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const;

  /** Get parent element from refinement history */
  size_t get_parent_element(
      const MeshBase& mesh,
      const std::vector<size_t>& children) const;

  /** Determine original refinement pattern */
  RefinementPattern determine_original_pattern(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const;
};

/**
 * @brief Element agglomeration coarsening rule
 *
 * Merges multiple elements into larger elements.
 */
class AgglomerationRule : public CoarseningRule {
public:
  /**
   * @brief Configuration for agglomeration
   */
  struct Config {
    /** Maximum elements to agglomerate */
    size_t max_agglomeration = 8;

    /** Agglomeration strategy */
    enum class Strategy {
      GEOMETRIC,     // Based on geometric proximity
      TOPOLOGICAL,   // Based on connectivity
      QUALITY_BASED, // Maximize quality
      ISOTROPIC      // Maintain isotropy
    };

    Strategy strategy = Strategy::GEOMETRIC;

    /** Shape regularity threshold */
    double shape_regularity = 0.5;

    /** Preserve convexity */
    bool preserve_convexity = true;
  };

  explicit AgglomerationRule(const Config& config = {});

  bool can_coarsen(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const override;

  CoarseningOperation determine_coarsening(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const override;

  CoarseningHistory apply_coarsening(
      MeshBase& mesh,
      const CoarseningOperation& operation) const override;

  void undo_coarsening(
      MeshBase& mesh,
      const CoarseningHistory& history) const override;

  std::string name() const override { return "Agglomeration"; }

  bool supports_element_type(ElementType type) const override;

private:
  Config config_;

  /** Find agglomeration candidates */
  std::vector<std::vector<size_t>> find_agglomeration_groups(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const;

  /** Create agglomerated element */
  std::vector<size_t> create_agglomerated_connectivity(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const;

  /** Check agglomeration quality */
  double compute_agglomeration_quality(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const;
};

/**
 * @brief Composite coarsening rule
 *
 * Combines multiple coarsening strategies.
 */
class CompositeCoarseningRule : public CoarseningRule {
public:
  /**
   * @brief Add a coarsening rule with priority
   */
  void add_rule(std::unique_ptr<CoarseningRule> rule, double priority = 1.0);

  bool can_coarsen(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const override;

  CoarseningOperation determine_coarsening(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const override;

  CoarseningHistory apply_coarsening(
      MeshBase& mesh,
      const CoarseningOperation& operation) const override;

  void undo_coarsening(
      MeshBase& mesh,
      const CoarseningHistory& history) const override;

  std::string name() const override { return "Composite"; }

  bool supports_element_type(ElementType type) const override;

private:
  std::vector<std::pair<std::unique_ptr<CoarseningRule>, double>> rules_;

  /** Select best rule for elements */
  const CoarseningRule* select_best_rule(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const;
};

/**
 * @brief Coarsening manager
 *
 * Manages coarsening operations and history.
 */
class CoarseningManager {
public:
  /**
   * @brief Configuration for coarsening manager
   */
  struct Config {
    /** Enable history tracking */
    bool track_history = true;

    /** Maximum history size */
    size_t max_history = 1000;

    /** Batch coarsening operations */
    bool batch_operations = true;

    /** Quality threshold for coarsening */
    double min_quality = 0.3;

    /** Check mesh validity after each operation */
    bool validate_mesh = true;
  };

  explicit CoarseningManager(const Config& config = {});

  /**
   * @brief Perform mesh coarsening
   *
   * @param mesh The mesh to coarsen
   * @param marked_elements Elements marked for coarsening
   * @param options Adaptivity options
   * @return Number of coarsening operations performed
   */
  size_t coarsen_mesh(
      MeshBase& mesh,
      const std::vector<size_t>& marked_elements,
      const AdaptivityOptions& options);

  /**
   * @brief Undo last coarsening operation
   *
   * @param mesh The mesh
   * @return True if undo was successful
   */
  bool undo_last_operation(MeshBase& mesh);

  /**
   * @brief Clear coarsening history
   */
  void clear_history();

  /**
   * @brief Get coarsening statistics
   */
  struct Statistics {
    size_t total_operations = 0;
    size_t successful_operations = 0;
    size_t failed_operations = 0;
    std::map<CoarseningPattern, size_t> pattern_counts;
    double total_quality_improvement = 0.0;
  };

  Statistics get_statistics() const { return stats_; }

private:
  Config config_;
  std::vector<CoarseningHistory> history_;
  std::unique_ptr<CompositeCoarseningRule> rule_;
  Statistics stats_;
  size_t next_operation_id_ = 0;

  /** Initialize coarsening rules */
  void initialize_rules(const AdaptivityOptions& options);

  /** Group elements for coarsening */
  std::vector<std::vector<size_t>> group_elements_for_coarsening(
      const MeshBase& mesh,
      const std::vector<size_t>& marked_elements) const;

  /** Validate coarsening operation */
  bool validate_operation(
      const MeshBase& mesh,
      const CoarseningOperation& operation) const;

  /** Update mesh topology after coarsening */
  void update_topology(
      MeshBase& mesh,
      const CoarseningOperation& operation);
};

/**
 * @brief Factory for creating coarsening rules
 */
class CoarseningRuleFactory {
public:
  /**
   * @brief Create coarsening rule based on options
   */
  static std::unique_ptr<CoarseningRule> create(const AdaptivityOptions& options);

  /**
   * @brief Create edge collapse rule
   */
  static std::unique_ptr<CoarseningRule> create_edge_collapse(
      const EdgeCollapseRule::Config& config = {});

  /**
   * @brief Create vertex removal rule
   */
  static std::unique_ptr<CoarseningRule> create_vertex_removal(
      const VertexRemovalRule::Config& config = {});

  /**
   * @brief Create reverse refinement rule
   */
  static std::unique_ptr<CoarseningRule> create_reverse_refinement(
      const ReverseRefinementRule::Config& config = {});

  /**
   * @brief Create agglomeration rule
   */
  static std::unique_ptr<CoarseningRule> create_agglomeration(
      const AgglomerationRule::Config& config = {});

  /**
   * @brief Create composite rule
   */
  static std::unique_ptr<CompositeCoarseningRule> create_composite();
};

/**
 * @brief Coarsening utilities
 */
class CoarseningUtils {
public:
  /**
   * @brief Check if mesh can be coarsened
   */
  static bool can_coarsen_mesh(const MeshBase& mesh);

  /**
   * @brief Find coarsening candidates
   */
  static std::vector<size_t> find_coarsening_candidates(
      const MeshBase& mesh,
      const std::vector<double>& error_field,
      double threshold);

  /**
   * @brief Estimate quality after coarsening
   */
  static double estimate_coarsening_quality(
      const MeshBase& mesh,
      const std::vector<size_t>& elements_to_coarsen);

  /**
   * @brief Check if coarsening preserves topology
   */
  static bool preserves_topology(
      const MeshBase& mesh,
      const CoarseningOperation& operation);

  /**
   * @brief Compute optimal coarsening sequence
   */
  static std::vector<CoarseningOperation> compute_optimal_sequence(
      const MeshBase& mesh,
      const std::vector<size_t>& marked_elements);
};

} // namespace svmp

#endif // SVMP_COARSENING_RULES_H