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

#ifndef SVMP_MULTILEVEL_ADAPTIVITY_H
#define SVMP_MULTILEVEL_ADAPTIVITY_H

#include "Options.h"
#include "AdaptivityManager.h"
#include <memory>
#include <vector>
#include <map>

namespace svmp {

// Forward declarations
class MeshBase;
class MeshFields;

/**
 * @brief Hierarchical mesh level
 */
struct MeshLevel {
  /** Level index (0 = coarsest) */
  size_t level = 0;

  /** Mesh at this level */
  std::shared_ptr<MeshBase> mesh;

  /** Fields at this level */
  std::shared_ptr<MeshFields> fields;

  /** Parent-child relationships to finer level */
  std::map<size_t, std::vector<size_t>> refinement_map;

  /** Child-parent relationships from finer level */
  std::map<size_t, size_t> coarsening_map;

  /** Transfer operators to finer level */
  struct TransferOperators {
    std::vector<std::vector<double>> prolongation;
    std::vector<std::vector<double>> restriction;
  } operators;

  /** Level statistics */
  struct Statistics {
    size_t num_elements = 0;
    size_t num_nodes = 0;
    double min_element_size = 0.0;
    double max_element_size = 0.0;
    size_t memory_usage = 0;
  } stats;
};

/**
 * @brief Multi-level mesh hierarchy
 */
class MeshHierarchy {
public:
  /**
   * @brief Configuration for mesh hierarchy
   */
  struct Config {
    /** Maximum number of levels */
    size_t max_levels = 5;

    /** Coarsening ratio between levels */
    double coarsening_ratio = 2.0;

    /** Build method */
    enum class BuildMethod {
      GEOMETRIC,     // Geometric coarsening
      ALGEBRAIC,     // Algebraic coarsening
      AGGLOMERATION, // Element agglomeration
      NESTED         // Nested refinement
    };

    BuildMethod build_method = BuildMethod::GEOMETRIC;

    /** Store all intermediate levels */
    bool store_all_levels = true;

    /** Use shared vertices between levels */
    bool share_vertices = false;

    /** Minimum elements per level */
    size_t min_elements = 100;
  };

  explicit MeshHierarchy(const Config& config = {});

  /**
   * @brief Build hierarchy from finest mesh
   */
  void build_from_fine(const MeshBase& fine_mesh);

  /**
   * @brief Build hierarchy from coarsest mesh
   */
  void build_from_coarse(const MeshBase& coarse_mesh);

  /**
   * @brief Get level by index
   */
  MeshLevel& get_level(size_t level) { return levels_[level]; }
  const MeshLevel& get_level(size_t level) const { return levels_[level]; }

  /**
   * @brief Get number of levels
   */
  size_t num_levels() const { return levels_.size(); }

  /**
   * @brief Get finest level
   */
  MeshLevel& get_finest() { return levels_.back(); }
  const MeshLevel& get_finest() const { return levels_.back(); }

  /**
   * @brief Get coarsest level
   */
  MeshLevel& get_coarsest() { return levels_.front(); }
  const MeshLevel& get_coarsest() const { return levels_.front(); }

  /**
   * @brief Add a level
   */
  void add_level(const MeshLevel& level);

  /**
   * @brief Transfer field between levels
   */
  void transfer_field(size_t from_level, size_t to_level,
                      const std::vector<double>& from_field,
                      std::vector<double>& to_field) const;

  /**
   * @brief Prolongate field from coarse to fine
   */
  void prolongate(size_t coarse_level, size_t fine_level,
                  const std::vector<double>& coarse_field,
                  std::vector<double>& fine_field) const;

  /**
   * @brief Restrict field from fine to coarse
   */
  void restrict(size_t fine_level, size_t coarse_level,
                const std::vector<double>& fine_field,
                std::vector<double>& coarse_field) const;

private:
  Config config_;
  std::vector<MeshLevel> levels_;

  /** Build transfer operators between levels */
  void build_transfer_operators(size_t coarse_level, size_t fine_level);

  /** Geometric coarsening */
  std::shared_ptr<MeshBase> geometric_coarsen(const MeshBase& mesh);

  /** Algebraic coarsening */
  std::shared_ptr<MeshBase> algebraic_coarsen(const MeshBase& mesh);

  /** Agglomeration coarsening */
  std::shared_ptr<MeshBase> agglomeration_coarsen(const MeshBase& mesh);
};

/**
 * @brief Multi-grid error estimator
 *
 * Uses multiple mesh levels for error estimation.
 */
class MultiGridErrorEstimator {
public:
  /**
   * @brief Configuration
   */
  struct Config {
    /** Error estimation strategy */
    enum class Strategy {
      RICHARDSON,      // Richardson extrapolation
      HIERARCHICAL,    // Hierarchical basis
      CASCADIC,        // Cascadic multigrid
      GRADIENT_BASED   // Multi-level gradients
    };

    Strategy strategy = Strategy::RICHARDSON;

    /** Use solution on all levels */
    bool use_all_levels = true;

    /** Extrapolation order */
    size_t extrapolation_order = 2;

    /** Smoothing iterations */
    size_t smoothing_iterations = 3;
  };

  explicit MultiGridErrorEstimator(const Config& config = {});

  /**
   * @brief Estimate error using hierarchy
   */
  std::vector<double> estimate_error(
      const MeshHierarchy& hierarchy,
      size_t level) const;

  /**
   * @brief Richardson extrapolation error
   */
  std::vector<double> richardson_error(
      const MeshHierarchy& hierarchy,
      size_t level) const;

  /**
   * @brief Hierarchical basis error
   */
  std::vector<double> hierarchical_error(
      const MeshHierarchy& hierarchy,
      size_t level) const;

private:
  Config config_;

  /** Compute extrapolated solution */
  std::vector<double> extrapolate_solution(
      const std::vector<double>& coarse,
      const std::vector<double>& fine,
      double ratio) const;

  /** Smooth error estimate */
  void smooth_error(std::vector<double>& error,
                    const MeshBase& mesh) const;
};

/**
 * @brief Multi-level marking strategy
 *
 * Marks elements across multiple levels.
 */
class MultiLevelMarker {
public:
  /**
   * @brief Configuration
   */
  struct Config {
    /** Marking propagation */
    enum class Propagation {
      TOP_DOWN,     // Mark from fine to coarse
      BOTTOM_UP,    // Mark from coarse to fine
      SYNCHRONIZED  // Mark all levels together
    };

    Propagation propagation = Propagation::TOP_DOWN;

    /** Mark parent if any child is marked */
    bool mark_parent_with_children = true;

    /** Mark all children if parent is marked */
    bool mark_children_with_parent = true;

    /** Level-specific thresholds */
    std::vector<double> level_thresholds;

    /** Adaptive threshold adjustment */
    bool adaptive_thresholds = true;
  };

  explicit MultiLevelMarker(const Config& config = {});

  /**
   * @brief Mark elements across hierarchy
   */
  std::vector<std::vector<MarkType>> mark_hierarchy(
      const MeshHierarchy& hierarchy,
      const std::vector<std::vector<double>>& errors) const;

  /**
   * @brief Propagate marks between levels
   */
  void propagate_marks(
      std::vector<std::vector<MarkType>>& marks,
      const MeshHierarchy& hierarchy) const;

private:
  Config config_;

  /** Mark single level */
  std::vector<MarkType> mark_level(
      const MeshLevel& level,
      const std::vector<double>& error,
      double threshold) const;

  /** Propagate marks down hierarchy */
  void propagate_down(
      std::vector<std::vector<MarkType>>& marks,
      const MeshHierarchy& hierarchy) const;

  /** Propagate marks up hierarchy */
  void propagate_up(
      std::vector<std::vector<MarkType>>& marks,
      const MeshHierarchy& hierarchy) const;
};

/**
 * @brief Full multigrid adaptivity
 *
 * Uses FMG cycle for adaptive refinement.
 */
class FullMultiGridAdaptivity {
public:
  /**
   * @brief Configuration
   */
  struct Config {
    /** FMG cycle type */
    enum class CycleType {
      V_CYCLE,  // V-cycle
      W_CYCLE,  // W-cycle
      F_CYCLE   // F-cycle
    };

    CycleType cycle_type = CycleType::V_CYCLE;

    /** Pre-smoothing steps */
    size_t pre_smooth = 2;

    /** Post-smoothing steps */
    size_t post_smooth = 2;

    /** Coarse grid solver iterations */
    size_t coarse_solver_iterations = 10;

    /** Adaptation frequency */
    size_t adaptation_frequency = 5;

    /** Use nested iteration */
    bool nested_iteration = true;
  };

  explicit FullMultiGridAdaptivity(const Config& config = {});

  /**
   * @brief Perform FMG adaptive cycle
   */
  void fmg_cycle(MeshHierarchy& hierarchy);

  /**
   * @brief Single multigrid V-cycle
   */
  void v_cycle(MeshHierarchy& hierarchy, size_t level);

  /**
   * @brief Single multigrid W-cycle
   */
  void w_cycle(MeshHierarchy& hierarchy, size_t level);

private:
  Config config_;

  /** Smooth on level */
  void smooth(MeshLevel& level, size_t iterations);

  /** Solve on coarsest level */
  void coarse_solve(MeshLevel& level);

  /** Compute residual */
  std::vector<double> compute_residual(const MeshLevel& level) const;

  /** Apply correction */
  void apply_correction(MeshLevel& level,
                        const std::vector<double>& correction);
};

/**
 * @brief Adaptive mesh refinement with octrees
 */
class OctreeAdaptivity {
public:
  /**
   * @brief Octree node
   */
  struct OctreeNode {
    /** Node level */
    size_t level = 0;

    /** Node bounds */
    std::array<double, 6> bounds; // [xmin, ymin, zmin, xmax, ymax, zmax]

    /** Children nodes (8 for octree, 4 for quadtree) */
    std::vector<std::unique_ptr<OctreeNode>> children;

    /** Elements in this node */
    std::vector<size_t> elements;

    /** Is leaf node */
    bool is_leaf = true;

    /** Refinement flag */
    bool marked_for_refinement = false;

    /** Node data */
    std::vector<double> data;
  };

  /**
   * @brief Configuration
   */
  struct Config {
    /** Maximum tree depth */
    size_t max_depth = 10;

    /** Minimum elements per leaf */
    size_t min_elements = 1;

    /** Maximum elements per leaf */
    size_t max_elements = 10;

    /** Balance constraint (2:1 rule) */
    bool enforce_balance = true;

    /** Use spatial hashing */
    bool use_hashing = true;

    /** Tree type */
    enum class TreeType {
      QUADTREE, // 2D
      OCTREE    // 3D
    };

    TreeType tree_type = TreeType::OCTREE;
  };

  explicit OctreeAdaptivity(const Config& config = {});

  /**
   * @brief Build octree from mesh
   */
  void build_tree(const MeshBase& mesh);

  /**
   * @brief Refine octree node
   */
  void refine_node(OctreeNode* node);

  /**
   * @brief Coarsen octree node
   */
  void coarsen_node(OctreeNode* node);

  /**
   * @brief Balance octree (2:1 rule)
   */
  void balance_tree();

  /**
   * @brief Generate mesh from octree
   */
  std::shared_ptr<MeshBase> generate_mesh() const;

  /**
   * @brief Mark nodes for refinement
   */
  void mark_nodes(const std::vector<double>& error_field,
                  double threshold);

private:
  Config config_;
  std::unique_ptr<OctreeNode> root_;

  /** Recursive tree building */
  void build_node(OctreeNode* node, const MeshBase& mesh,
                  const std::vector<size_t>& elements, size_t depth);

  /** Split node into children */
  void split_node(OctreeNode* node);

  /** Merge children into parent */
  void merge_node(OctreeNode* node);

  /** Find neighbor nodes */
  std::vector<OctreeNode*> find_neighbors(OctreeNode* node);

  /** Check balance constraint */
  bool is_balanced(OctreeNode* node) const;

  /** Compute node bounds */
  std::array<double, 6> compute_bounds(
      const MeshBase& mesh,
      const std::vector<size_t>& elements) const;
};

/**
 * @brief Space-time adaptivity
 *
 * Adaptive refinement in space and time.
 */
class SpaceTimeAdaptivity {
public:
  /**
   * @brief Configuration
   */
  struct Config {
    /** Adaptation strategy */
    enum class Strategy {
      SIMULTANEOUS,  // Adapt space and time together
      SEQUENTIAL,    // Adapt space then time
      DECOUPLED      // Independent adaptation
    };

    Strategy strategy = Strategy::SIMULTANEOUS;

    /** Time step adaptation */
    bool adapt_time_step = true;

    /** Spatial adaptation frequency */
    size_t spatial_frequency = 5;

    /** Temporal error tolerance */
    double temporal_tolerance = 1e-3;

    /** CFL number */
    double cfl_number = 0.5;

    /** Maximum time step */
    double max_time_step = 1.0;

    /** Minimum time step */
    double min_time_step = 1e-6;
  };

  explicit SpaceTimeAdaptivity(const Config& config = {});

  /**
   * @brief Adapt in space and time
   */
  void adapt(MeshBase& mesh,
             double& time_step,
             const std::vector<double>& solution,
             double current_time);

  /**
   * @brief Estimate space-time error
   */
  std::vector<double> estimate_spacetime_error(
      const MeshBase& mesh,
      const std::vector<double>& solution,
      const std::vector<double>& previous_solution,
      double time_step) const;

  /**
   * @brief Compute optimal time step
   */
  double compute_time_step(const MeshBase& mesh,
                           const std::vector<double>& solution,
                           const std::vector<double>& error) const;

private:
  Config config_;
  std::vector<double> time_error_history_;
  std::vector<double> space_error_history_;

  /** Adapt spatial mesh */
  void adapt_space(MeshBase& mesh,
                   const std::vector<double>& spatial_error);

  /** Adapt time step */
  void adapt_time(double& time_step,
                  const std::vector<double>& temporal_error);

  /** Estimate temporal error */
  double estimate_temporal_error(
      const std::vector<double>& solution,
      const std::vector<double>& previous_solution,
      double time_step) const;
};

/**
 * @brief Multi-level adaptivity manager
 *
 * Orchestrates hierarchical adaptive refinement.
 */
class MultiLevelAdaptivityManager {
public:
  /**
   * @brief Configuration
   */
  struct Config {
    /** Use hierarchy */
    bool use_hierarchy = true;

    /** Use octree */
    bool use_octree = false;

    /** Use space-time */
    bool use_spacetime = false;

    /** Maximum hierarchy levels */
    size_t max_levels = 5;

    /** Adaptation strategy */
    enum class Strategy {
      MULTIGRID,    // Multi-grid based
      HIERARCHICAL, // Hierarchical basis
      CASCADIC,     // Cascadic multigrid
      ADAPTIVE_FMG  // Adaptive full multigrid
    };

    Strategy strategy = Strategy::MULTIGRID;
  };

  explicit MultiLevelAdaptivityManager(const Config& config = {});

  /**
   * @brief Perform multi-level adaptation
   */
  AdaptivityResult adapt_multilevel(
      MeshHierarchy& hierarchy,
      const AdaptivityOptions& options);

  /**
   * @brief Build initial hierarchy
   */
  void build_hierarchy(const MeshBase& initial_mesh);

  /**
   * @brief Get hierarchy
   */
  MeshHierarchy& get_hierarchy() { return hierarchy_; }

private:
  Config config_;
  MeshHierarchy hierarchy_;
  std::unique_ptr<MultiGridErrorEstimator> error_estimator_;
  std::unique_ptr<MultiLevelMarker> marker_;
  std::unique_ptr<FullMultiGridAdaptivity> fmg_solver_;
  std::unique_ptr<OctreeAdaptivity> octree_;
  std::unique_ptr<SpaceTimeAdaptivity> spacetime_;

  /** Adapt single level */
  void adapt_level(size_t level);

  /** Synchronize levels after adaptation */
  void synchronize_levels();

  /** Update transfer operators */
  void update_operators();
};

} // namespace svmp

#endif // SVMP_MULTILEVEL_ADAPTIVITY_H