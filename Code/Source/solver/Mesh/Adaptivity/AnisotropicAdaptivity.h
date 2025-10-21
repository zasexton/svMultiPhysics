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

#ifndef SVMP_ANISOTROPIC_ADAPTIVITY_H
#define SVMP_ANISOTROPIC_ADAPTIVITY_H

#include "Options.h"
#include "ErrorEstimator.h"
#include "Marker.h"
#include <array>
#include <memory>
#include <vector>

namespace svmp {

// Forward declarations
class MeshBase;
class MeshFields;

/**
 * @brief Metric tensor for anisotropic mesh adaptation
 *
 * Represents the desired mesh spacing and orientation at a point.
 */
struct MetricTensor {
  /** 2D metric tensor (symmetric 2x2) */
  std::array<double, 3> metric_2d = {1.0, 0.0, 1.0}; // [M11, M12, M22]

  /** 3D metric tensor (symmetric 3x3) */
  std::array<double, 6> metric_3d = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0}; // [M11, M12, M13, M22, M23, M33]

  /** Eigenvalues of the metric */
  std::array<double, 3> eigenvalues = {1.0, 1.0, 1.0};

  /** Eigenvectors of the metric (column-major) */
  std::array<std::array<double, 3>, 3> eigenvectors = {{
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0}
  }};

  /** Anisotropy ratio (max eigenvalue / min eigenvalue) */
  double anisotropy_ratio = 1.0;

  /** Is this a 2D or 3D metric */
  bool is_3d = false;

  /** Compute eigendecomposition */
  void compute_eigendecomposition();

  /** Get metric in direction */
  double evaluate_in_direction(const std::array<double, 3>& direction) const;

  /** Interpolate between two metrics */
  static MetricTensor interpolate(const MetricTensor& m1, const MetricTensor& m2, double t);

  /** Intersect two metrics (take minimum spacing) */
  static MetricTensor intersect(const MetricTensor& m1, const MetricTensor& m2);
};

/**
 * @brief Anisotropic marking for directional refinement
 */
struct AnisotropicMark {
  /** Element ID */
  size_t element_id;

  /** Refinement directions (bit flags) */
  uint8_t directions = 0;

  /** Direction vectors for refinement */
  std::vector<std::array<double, 3>> refine_directions;

  /** Refinement levels per direction */
  std::vector<size_t> direction_levels;

  /** Overall mark type */
  MarkType base_mark = MarkType::NONE;

  /** Check if marked for refinement in direction */
  bool refine_in_direction(size_t dir) const {
    return (directions & (1 << dir)) != 0;
  }

  /** Set refinement in direction */
  void set_direction(size_t dir) {
    directions |= (1 << dir);
  }
};

/**
 * @brief Anisotropic error estimator
 *
 * Estimates directional error and produces metric field.
 */
class AnisotropicErrorEstimator : public ErrorEstimator {
public:
  /**
   * @brief Configuration for anisotropic error estimation
   */
  struct Config {
    /** Method for directional error estimation */
    enum class Method {
      HESSIAN_BASED,      // Based on solution Hessian
      GRADIENT_BASED,     // Based on gradient direction
      RECOVERY_BASED,     // Directional recovery
      INTERPOLATION_ERROR // Interpolation error metric
    };

    Method method = Method::HESSIAN_BASED;

    /** Target number of elements */
    size_t target_elements = 10000;

    /** Minimum element size */
    double min_size = 1e-3;

    /** Maximum element size */
    double max_size = 1.0;

    /** Maximum anisotropy ratio */
    double max_anisotropy = 100.0;

    /** Metric gradation control */
    double gradation_parameter = 1.5;

    /** Use metric intersection */
    bool use_intersection = true;

    /** Smooth metric field */
    bool smooth_metric = true;
  };

  explicit AnisotropicErrorEstimator(const Config& config = {});

  void estimate_error(
      const MeshBase& mesh,
      const MeshFields* fields,
      std::vector<double>& error_estimate) const override;

  std::string name() const override { return "AnisotropicError"; }

  /**
   * @brief Compute metric field for mesh
   */
  std::vector<MetricTensor> compute_metric_field(
      const MeshBase& mesh,
      const MeshFields* fields) const;

  /**
   * @brief Compute directional error indicators
   */
  std::vector<std::array<double, 3>> compute_directional_errors(
      const MeshBase& mesh,
      const MeshFields* fields) const;

private:
  Config config_;

  /** Compute Hessian-based metric */
  MetricTensor compute_hessian_metric(
      const MeshBase& mesh,
      size_t element_id,
      const MeshFields* fields) const;

  /** Compute gradient-based metric */
  MetricTensor compute_gradient_metric(
      const MeshBase& mesh,
      size_t element_id,
      const MeshFields* fields) const;

  /** Compute solution Hessian */
  std::array<double, 6> compute_hessian(
      const MeshBase& mesh,
      size_t element_id,
      const std::vector<double>& field) const;

  /** Apply metric gradation */
  void apply_gradation(
      const MeshBase& mesh,
      std::vector<MetricTensor>& metrics) const;

  /** Smooth metric field */
  void smooth_metric_field(
      const MeshBase& mesh,
      std::vector<MetricTensor>& metrics) const;

  /** Normalize metric field to target complexity */
  void normalize_metric_field(
      const MeshBase& mesh,
      std::vector<MetricTensor>& metrics) const;
};

/**
 * @brief Anisotropic marking strategy
 *
 * Marks elements for directional refinement based on metric field.
 */
class AnisotropicMarker : public Marker {
public:
  /**
   * @brief Configuration for anisotropic marking
   */
  struct Config {
    /** Marking strategy */
    enum class Strategy {
      METRIC_BASED,       // Based on metric conformity
      ERROR_BASED,        // Based on directional errors
      FEATURE_ALIGNED,    // Aligned with features
      BOUNDARY_LAYER      // Boundary layer refinement
    };

    Strategy strategy = Strategy::METRIC_BASED;

    /** Metric conformity threshold */
    double conformity_threshold = 0.5;

    /** Directional refinement threshold */
    double direction_threshold = 0.7;

    /** Prefer structured patterns */
    bool prefer_structured = false;

    /** Maximum directional refinement level */
    size_t max_direction_level = 3;

    /** Boundary layer parameters */
    struct BoundaryLayer {
      double first_layer_height = 0.001;
      double growth_ratio = 1.2;
      size_t num_layers = 10;
    };

    BoundaryLayer boundary_layer;
  };

  explicit AnisotropicMarker(const Config& config = {});

  void mark_elements(
      const MeshBase& mesh,
      const std::vector<double>& error_indicator,
      std::vector<MarkType>& marks,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "AnisotropicMarker"; }

  /**
   * @brief Mark elements with directional information
   */
  std::vector<AnisotropicMark> mark_anisotropic(
      const MeshBase& mesh,
      const std::vector<MetricTensor>& metric_field) const;

private:
  Config config_;

  /** Check metric conformity */
  double compute_metric_conformity(
      const MeshBase& mesh,
      size_t element_id,
      const MetricTensor& metric) const;

  /** Determine refinement directions */
  std::vector<std::array<double, 3>> determine_directions(
      const MeshBase& mesh,
      size_t element_id,
      const MetricTensor& metric) const;

  /** Check if element needs directional refinement */
  bool needs_directional_refinement(
      const MeshBase& mesh,
      size_t element_id,
      const MetricTensor& metric) const;

  /** Generate boundary layer marks */
  std::vector<AnisotropicMark> generate_boundary_layer_marks(
      const MeshBase& mesh) const;
};

/**
 * @brief Anisotropic refinement rules
 *
 * Implements directional refinement patterns.
 */
class AnisotropicRefinementRules {
public:
  /**
   * @brief Configuration for anisotropic refinement
   */
  struct Config {
    /** Allow directional bisection */
    bool allow_bisection = true;

    /** Allow quad directional refinement */
    bool allow_quad_directional = true;

    /** Allow hex directional refinement */
    bool allow_hex_directional = true;

    /** Transition element handling */
    enum class TransitionType {
      PYRAMIDS,     // Use pyramid elements
      HANGING_NODE, // Allow hanging nodes
      TEMPLATE      // Use templates
    };

    TransitionType transition_type = TransitionType::TEMPLATE;
  };

  explicit AnisotropicRefinementRules(const Config& config = {});

  /**
   * @brief Apply anisotropic refinement
   */
  void refine_anisotropic(
      MeshBase& mesh,
      const std::vector<AnisotropicMark>& marks);

  /**
   * @brief Get refinement pattern for element
   */
  RefinementPattern get_anisotropic_pattern(
      ElementType type,
      const AnisotropicMark& mark) const;

private:
  Config config_;

  /** Refine triangle anisotropically */
  void refine_triangle_directional(
      MeshBase& mesh,
      size_t element_id,
      const AnisotropicMark& mark);

  /** Refine quadrilateral anisotropically */
  void refine_quad_directional(
      MeshBase& mesh,
      size_t element_id,
      const AnisotropicMark& mark);

  /** Refine tetrahedron anisotropically */
  void refine_tet_directional(
      MeshBase& mesh,
      size_t element_id,
      const AnisotropicMark& mark);

  /** Refine hexahedron anisotropically */
  void refine_hex_directional(
      MeshBase& mesh,
      size_t element_id,
      const AnisotropicMark& mark);

  /** Create transition elements */
  void create_transition_elements(
      MeshBase& mesh,
      size_t element_id,
      const std::vector<size_t>& refined_neighbors);
};

/**
 * @brief Metric-based mesh optimization
 *
 * Optimizes mesh to conform to metric field.
 */
class MetricMeshOptimizer {
public:
  /**
   * @brief Configuration for metric optimization
   */
  struct Config {
    /** Optimization method */
    enum class Method {
      EDGE_SWAPPING,    // Edge/face swapping
      VERTEX_SMOOTHING, // Vertex relocation
      TOPOLOGY_CHANGE,  // Topology modification
      COMBINED          // All methods
    };

    Method method = Method::COMBINED;

    /** Maximum optimization iterations */
    size_t max_iterations = 10;

    /** Convergence tolerance */
    double convergence_tolerance = 0.01;

    /** Quality threshold */
    double quality_threshold = 0.3;

    /** Enable edge swapping */
    bool enable_swapping = true;

    /** Enable vertex smoothing */
    bool enable_smoothing = true;

    /** Enable topology changes */
    bool enable_topology = false;
  };

  explicit MetricMeshOptimizer(const Config& config = {});

  /**
   * @brief Optimize mesh to metric field
   */
  void optimize(
      MeshBase& mesh,
      const std::vector<MetricTensor>& metric_field);

  /**
   * @brief Compute mesh-metric conformity
   */
  double compute_conformity(
      const MeshBase& mesh,
      const std::vector<MetricTensor>& metric_field) const;

private:
  Config config_;

  /** Perform edge swapping */
  size_t edge_swapping_pass(
      MeshBase& mesh,
      const std::vector<MetricTensor>& metric_field);

  /** Perform vertex smoothing */
  size_t vertex_smoothing_pass(
      MeshBase& mesh,
      const std::vector<MetricTensor>& metric_field);

  /** Check if edge swap improves quality */
  bool should_swap_edge(
      const MeshBase& mesh,
      size_t v1, size_t v2,
      const std::vector<MetricTensor>& metric_field) const;

  /** Compute optimal vertex position */
  std::array<double, 3> compute_optimal_position(
      const MeshBase& mesh,
      size_t vertex_id,
      const std::vector<MetricTensor>& metric_field) const;

  /** Compute element quality in metric */
  double compute_metric_quality(
      const MeshBase& mesh,
      size_t element_id,
      const MetricTensor& metric) const;
};

/**
 * @brief Size field for mesh adaptation
 *
 * Represents desired element sizes throughout the domain.
 */
class SizeField {
public:
  /**
   * @brief Size field types
   */
  enum class Type {
    UNIFORM,      // Uniform size
    ANALYTICAL,   // Analytical function
    DISCRETE,     // Discrete values at nodes
    BACKGROUND    // Background mesh interpolation
  };

  /**
   * @brief Configuration
   */
  struct Config {
    Type type = Type::UNIFORM;

    /** Uniform size value */
    double uniform_size = 1.0;

    /** Analytical function */
    std::function<double(const std::array<double, 3>&)> size_function;

    /** Background mesh for interpolation */
    const MeshBase* background_mesh = nullptr;

    /** Discrete size values */
    std::vector<double> node_sizes;

    /** Gradation control */
    double gradation = 1.5;

    /** Minimum size */
    double min_size = 1e-3;

    /** Maximum size */
    double max_size = 10.0;
  };

  explicit SizeField(const Config& config);

  /**
   * @brief Evaluate size at point
   */
  double evaluate(const std::array<double, 3>& point) const;

  /**
   * @brief Get size for element
   */
  double get_element_size(const MeshBase& mesh, size_t element_id) const;

  /**
   * @brief Convert to metric field
   */
  std::vector<MetricTensor> to_metric_field(const MeshBase& mesh) const;

  /**
   * @brief Apply gradation control
   */
  void apply_gradation(const MeshBase& mesh);

private:
  Config config_;

  /** Interpolate from background mesh */
  double interpolate_from_background(const std::array<double, 3>& point) const;

  /** Apply size limits */
  double apply_limits(double size) const;
};

/**
 * @brief Boundary layer mesh generator
 *
 * Creates anisotropic meshes for boundary layers.
 */
class BoundaryLayerGenerator {
public:
  /**
   * @brief Configuration for boundary layer generation
   */
  struct Config {
    /** First layer height */
    double first_height = 0.001;

    /** Growth ratio between layers */
    double growth_ratio = 1.2;

    /** Number of layers */
    size_t num_layers = 10;

    /** Total layer thickness */
    double total_thickness = 0.1;

    /** Blend with volume mesh */
    bool blend_with_volume = true;

    /** Transition type */
    enum class Transition {
      SMOOTH,    // Smooth transition
      GEOMETRIC, // Geometric progression
      HYPERBOLIC // Hyperbolic tangent
    };

    Transition transition = Transition::GEOMETRIC;

    /** Boundary surfaces to apply layers */
    std::vector<int> boundary_ids;
  };

  explicit BoundaryLayerGenerator(const Config& config);

  /**
   * @brief Generate boundary layer mesh
   */
  void generate_layers(
      MeshBase& mesh,
      const std::vector<int>& boundary_surfaces);

  /**
   * @brief Compute layer metric field
   */
  std::vector<MetricTensor> compute_layer_metrics(
      const MeshBase& mesh) const;

  /**
   * @brief Check layer quality
   */
  bool check_layer_quality(const MeshBase& mesh) const;

private:
  Config config_;

  /** Extrude boundary faces */
  void extrude_boundary_faces(
      MeshBase& mesh,
      const std::vector<size_t>& boundary_faces);

  /** Compute layer heights */
  std::vector<double> compute_layer_heights() const;

  /** Compute growth direction */
  std::array<double, 3> compute_growth_direction(
      const MeshBase& mesh,
      size_t boundary_face) const;

  /** Create prism/hex elements */
  void create_layer_elements(
      MeshBase& mesh,
      const std::vector<size_t>& base_nodes,
      const std::vector<size_t>& top_nodes);

  /** Smooth layer mesh */
  void smooth_layers(MeshBase& mesh);
};

/**
 * @brief Anisotropic adaptivity manager
 *
 * Orchestrates anisotropic mesh adaptation.
 */
class AnisotropicAdaptivityManager {
public:
  /**
   * @brief Configuration
   */
  struct Config {
    /** Enable metric-based adaptation */
    bool use_metric = true;

    /** Enable directional refinement */
    bool use_directional = true;

    /** Enable boundary layers */
    bool use_boundary_layers = false;

    /** Optimization after adaptation */
    bool optimize_mesh = true;

    /** Maximum adaptation iterations */
    size_t max_iterations = 10;

    /** Target complexity */
    size_t target_complexity = 10000;
  };

  explicit AnisotropicAdaptivityManager(const Config& config = {});

  /**
   * @brief Perform anisotropic adaptation
   */
  AdaptivityResult adapt_anisotropic(
      MeshBase& mesh,
      MeshFields* fields,
      const AdaptivityOptions& options);

  /**
   * @brief Set metric field
   */
  void set_metric_field(const std::vector<MetricTensor>& metrics) {
    metric_field_ = metrics;
  }

  /**
   * @brief Get current metric field
   */
  const std::vector<MetricTensor>& get_metric_field() const {
    return metric_field_;
  }

private:
  Config config_;
  std::vector<MetricTensor> metric_field_;
  std::unique_ptr<AnisotropicErrorEstimator> error_estimator_;
  std::unique_ptr<AnisotropicMarker> marker_;
  std::unique_ptr<AnisotropicRefinementRules> refinement_rules_;
  std::unique_ptr<MetricMeshOptimizer> optimizer_;
  std::unique_ptr<BoundaryLayerGenerator> boundary_layer_gen_;

  /** Compute metric field from solution */
  void compute_metric_from_solution(
      const MeshBase& mesh,
      const MeshFields* fields);

  /** Apply anisotropic refinement */
  void apply_refinement(
      MeshBase& mesh,
      const std::vector<AnisotropicMark>& marks);

  /** Optimize mesh to metric */
  void optimize_to_metric(MeshBase& mesh);

  /** Check convergence */
  bool check_convergence(
      const MeshBase& mesh,
      const std::vector<MetricTensor>& old_metrics,
      const std::vector<MetricTensor>& new_metrics) const;
};

} // namespace svmp

#endif // SVMP_ANISOTROPIC_ADAPTIVITY_H