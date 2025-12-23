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

#ifndef SVMP_QUALITY_GUARDS_H
#define SVMP_QUALITY_GUARDS_H

#include "Options.h"
#include <array>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace svmp {

// Forward declarations
class MeshBase;

// Quality options are currently sourced from AdaptivityOptions.
using QualityOptions = AdaptivityOptions;

/**
 * @brief Quality metrics for elements
 */
struct ElementQuality {
  /** Element ID */
  size_t element_id;

  /** Aspect ratio (1 = perfect) */
  double aspect_ratio = 1.0;

  /** Skewness (0 = perfect) */
  double skewness = 0.0;

  /** Jacobian determinant */
  double jacobian = 1.0;

  /** Minimum angle (degrees) */
  double min_angle = 60.0;

  /** Maximum angle (degrees) */
  double max_angle = 90.0;

  /** Volume or area */
  double size = 0.0;

  /** Edge length ratio */
  double edge_ratio = 1.0;

  /** Shape quality (0-1, 1 = perfect) */
  double shape_quality = 1.0;

  /** Distortion metric */
  double distortion = 0.0;

  /** Is element inverted */
  bool inverted = false;

  /** Overall quality score (0-1, 1 = perfect) */
  double overall_quality() const {
    if (inverted) return 0.0;

    // Weighted combination of metrics
    double score = 0.25 * (2.0 - aspect_ratio) +
                   0.25 * (1.0 - skewness) +
                   0.25 * shape_quality +
                   0.25 * std::min(1.0, min_angle / 30.0);

    return std::max(0.0, std::min(1.0, score));
  }
};

/**
 * @brief Mesh quality statistics
 */
struct MeshQualityReport {
  /** Minimum element quality */
  double min_quality = 0.0;

  /** Maximum element quality */
  double max_quality = 1.0;

  /** Average element quality */
  double avg_quality = 0.5;

  /** Number of poor quality elements */
  size_t num_poor_elements = 0;

  /** Number of inverted elements */
  size_t num_inverted = 0;

  /** Quality histogram (bins) */
  std::vector<size_t> quality_histogram;

  /** Worst elements by quality */
  std::vector<ElementQuality> worst_elements;

  /** Elements that failed quality checks */
  std::set<size_t> failed_elements;

  /** Is mesh acceptable */
  bool acceptable = true;

  /** Quality improvement suggestions */
  std::vector<std::string> suggestions;
};

/**
 * @brief Abstract base class for quality checking strategies
 */
class QualityChecker {
public:
  virtual ~QualityChecker() = default;

  /**
   * @brief Compute quality of single element
   *
   * @param mesh The mesh
   * @param elem_id Element ID
   * @return Element quality metrics
   */
  virtual ElementQuality compute_element_quality(
      const MeshBase& mesh,
      size_t elem_id) const = 0;

  /**
   * @brief Compute overall mesh quality
   *
   * @param mesh The mesh
   * @param options Quality options
   * @return Mesh quality statistics
   */
  virtual MeshQualityReport compute_mesh_quality(
      const MeshBase& mesh,
      const QualityOptions& options) const = 0;

  /**
   * @brief Check if element passes quality requirements
   *
   * @param quality Element quality
   * @param options Quality options
   * @return True if element passes
   */
  virtual bool check_element(
      const ElementQuality& quality,
      const QualityOptions& options) const = 0;

  /**
   * @brief Get checker name
   */
  virtual std::string name() const = 0;
};

/**
 * @brief Geometric quality checker
 *
 * Checks geometric properties like aspect ratio, angles, and shape.
 */
class GeometricQualityChecker : public QualityChecker {
public:
  /**
   * @brief Configuration for geometric checking
   */
  struct Config {
    /** Use normalized metrics */
    bool use_normalized = true;

    /** Check for self-intersection */
    bool check_self_intersection = false;

    /** Reference element for shape comparison */
    enum class ReferenceElement {
      EQUILATERAL,   // Equilateral triangle/tetrahedron
      SQUARE,        // Unit square/cube
      REGULAR        // Regular polygon/polyhedron
    };

    ReferenceElement reference = ReferenceElement::EQUILATERAL;

    /** Quality metric type */
    enum class MetricType {
      ASPECT_RATIO,
      CONDITION_NUMBER,
      EDGE_RATIO,
      RADIUS_RATIO,
      MEAN_RATIO
    };

    MetricType primary_metric = MetricType::ASPECT_RATIO;
  };

  GeometricQualityChecker() : GeometricQualityChecker(Config{}) {}
  explicit GeometricQualityChecker(const Config& config);

  ElementQuality compute_element_quality(
      const MeshBase& mesh,
      size_t elem_id) const override;

  MeshQualityReport compute_mesh_quality(
      const MeshBase& mesh,
      const QualityOptions& options) const override;

  bool check_element(
      const ElementQuality& quality,
      const QualityOptions& options) const override;

  std::string name() const override { return "GeometricQuality"; }

private:
  Config config_;

  /** Compute triangle quality */
  ElementQuality compute_triangle_quality(
      const MeshBase& mesh,
      size_t elem_id) const;

  /** Compute quad quality */
  ElementQuality compute_quad_quality(
      const MeshBase& mesh,
      size_t elem_id) const;

  /** Compute tet quality */
  ElementQuality compute_tet_quality(
      const MeshBase& mesh,
      size_t elem_id) const;

  /** Compute hex quality */
  ElementQuality compute_hex_quality(
      const MeshBase& mesh,
      size_t elem_id) const;

  /** Compute aspect ratio */
  double compute_aspect_ratio(
      const std::vector<std::array<double, 3>>& vertices) const;

  /** Compute skewness */
  double compute_skewness(
      const std::vector<std::array<double, 3>>& vertices) const;

  /** Compute angles */
  void compute_angles(
      const std::vector<std::array<double, 3>>& vertices,
      double& min_angle,
      double& max_angle) const;
};

/**
 * @brief Jacobian-based quality checker
 *
 * Checks Jacobian determinant and related metrics.
 */
class JacobianQualityChecker : public QualityChecker {
public:
  /**
   * @brief Configuration for Jacobian checking
   */
  struct Config {
    /** Number of sample points for Jacobian evaluation */
    size_t num_sample_points = 8;

    /** Check Jacobian at element corners only */
    bool corners_only = false;

    /** Use scaled Jacobian */
    bool use_scaled = true;

    /** Tolerance for zero Jacobian */
    double zero_tolerance = 1e-12;

    /** Check condition number */
    bool check_condition = true;
  };

  JacobianQualityChecker() : JacobianQualityChecker(Config{}) {}
  explicit JacobianQualityChecker(const Config& config);

  ElementQuality compute_element_quality(
      const MeshBase& mesh,
      size_t elem_id) const override;

  MeshQualityReport compute_mesh_quality(
      const MeshBase& mesh,
      const QualityOptions& options) const override;

  bool check_element(
      const ElementQuality& quality,
      const QualityOptions& options) const override;

  std::string name() const override { return "JacobianQuality"; }

private:
  Config config_;

  /** Compute Jacobian at parametric point */
  double compute_jacobian_at_point(
      const std::vector<std::array<double, 3>>& vertices,
      const std::array<double, 3>& xi) const;

  /** Get sample points in reference element */
  std::vector<std::array<double, 3>> get_sample_points(
      size_t elem_type) const;

  /** Compute condition number */
  double compute_condition_number(
      const std::vector<std::array<double, 3>>& vertices) const;
};

/**
 * @brief Size-based quality checker
 *
 * Checks element size variation and gradation.
 */
class SizeQualityChecker : public QualityChecker {
public:
  /**
   * @brief Configuration for size checking
   */
  struct Config {
    /** Maximum allowed size ratio between neighbors */
    double max_size_ratio = 2.0;

    /** Check anisotropy */
    bool check_anisotropy = true;

    /** Maximum allowed anisotropy */
    double max_anisotropy = 10.0;

    /** Use volume-based size metric */
    bool use_volume_metric = true;

    /** Check size field variation */
    bool check_size_field = false;
  };

  SizeQualityChecker() : SizeQualityChecker(Config{}) {}
  explicit SizeQualityChecker(const Config& config);

  ElementQuality compute_element_quality(
      const MeshBase& mesh,
      size_t elem_id) const override;

  MeshQualityReport compute_mesh_quality(
      const MeshBase& mesh,
      const QualityOptions& options) const override;

  bool check_element(
      const ElementQuality& quality,
      const QualityOptions& options) const override;

  std::string name() const override { return "SizeQuality"; }

private:
  Config config_;

  /** Compute size gradation */
  double compute_size_gradation(
      const MeshBase& mesh,
      size_t elem_id) const;

  /** Compute anisotropy */
  double compute_anisotropy(
      const std::vector<std::array<double, 3>>& vertices) const;

  /** Get neighbor elements */
  std::vector<size_t> get_neighbors(
      const MeshBase& mesh,
      size_t elem_id) const;
};

/**
 * @brief Composite quality checker
 *
 * Combines multiple quality checkers.
 */
class CompositeQualityChecker : public QualityChecker {
public:
  /**
   * @brief Add a quality checker with weight
   */
  void add_checker(
      std::unique_ptr<QualityChecker> checker,
      double weight = 1.0);

  ElementQuality compute_element_quality(
      const MeshBase& mesh,
      size_t elem_id) const override;

  MeshQualityReport compute_mesh_quality(
      const MeshBase& mesh,
      const QualityOptions& options) const override;

  bool check_element(
      const ElementQuality& quality,
      const QualityOptions& options) const override;

  std::string name() const override { return "CompositeQuality"; }

private:
  std::vector<std::pair<std::unique_ptr<QualityChecker>, double>> checkers_;

  /** Combine element qualities */
  ElementQuality combine_qualities(
      const std::vector<std::pair<ElementQuality, double>>& qualities) const;

  /** Combine mesh qualities */
  MeshQualityReport combine_mesh_qualities(
      const std::vector<std::pair<MeshQualityReport, double>>& qualities) const;
};

/**
 * @brief Quality improvement through smoothing
 */
class QualitySmoother {
public:
  /**
   * @brief Configuration for smoothing
   */
  struct Config {
    /** Maximum smoothing iterations */
    size_t max_iterations = 10;

    /** Convergence tolerance */
    double convergence_tolerance = 1e-4;

    /** Smoothing method */
    enum class Method {
      LAPLACIAN,           // Laplacian smoothing
      SMART_LAPLACIAN,     // Smart Laplacian (quality-aware)
      OPTIMIZATION_BASED,  // Optimization-based smoothing
      ANGLE_BASED,         // Angle-based smoothing
      COMBINED             // Combined approach
    };

    Method method = Method::SMART_LAPLACIAN;

    /** Preserve boundary nodes */
    bool preserve_boundary = true;

    /** Preserve feature edges */
    bool preserve_features = true;

    /** Feature angle threshold (degrees) */
    double feature_angle = 30.0;

    /** Relaxation factor */
    double relaxation = 0.5;
  };

  QualitySmoother() : QualitySmoother(Config{}) {}
  explicit QualitySmoother(const Config& config);

  /**
   * @brief Smooth mesh to improve quality
   *
   * @param mesh The mesh to smooth
   * @param checker Quality checker to evaluate improvements
   * @param options Quality options
   * @return Number of iterations performed
   */
  size_t smooth(
      MeshBase& mesh,
      const QualityChecker& checker,
      const QualityOptions& options);

  /**
   * @brief Smooth specific elements
   *
   * @param mesh The mesh
   * @param element_ids Elements to smooth
   * @param checker Quality checker
   * @param options Quality options
   * @return Number of nodes moved
   */
  size_t smooth_elements(
      MeshBase& mesh,
      const std::set<size_t>& element_ids,
      const QualityChecker& checker,
      const QualityOptions& options);

private:
  Config config_;

  /** Laplacian smoothing step */
  void laplacian_smooth(
      MeshBase& mesh,
      const std::set<size_t>& nodes);

  /** Smart Laplacian smoothing step */
  void smart_laplacian_smooth(
      MeshBase& mesh,
      const std::set<size_t>& nodes,
      const QualityChecker& checker);

  /** Optimization-based smoothing step */
  void optimization_smooth(
      MeshBase& mesh,
      const std::set<size_t>& nodes,
      const QualityChecker& checker);

  /** Find nodes to smooth */
  std::set<size_t> find_smoothing_nodes(
      const MeshBase& mesh,
      const std::set<size_t>& element_ids) const;

  /** Check if node is on boundary */
  bool is_boundary_node(
      const MeshBase& mesh,
      size_t node_id) const;

  /** Check if edge is a feature */
  bool is_feature_edge(
      const MeshBase& mesh,
      size_t v1, size_t v2) const;
};

/**
 * @brief Factory for creating quality checkers
 */
class QualityCheckerFactory {
public:
  /**
   * @brief Create quality checker based on options
   */
  static std::unique_ptr<QualityChecker> create(const QualityOptions& options);

  /**
   * @brief Create geometric quality checker
   */
  static std::unique_ptr<QualityChecker> create_geometric(
      const GeometricQualityChecker::Config& config = GeometricQualityChecker::Config{});

  /**
   * @brief Create Jacobian quality checker
   */
  static std::unique_ptr<QualityChecker> create_jacobian(
      const JacobianQualityChecker::Config& config = JacobianQualityChecker::Config{});

  /**
   * @brief Create size quality checker
   */
  static std::unique_ptr<QualityChecker> create_size(
      const SizeQualityChecker::Config& config = SizeQualityChecker::Config{});

  /**
   * @brief Create composite quality checker
   */
  static std::unique_ptr<QualityChecker> create_composite(
      const QualityOptions& options);
};

/**
 * @brief Quality guard utilities
 */
class QualityGuardUtils {
public:
  /**
   * @brief Check if mesh meets quality requirements
   */
  static bool check_mesh_quality(
      const MeshBase& mesh,
      const QualityOptions& options);

  /**
   * @brief Find elements that fail quality checks
   */
  static std::set<size_t> find_poor_elements(
      const MeshBase& mesh,
      const QualityChecker& checker,
      const QualityOptions& options);

  /**
   * @brief Compute quality improvement
   */
  static double compute_quality_improvement(
      const MeshQualityReport& before,
      const MeshQualityReport& after);

  /**
   * @brief Write quality report
   */
  static void write_quality_report(
      const MeshQualityReport& quality,
      const std::string& filename);

  /**
   * @brief Suggest quality improvements
   */
  static std::vector<std::string> suggest_improvements(
      const MeshBase& mesh,
      const MeshQualityReport& quality);
};

} // namespace svmp

#endif // SVMP_QUALITY_GUARDS_H
