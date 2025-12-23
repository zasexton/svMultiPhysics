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

#ifndef SVMP_MARKER_H
#define SVMP_MARKER_H

#include "Options.h"
#include <memory>
#include <string>
#include <vector>

namespace svmp {

// Forward declaration
class MeshBase;

/**
 * @brief Element marking decision
 */
enum class MarkType {
  NONE,      // No action
  REFINE,    // Mark for refinement
  COARSEN    // Mark for coarsening
};

/**
 * @brief Abstract base class for element marking strategies
 *
 * Markers determine which elements should be refined or coarsened
 * based on error indicators and adaptivity options.
 */
class Marker {
public:
  virtual ~Marker() = default;

  /**
   * @brief Mark elements for refinement/coarsening
   *
   * @param indicators Error indicators (one per element)
   * @param mesh The mesh containing the elements
   * @param options Adaptivity options
   * @return Vector of marking decisions (one per element)
   */
  virtual std::vector<MarkType> mark(
      const std::vector<double>& indicators,
      const MeshBase& mesh,
      const AdaptivityOptions& options) const = 0;

  /**
   * @brief Get marker name for diagnostics
   */
  virtual std::string name() const = 0;

  /**
   * @brief Get statistics about marking
   */
  struct MarkingStats {
    size_t num_marked_refine = 0;
    size_t num_marked_coarsen = 0;
    size_t num_unmarked = 0;
    double refine_threshold = 0.0;
    double coarsen_threshold = 0.0;
  };

  virtual MarkingStats get_stats() const { return last_stats_; }

protected:
  mutable MarkingStats last_stats_;
};

/**
 * @brief Fixed fraction (bulk/Dörfler) marking strategy
 *
 * Marks a fixed fraction of elements with the highest/lowest errors.
 */
class FixedFractionMarker : public Marker {
public:
  /**
   * @brief Configuration for fixed fraction marking
   */
  struct Config {
    /** Fraction of elements to mark for refinement */
    double refine_fraction = 0.3;

    /** Fraction of elements to mark for coarsening */
    double coarsen_fraction = 0.1;

    /** Use Dörfler marking (mark smallest set covering fraction of error) */
    bool use_doerfler = true;

    /** Minimum indicator value to consider for refinement */
    double min_refine_indicator = 0.0;

    /** Maximum indicator value to consider for coarsening */
    double max_coarsen_indicator = std::numeric_limits<double>::max();
	  };

	  FixedFractionMarker() : FixedFractionMarker(Config{}) {}
	  explicit FixedFractionMarker(const Config& config);

  std::vector<MarkType> mark(
      const std::vector<double>& indicators,
      const MeshBase& mesh,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "FixedFraction"; }

private:
  Config config_;

  /** Compute thresholds using Dörfler marking */
  std::pair<double, double> compute_doerfler_thresholds(
      const std::vector<double>& indicators) const;

  /** Compute thresholds using simple fraction */
  std::pair<double, double> compute_fraction_thresholds(
      const std::vector<double>& indicators) const;
};

/**
 * @brief Threshold-based marking strategy
 *
 * Marks elements based on absolute or relative error thresholds.
 */
class ThresholdMarker : public Marker {
public:
  /**
   * @brief Configuration for threshold marking
   */
  struct Config {
    /** Type of threshold */
    enum class ThresholdType {
      ABSOLUTE,   // Use absolute threshold values
      RELATIVE    // Use relative to max error
    };

    ThresholdType threshold_type = ThresholdType::RELATIVE;

    /** Refinement threshold (absolute or relative) */
    double refine_threshold = 0.5;

    /** Coarsening threshold (absolute or relative) */
    double coarsen_threshold = 0.1;

    /** Use mean + k*stddev for threshold */
    bool use_statistical = false;

    /** Number of standard deviations for refinement */
    double refine_std_dev = 2.0;

    /** Number of standard deviations for coarsening */
    double coarsen_std_dev = -1.0;
	  };

	  ThresholdMarker() : ThresholdMarker(Config{}) {}
	  explicit ThresholdMarker(const Config& config);

  std::vector<MarkType> mark(
      const std::vector<double>& indicators,
      const MeshBase& mesh,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "Threshold"; }

private:
  Config config_;

  /** Compute statistical thresholds */
  std::pair<double, double> compute_statistical_thresholds(
      const std::vector<double>& indicators) const;
};

/**
 * @brief Fixed count marking strategy
 *
 * Marks a fixed number of elements.
 */
class FixedCountMarker : public Marker {
public:
  /**
   * @brief Configuration for fixed count marking
   */
  struct Config {
    /** Number of elements to mark for refinement */
    size_t refine_count = 100;

    /** Number of elements to mark for coarsening */
    size_t coarsen_count = 50;

    /** Ensure non-overlapping marks */
    bool exclusive_marking = true;
	  };

	  FixedCountMarker() : FixedCountMarker(Config{}) {}
	  explicit FixedCountMarker(const Config& config);

  std::vector<MarkType> mark(
      const std::vector<double>& indicators,
      const MeshBase& mesh,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "FixedCount"; }

private:
  Config config_;
};

/**
 * @brief Region-aware marking strategy
 *
 * Marks elements with region/boundary constraints.
 */
class RegionAwareMarker : public Marker {
public:
  /**
   * @brief Configuration for region-aware marking
   */
  struct Config {
    /** Base marker to use */
    std::unique_ptr<Marker> base_marker;

    /** Region labels to include (empty = all) */
    std::set<int> include_regions;

    /** Region labels to exclude */
    std::set<int> exclude_regions;

    /** Boundary labels to exclude from marking */
    std::set<int> exclude_boundaries;

    /** Mark all elements in specified regions */
    std::set<int> force_refine_regions;

    /** Never mark elements in specified regions */
    std::set<int> force_preserve_regions;
  };

  explicit RegionAwareMarker(Config config);

  std::vector<MarkType> mark(
      const std::vector<double>& indicators,
      const MeshBase& mesh,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "RegionAware"; }

private:
  Config config_;

  /** Check if element should be considered for marking */
  bool should_consider_element(
      const MeshBase& mesh,
      size_t elem_id) const;
};

/**
 * @brief Gradient-based marking strategy
 *
 * Marks elements based on gradient of error indicators.
 */
class GradientMarker : public Marker {
public:
  /**
   * @brief Configuration for gradient marking
   */
  struct Config {
    /** Mark based on gradient magnitude */
    bool use_gradient_magnitude = true;

    /** Mark based on gradient direction changes */
    bool detect_discontinuities = false;

    /** Gradient threshold for refinement */
    double refine_gradient_threshold = 1.0;

    /** Gradient threshold for coarsening */
    double coarsen_gradient_threshold = 0.1;

    /** Normalize gradients */
    bool normalize_gradients = true;
	  };

	  GradientMarker() : GradientMarker(Config{}) {}
	  explicit GradientMarker(const Config& config);

  std::vector<MarkType> mark(
      const std::vector<double>& indicators,
      const MeshBase& mesh,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "Gradient"; }

private:
  Config config_;

  /** Compute gradient of indicators */
  std::vector<double> compute_gradients(
      const std::vector<double>& indicators,
      const MeshBase& mesh) const;
};

/**
 * @brief Composite marking strategy
 *
 * Combines multiple marking strategies.
 */
class CompositeMarker : public Marker {
public:
  /**
   * @brief Combination method
   */
  enum class CombinationMethod {
    UNION,         // Mark if any strategy marks
    INTERSECTION,  // Mark only if all strategies mark
    MAJORITY,      // Mark if majority of strategies mark
    WEIGHTED       // Weighted voting
  };

  explicit CompositeMarker(CombinationMethod method = CombinationMethod::UNION);

  /**
   * @brief Add a marking strategy
   */
  void add_marker(std::unique_ptr<Marker> marker, double weight = 1.0);

  std::vector<MarkType> mark(
      const std::vector<double>& indicators,
      const MeshBase& mesh,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "Composite"; }

private:
  CombinationMethod combination_method_;
  std::vector<std::pair<std::unique_ptr<Marker>, double>> markers_;

  /** Combine marking decisions */
  std::vector<MarkType> combine_marks(
      const std::vector<std::vector<MarkType>>& all_marks,
      const std::vector<double>& weights) const;
};

/**
 * @brief Factory for creating markers
 */
class MarkerFactory {
public:
  /**
   * @brief Create a marker based on options
   */
  static std::unique_ptr<Marker> create(const AdaptivityOptions& options);

  /**
   * @brief Create a fixed fraction marker
   */
  static std::unique_ptr<Marker> create_fixed_fraction(
      double refine_fraction = 0.3,
      double coarsen_fraction = 0.1,
      bool use_doerfler = true);

  /**
   * @brief Create a threshold marker
   */
  static std::unique_ptr<Marker> create_threshold(
      double refine_threshold,
      double coarsen_threshold,
      bool relative = true);

  /**
   * @brief Create a fixed count marker
   */
  static std::unique_ptr<Marker> create_fixed_count(
      size_t refine_count,
      size_t coarsen_count);

  /**
   * @brief Create a region-aware marker
   */
  static std::unique_ptr<Marker> create_region_aware(
      std::unique_ptr<Marker> base_marker,
      const std::set<int>& include_regions = {},
      const std::set<int>& exclude_regions = {});
};

/**
 * @brief Utility functions for marking
 */
class MarkerUtils {
public:
  /**
   * @brief Count marks by type
   */
  static Marker::MarkingStats count_marks(const std::vector<MarkType>& marks);

  /**
   * @brief Convert marks to boolean vectors
   */
  static std::pair<std::vector<bool>, std::vector<bool>> marks_to_flags(
      const std::vector<MarkType>& marks);

  /**
   * @brief Apply marking constraints (max level, element count, etc.)
   */
  static void apply_constraints(
      std::vector<MarkType>& marks,
      const MeshBase& mesh,
      const AdaptivityOptions& options);

  /**
   * @brief Smooth marking to avoid isolated marks
   */
  static void smooth_marking(
      std::vector<MarkType>& marks,
      const MeshBase& mesh,
      int num_smooth_passes = 1);

  /**
   * @brief Write marks to field for visualization
   */
  static void write_marks_to_field(
      MeshBase& mesh,
      const std::string& field_name,
      const std::vector<MarkType>& marks);
};

} // namespace svmp

#endif // SVMP_MARKER_H
