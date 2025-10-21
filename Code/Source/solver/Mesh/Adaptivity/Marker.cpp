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

#include "Marker.h"
#include "../Core/MeshBase.h"
#include "../Fields/MeshFields.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace svmp {

// ====================
// FixedFractionMarker Implementation
// ====================

FixedFractionMarker::FixedFractionMarker(const Config& config)
    : config_(config) {
  if (config_.refine_fraction < 0.0 || config_.refine_fraction > 1.0) {
    throw std::invalid_argument("Refine fraction must be in [0, 1]");
  }
  if (config_.coarsen_fraction < 0.0 || config_.coarsen_fraction > 1.0) {
    throw std::invalid_argument("Coarsen fraction must be in [0, 1]");
  }
  if (config_.refine_fraction + config_.coarsen_fraction > 1.0) {
    throw std::invalid_argument("Sum of refine and coarsen fractions cannot exceed 1.0");
  }
}

std::vector<MarkType> FixedFractionMarker::mark(
    const std::vector<double>& indicators,
    const MeshBase& mesh,
    const AdaptivityOptions& options) const {

  size_t num_elems = indicators.size();
  std::vector<MarkType> marks(num_elems, MarkType::NONE);

  if (num_elems == 0) return marks;

  // Get actual fractions from config or options
  double refine_frac = options.enable_refinement ?
      (options.refine_fraction > 0 ? options.refine_fraction : config_.refine_fraction) : 0.0;
  double coarsen_frac = options.enable_coarsening ?
      (options.coarsen_fraction > 0 ? options.coarsen_fraction : config_.coarsen_fraction) : 0.0;

  // Compute thresholds
  std::pair<double, double> thresholds;
  if (config_.use_doerfler) {
    thresholds = compute_doerfler_thresholds(indicators);
  } else {
    thresholds = compute_fraction_thresholds(indicators);
  }

  last_stats_.refine_threshold = thresholds.first;
  last_stats_.coarsen_threshold = thresholds.second;

  // Apply marking
  for (size_t i = 0; i < num_elems; ++i) {
    if (indicators[i] >= thresholds.first && indicators[i] >= config_.min_refine_indicator) {
      marks[i] = MarkType::REFINE;
      last_stats_.num_marked_refine++;
    } else if (indicators[i] <= thresholds.second && indicators[i] <= config_.max_coarsen_indicator) {
      marks[i] = MarkType::COARSEN;
      last_stats_.num_marked_coarsen++;
    } else {
      last_stats_.num_unmarked++;
    }
  }

  return marks;
}

std::pair<double, double> FixedFractionMarker::compute_doerfler_thresholds(
    const std::vector<double>& indicators) const {

  // Dörfler marking: mark smallest set of elements that contribute
  // a fraction theta of the total error

  // Create sorted indices
  std::vector<size_t> sorted_indices(indicators.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
           [&indicators](size_t a, size_t b) {
             return indicators[a] > indicators[b];  // Descending order
           });

  // Compute total error (L2 norm)
  double total_error = 0.0;
  for (double val : indicators) {
    total_error += val * val;
  }
  total_error = std::sqrt(total_error);

  // Find refinement threshold (Dörfler marking)
  double refine_threshold = 0.0;
  if (config_.refine_fraction > 0.0 && total_error > 0.0) {
    double target_error = config_.refine_fraction * total_error;
    double accumulated_error = 0.0;

    for (size_t idx : sorted_indices) {
      accumulated_error += indicators[idx] * indicators[idx];
      refine_threshold = indicators[idx];
      if (std::sqrt(accumulated_error) >= target_error) {
        break;
      }
    }
  }

  // Find coarsening threshold (mark elements with smallest errors)
  double coarsen_threshold = std::numeric_limits<double>::max();
  if (config_.coarsen_fraction > 0.0) {
    size_t num_to_coarsen = static_cast<size_t>(config_.coarsen_fraction * indicators.size());
    if (num_to_coarsen > 0 && num_to_coarsen <= indicators.size()) {
      size_t coarsen_idx = indicators.size() - num_to_coarsen;
      coarsen_threshold = indicators[sorted_indices[coarsen_idx]];
    }
  }

  return {refine_threshold, coarsen_threshold};
}

std::pair<double, double> FixedFractionMarker::compute_fraction_thresholds(
    const std::vector<double>& indicators) const {

  // Simple fraction-based marking
  std::vector<double> sorted = indicators;
  std::sort(sorted.begin(), sorted.end(), std::greater<double>());

  double refine_threshold = 0.0;
  double coarsen_threshold = std::numeric_limits<double>::max();

  // Refinement threshold
  if (config_.refine_fraction > 0.0) {
    size_t refine_idx = static_cast<size_t>(config_.refine_fraction * sorted.size());
    if (refine_idx > 0 && refine_idx <= sorted.size()) {
      refine_threshold = sorted[refine_idx - 1];
    }
  }

  // Coarsening threshold
  if (config_.coarsen_fraction > 0.0) {
    size_t coarsen_idx = sorted.size() -
        static_cast<size_t>(config_.coarsen_fraction * sorted.size());
    if (coarsen_idx < sorted.size()) {
      coarsen_threshold = sorted[coarsen_idx];
    }
  }

  return {refine_threshold, coarsen_threshold};
}

// ====================
// ThresholdMarker Implementation
// ====================

ThresholdMarker::ThresholdMarker(const Config& config)
    : config_(config) {
}

std::vector<MarkType> ThresholdMarker::mark(
    const std::vector<double>& indicators,
    const MeshBase& mesh,
    const AdaptivityOptions& options) const {

  size_t num_elems = indicators.size();
  std::vector<MarkType> marks(num_elems, MarkType::NONE);

  if (num_elems == 0) return marks;

  // Compute thresholds
  double refine_threshold, coarsen_threshold;

  if (config_.use_statistical) {
    auto thresholds = compute_statistical_thresholds(indicators);
    refine_threshold = thresholds.first;
    coarsen_threshold = thresholds.second;
  } else {
    // Use configured thresholds
    if (config_.threshold_type == Config::ThresholdType::RELATIVE) {
      double max_indicator = *std::max_element(indicators.begin(), indicators.end());
      refine_threshold = config_.refine_threshold * max_indicator;
      coarsen_threshold = config_.coarsen_threshold * max_indicator;
    } else {
      refine_threshold = config_.refine_threshold;
      coarsen_threshold = config_.coarsen_threshold;
    }
  }

  last_stats_.refine_threshold = refine_threshold;
  last_stats_.coarsen_threshold = coarsen_threshold;

  // Apply marking
  for (size_t i = 0; i < num_elems; ++i) {
    if (options.enable_refinement && indicators[i] >= refine_threshold) {
      marks[i] = MarkType::REFINE;
      last_stats_.num_marked_refine++;
    } else if (options.enable_coarsening && indicators[i] <= coarsen_threshold) {
      marks[i] = MarkType::COARSEN;
      last_stats_.num_marked_coarsen++;
    } else {
      last_stats_.num_unmarked++;
    }
  }

  return marks;
}

std::pair<double, double> ThresholdMarker::compute_statistical_thresholds(
    const std::vector<double>& indicators) const {

  // Compute mean and standard deviation
  double mean = std::accumulate(indicators.begin(), indicators.end(), 0.0) / indicators.size();

  double variance = 0.0;
  for (double val : indicators) {
    double diff = val - mean;
    variance += diff * diff;
  }
  double std_dev = std::sqrt(variance / indicators.size());

  // Compute thresholds based on mean and standard deviation
  double refine_threshold = mean + config_.refine_std_dev * std_dev;
  double coarsen_threshold = mean + config_.coarsen_std_dev * std_dev;

  return {refine_threshold, coarsen_threshold};
}

// ====================
// FixedCountMarker Implementation
// ====================

FixedCountMarker::FixedCountMarker(const Config& config)
    : config_(config) {
}

std::vector<MarkType> FixedCountMarker::mark(
    const std::vector<double>& indicators,
    const MeshBase& mesh,
    const AdaptivityOptions& options) const {

  size_t num_elems = indicators.size();
  std::vector<MarkType> marks(num_elems, MarkType::NONE);

  if (num_elems == 0) return marks;

  // Create sorted indices
  std::vector<size_t> sorted_indices(num_elems);
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
           [&indicators](size_t a, size_t b) {
             return indicators[a] > indicators[b];  // Descending order
           });

  // Determine actual counts
  size_t refine_count = options.enable_refinement ?
      std::min(config_.refine_count, num_elems) : 0;
  size_t coarsen_count = options.enable_coarsening ?
      std::min(config_.coarsen_count, num_elems) : 0;

  // Ensure non-overlapping if exclusive
  if (config_.exclusive_marking) {
    if (refine_count + coarsen_count > num_elems) {
      // Scale down proportionally
      double scale = static_cast<double>(num_elems) / (refine_count + coarsen_count);
      refine_count = static_cast<size_t>(refine_count * scale);
      coarsen_count = num_elems - refine_count;
    }
  }

  // Mark for refinement (highest errors)
  for (size_t i = 0; i < refine_count; ++i) {
    marks[sorted_indices[i]] = MarkType::REFINE;
    last_stats_.num_marked_refine++;
  }

  // Mark for coarsening (lowest errors)
  size_t coarsen_start = config_.exclusive_marking ?
      num_elems - coarsen_count : std::max(num_elems - coarsen_count, refine_count);

  for (size_t i = coarsen_start; i < num_elems; ++i) {
    if (marks[sorted_indices[i]] == MarkType::NONE) {
      marks[sorted_indices[i]] = MarkType::COARSEN;
      last_stats_.num_marked_coarsen++;
    }
  }

  last_stats_.num_unmarked = num_elems - last_stats_.num_marked_refine - last_stats_.num_marked_coarsen;

  return marks;
}

// ====================
// RegionAwareMarker Implementation
// ====================

RegionAwareMarker::RegionAwareMarker(Config config)
    : config_(std::move(config)) {
  if (!config_.base_marker) {
    throw std::invalid_argument("Base marker must be provided for RegionAwareMarker");
  }
}

std::vector<MarkType> RegionAwareMarker::mark(
    const std::vector<double>& indicators,
    const MeshBase& mesh,
    const AdaptivityOptions& options) const {

  // Get base marking
  std::vector<MarkType> marks = config_.base_marker->mark(indicators, mesh, options);

  // Apply region constraints
  for (size_t i = 0; i < marks.size(); ++i) {
    // Check if element should be considered
    if (!should_consider_element(mesh, i)) {
      marks[i] = MarkType::NONE;
      continue;
    }

    // Force refinement in specified regions
    int region_label = mesh.cell_label(i, 0);  // Assuming region label at index 0
    if (config_.force_refine_regions.count(region_label) > 0) {
      marks[i] = MarkType::REFINE;
    }

    // Force preservation in specified regions
    if (config_.force_preserve_regions.count(region_label) > 0) {
      marks[i] = MarkType::NONE;
    }
  }

  // Update statistics
  last_stats_ = MarkerUtils::count_marks(marks);

  return marks;
}

bool RegionAwareMarker::should_consider_element(
    const MeshBase& mesh,
    size_t elem_id) const {

  int region_label = mesh.cell_label(elem_id, 0);

  // Check inclusion list
  if (!config_.include_regions.empty()) {
    if (config_.include_regions.count(region_label) == 0) {
      return false;
    }
  }

  // Check exclusion list
  if (config_.exclude_regions.count(region_label) > 0) {
    return false;
  }

  // Check boundary exclusion
  // TODO: Check if element is on boundary with excluded label

  return true;
}

// ====================
// GradientMarker Implementation
// ====================

GradientMarker::GradientMarker(const Config& config)
    : config_(config) {
}

std::vector<MarkType> GradientMarker::mark(
    const std::vector<double>& indicators,
    const MeshBase& mesh,
    const AdaptivityOptions& options) const {

  size_t num_elems = indicators.size();
  std::vector<MarkType> marks(num_elems, MarkType::NONE);

  if (num_elems == 0) return marks;

  // Compute gradients
  std::vector<double> gradients = compute_gradients(indicators, mesh);

  // Normalize if requested
  if (config_.normalize_gradients && !gradients.empty()) {
    double max_grad = *std::max_element(gradients.begin(), gradients.end());
    if (max_grad > 0.0) {
      for (auto& grad : gradients) {
        grad /= max_grad;
      }
    }
  }

  // Apply marking based on gradients
  for (size_t i = 0; i < num_elems; ++i) {
    if (options.enable_refinement && gradients[i] >= config_.refine_gradient_threshold) {
      marks[i] = MarkType::REFINE;
      last_stats_.num_marked_refine++;
    } else if (options.enable_coarsening && gradients[i] <= config_.coarsen_gradient_threshold) {
      marks[i] = MarkType::COARSEN;
      last_stats_.num_marked_coarsen++;
    } else {
      last_stats_.num_unmarked++;
    }
  }

  return marks;
}

std::vector<double> GradientMarker::compute_gradients(
    const std::vector<double>& indicators,
    const MeshBase& mesh) const {

  size_t num_elems = indicators.size();
  std::vector<double> gradients(num_elems, 0.0);

  // Compute gradient magnitude for each element
  for (size_t i = 0; i < num_elems; ++i) {
    // Get element neighbors
    auto neighbors = mesh.cell_neighbors(i);

    if (neighbors.empty()) continue;

    double grad_mag = 0.0;
    size_t count = 0;

    for (size_t neighbor : neighbors) {
      if (neighbor < num_elems) {
        double diff = indicators[neighbor] - indicators[i];
        // Approximate distance using element sizes
        double h = std::pow(mesh.cell_measure(i), 1.0/mesh.spatial_dim());
        double h_neighbor = std::pow(mesh.cell_measure(neighbor), 1.0/mesh.spatial_dim());
        double dist = 0.5 * (h + h_neighbor);

        grad_mag += (diff / dist) * (diff / dist);
        count++;
      }
    }

    if (count > 0) {
      gradients[i] = std::sqrt(grad_mag / count);
    }
  }

  return gradients;
}

// ====================
// CompositeMarker Implementation
// ====================

CompositeMarker::CompositeMarker(CombinationMethod method)
    : combination_method_(method) {
}

void CompositeMarker::add_marker(std::unique_ptr<Marker> marker, double weight) {
  if (!marker) {
    throw std::invalid_argument("Cannot add null marker");
  }
  markers_.emplace_back(std::move(marker), weight);
}

std::vector<MarkType> CompositeMarker::mark(
    const std::vector<double>& indicators,
    const MeshBase& mesh,
    const AdaptivityOptions& options) const {

  if (markers_.empty()) {
    return std::vector<MarkType>(indicators.size(), MarkType::NONE);
  }

  // Collect marks from all markers
  std::vector<std::vector<MarkType>> all_marks;
  std::vector<double> weights;

  for (const auto& [marker, weight] : markers_) {
    all_marks.push_back(marker->mark(indicators, mesh, options));
    weights.push_back(weight);
  }

  // Combine marks
  auto combined = combine_marks(all_marks, weights);

  // Update statistics
  last_stats_ = MarkerUtils::count_marks(combined);

  return combined;
}

std::vector<MarkType> CompositeMarker::combine_marks(
    const std::vector<std::vector<MarkType>>& all_marks,
    const std::vector<double>& weights) const {

  if (all_marks.empty()) return {};

  size_t num_elems = all_marks[0].size();
  std::vector<MarkType> combined(num_elems, MarkType::NONE);

  switch (combination_method_) {
    case CombinationMethod::UNION:
      for (size_t i = 0; i < num_elems; ++i) {
        for (const auto& marks : all_marks) {
          if (marks[i] == MarkType::REFINE) {
            combined[i] = MarkType::REFINE;
            break;
          } else if (marks[i] == MarkType::COARSEN && combined[i] == MarkType::NONE) {
            combined[i] = MarkType::COARSEN;
          }
        }
      }
      break;

    case CombinationMethod::INTERSECTION:
      for (size_t i = 0; i < num_elems; ++i) {
        MarkType first_mark = all_marks[0][i];
        bool all_agree = true;
        for (size_t j = 1; j < all_marks.size(); ++j) {
          if (all_marks[j][i] != first_mark) {
            all_agree = false;
            break;
          }
        }
        if (all_agree) {
          combined[i] = first_mark;
        }
      }
      break;

    case CombinationMethod::MAJORITY:
      for (size_t i = 0; i < num_elems; ++i) {
        int refine_votes = 0, coarsen_votes = 0, none_votes = 0;
        for (const auto& marks : all_marks) {
          switch (marks[i]) {
            case MarkType::REFINE: refine_votes++; break;
            case MarkType::COARSEN: coarsen_votes++; break;
            case MarkType::NONE: none_votes++; break;
          }
        }
        if (refine_votes > coarsen_votes && refine_votes > none_votes) {
          combined[i] = MarkType::REFINE;
        } else if (coarsen_votes > refine_votes && coarsen_votes > none_votes) {
          combined[i] = MarkType::COARSEN;
        }
      }
      break;

    case CombinationMethod::WEIGHTED:
      for (size_t i = 0; i < num_elems; ++i) {
        double refine_weight = 0.0, coarsen_weight = 0.0;
        for (size_t j = 0; j < all_marks.size(); ++j) {
          switch (all_marks[j][i]) {
            case MarkType::REFINE: refine_weight += weights[j]; break;
            case MarkType::COARSEN: coarsen_weight += weights[j]; break;
            default: break;
          }
        }
        if (refine_weight > coarsen_weight && refine_weight > 0.5) {
          combined[i] = MarkType::REFINE;
        } else if (coarsen_weight > refine_weight && coarsen_weight > 0.5) {
          combined[i] = MarkType::COARSEN;
        }
      }
      break;
  }

  return combined;
}

// ====================
// MarkerFactory Implementation
// ====================

std::unique_ptr<Marker> MarkerFactory::create(const AdaptivityOptions& options) {
  switch (options.marking_strategy) {
    case AdaptivityOptions::MarkingStrategy::FIXED_FRACTION:
      return create_fixed_fraction(options.refine_fraction, options.coarsen_fraction, true);

    case AdaptivityOptions::MarkingStrategy::THRESHOLD_ABSOLUTE:
      return create_threshold(options.refine_threshold, options.coarsen_threshold, false);

    case AdaptivityOptions::MarkingStrategy::THRESHOLD_RELATIVE:
      return create_threshold(options.refine_threshold, options.coarsen_threshold, true);

    case AdaptivityOptions::MarkingStrategy::FIXED_COUNT:
      return create_fixed_count(options.refine_count, options.coarsen_count);

    case AdaptivityOptions::MarkingStrategy::REGION_AWARE: {
      auto base = create_fixed_fraction(options.refine_fraction, options.coarsen_fraction);
      return create_region_aware(std::move(base), options.include_regions, options.exclude_regions);
    }

    default:
      throw std::runtime_error("Unknown marking strategy");
  }
}

std::unique_ptr<Marker> MarkerFactory::create_fixed_fraction(
    double refine_fraction, double coarsen_fraction, bool use_doerfler) {
  FixedFractionMarker::Config config;
  config.refine_fraction = refine_fraction;
  config.coarsen_fraction = coarsen_fraction;
  config.use_doerfler = use_doerfler;
  return std::make_unique<FixedFractionMarker>(config);
}

std::unique_ptr<Marker> MarkerFactory::create_threshold(
    double refine_threshold, double coarsen_threshold, bool relative) {
  ThresholdMarker::Config config;
  config.threshold_type = relative ?
      ThresholdMarker::Config::ThresholdType::RELATIVE :
      ThresholdMarker::Config::ThresholdType::ABSOLUTE;
  config.refine_threshold = refine_threshold;
  config.coarsen_threshold = coarsen_threshold;
  return std::make_unique<ThresholdMarker>(config);
}

std::unique_ptr<Marker> MarkerFactory::create_fixed_count(
    size_t refine_count, size_t coarsen_count) {
  FixedCountMarker::Config config;
  config.refine_count = refine_count;
  config.coarsen_count = coarsen_count;
  return std::make_unique<FixedCountMarker>(config);
}

std::unique_ptr<Marker> MarkerFactory::create_region_aware(
    std::unique_ptr<Marker> base_marker,
    const std::set<int>& include_regions,
    const std::set<int>& exclude_regions) {
  RegionAwareMarker::Config config;
  config.base_marker = std::move(base_marker);
  config.include_regions = include_regions;
  config.exclude_regions = exclude_regions;
  return std::make_unique<RegionAwareMarker>(std::move(config));
}

// ====================
// MarkerUtils Implementation
// ====================

Marker::MarkingStats MarkerUtils::count_marks(const std::vector<MarkType>& marks) {
  Marker::MarkingStats stats;

  for (MarkType mark : marks) {
    switch (mark) {
      case MarkType::REFINE:
        stats.num_marked_refine++;
        break;
      case MarkType::COARSEN:
        stats.num_marked_coarsen++;
        break;
      case MarkType::NONE:
        stats.num_unmarked++;
        break;
    }
  }

  return stats;
}

std::pair<std::vector<bool>, std::vector<bool>> MarkerUtils::marks_to_flags(
    const std::vector<MarkType>& marks) {

  size_t num_elems = marks.size();
  std::vector<bool> refine_flags(num_elems, false);
  std::vector<bool> coarsen_flags(num_elems, false);

  for (size_t i = 0; i < num_elems; ++i) {
    switch (marks[i]) {
      case MarkType::REFINE:
        refine_flags[i] = true;
        break;
      case MarkType::COARSEN:
        coarsen_flags[i] = true;
        break;
      default:
        break;
    }
  }

  return {refine_flags, coarsen_flags};
}

void MarkerUtils::apply_constraints(
    std::vector<MarkType>& marks,
    const MeshBase& mesh,
    const AdaptivityOptions& options) {

  // Apply max level constraint
  for (size_t i = 0; i < marks.size(); ++i) {
    if (marks[i] == MarkType::REFINE) {
      size_t level = mesh.cell_refinement_level(i);
      if (level >= options.max_refinement_level) {
        marks[i] = MarkType::NONE;
      }
    }
  }

  // Apply element count constraints
  auto stats = count_marks(marks);
  size_t predicted_count = mesh.num_cells() - stats.num_marked_coarsen +
                          stats.num_marked_refine * 4;  // Assuming 1:4 refinement

  if (predicted_count > options.max_element_count) {
    // Remove refinement marks
    for (auto& mark : marks) {
      if (mark == MarkType::REFINE) {
        mark = MarkType::NONE;
      }
    }
  }

  if (predicted_count < options.min_element_count) {
    // Remove coarsening marks
    for (auto& mark : marks) {
      if (mark == MarkType::COARSEN) {
        mark = MarkType::NONE;
      }
    }
  }
}

void MarkerUtils::smooth_marking(
    std::vector<MarkType>& marks,
    const MeshBase& mesh,
    int num_smooth_passes) {

  for (int pass = 0; pass < num_smooth_passes; ++pass) {
    std::vector<MarkType> new_marks = marks;

    for (size_t i = 0; i < marks.size(); ++i) {
      auto neighbors = mesh.cell_neighbors(i);

      int refine_neighbors = 0;
      for (size_t neighbor : neighbors) {
        if (neighbor < marks.size() && marks[neighbor] == MarkType::REFINE) {
          refine_neighbors++;
        }
      }

      // If most neighbors are marked for refinement, mark this one too
      if (refine_neighbors > static_cast<int>(neighbors.size()) / 2) {
        if (marks[i] == MarkType::NONE) {
          new_marks[i] = MarkType::REFINE;
        }
      }
    }

    marks = new_marks;
  }
}

void MarkerUtils::write_marks_to_field(
    MeshFields& fields,
    const std::string& field_name,
    const std::vector<MarkType>& marks) {

  auto field = fields.create_field(field_name, FieldLocation::CELL, 1);

  for (size_t i = 0; i < marks.size(); ++i) {
    double value = 0.0;
    switch (marks[i]) {
      case MarkType::REFINE: value = 1.0; break;
      case MarkType::COARSEN: value = -1.0; break;
      case MarkType::NONE: value = 0.0; break;
    }
    field->set_cell_value(i, value);
  }
}

} // namespace svmp