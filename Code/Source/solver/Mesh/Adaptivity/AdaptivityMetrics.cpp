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

#include "AdaptivityMetrics.h"
#include "../MeshBase.h"
#include "QualityGuards.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <sys/resource.h>

namespace svmp {

namespace {

// Helper to get current memory usage
double get_current_memory_mb() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss / 1024.0; // Convert KB to MB
}

// Compute statistics for a vector of values
ErrorMetrics::Distribution compute_distribution(const std::vector<double>& values) {
  ErrorMetrics::Distribution dist;

  if (values.empty()) return dist;

  // Mean
  dist.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

  // Min/Max
  auto minmax = std::minmax_element(values.begin(), values.end());
  dist.min = *minmax.first;
  dist.max = *minmax.second;

  // Median
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  if (sorted.size() % 2 == 0) {
    dist.median = (sorted[sorted.size()/2 - 1] + sorted[sorted.size()/2]) / 2.0;
  } else {
    dist.median = sorted[sorted.size()/2];
  }

  // Standard deviation
  double sum_sq = 0.0;
  for (double v : values) {
    double diff = v - dist.mean;
    sum_sq += diff * diff;
  }
  dist.std_dev = std::sqrt(sum_sq / values.size());

  return dist;
}

} // anonymous namespace

//=============================================================================
// AdaptivityMetricsCollector Implementation
//=============================================================================

AdaptivityMetricsCollector::AdaptivityMetricsCollector(const Config& config)
    : config_(config) {

  if (config_.write_to_file) {
    log_file_.open(config_.output_file);
    if (log_file_.is_open()) {
      write_csv_header();
    }
  }
}

void AdaptivityMetricsCollector::begin_iteration(const MeshBase& mesh) {
  current_entry_ = AdaptationHistoryEntry{};
  current_entry_.iteration = history_.size();
  current_entry_.timestamp = std::chrono::system_clock::now();
  current_entry_.elements_before = mesh.num_elements();

  if (config_.profile_memory) {
    current_entry_.performance.memory_before = get_memory_usage();
  }

  timers_.clear();
  start_timer("total");
}

void AdaptivityMetricsCollector::record_result(
    const MeshBase& mesh,
    const AdaptivityResult& result,
    const AdaptivityOptions& options) {

  stop_timer("total");

  current_entry_.elements_after = mesh.num_elements();
  current_entry_.num_refined = result.num_refined;
  current_entry_.num_coarsened = result.num_coarsened;
  current_entry_.options = options;

  // Collect metrics
  if (config_.profile_performance) {
    current_entry_.performance = collect_performance_metrics();
  }

  if (config_.track_quality) {
    current_entry_.quality = collect_quality_metrics(mesh);
  }

  if (config_.profile_memory) {
    current_entry_.performance.memory_after = get_memory_usage();
    current_entry_.performance.memory_peak = std::max(
        current_entry_.performance.memory_before,
        current_entry_.performance.memory_after);
  }

  // Add to history
  history_.push_back(current_entry_);

  // Limit history size
  if (history_.size() > config_.max_history) {
    history_.erase(history_.begin());
  }

  // Write to file if enabled
  if (config_.write_to_file && log_file_.is_open()) {
    write_csv_row(current_entry_);
  }
}

void AdaptivityMetricsCollector::start_timer(const std::string& phase) {
  timers_[phase] = std::chrono::steady_clock::now();
}

void AdaptivityMetricsCollector::stop_timer(const std::string& phase) {
  auto it = timers_.find(phase);
  if (it != timers_.end()) {
    auto duration = std::chrono::steady_clock::now() - it->second;
    double time = std::chrono::duration<double>(duration).count();

    if (phase == "total") {
      current_entry_.performance.total_time = time;
    } else if (phase == "error_estimation") {
      current_entry_.performance.error_estimation_time = time;
    } else if (phase == "marking") {
      current_entry_.performance.marking_time = time;
    } else if (phase == "refinement") {
      current_entry_.performance.refinement_time = time;
    } else if (phase == "coarsening") {
      current_entry_.performance.coarsening_time = time;
    } else if (phase == "conformity") {
      current_entry_.performance.conformity_time = time;
    } else if (phase == "transfer") {
      current_entry_.performance.transfer_time = time;
    } else if (phase == "quality") {
      current_entry_.performance.quality_time = time;
    } else if (phase == "update") {
      current_entry_.performance.update_time = time;
    }
  }
}

void AdaptivityMetricsCollector::record_memory() {
  if (config_.profile_memory) {
    double current_mem = get_memory_usage();
    current_entry_.performance.memory_peak = std::max(
        current_entry_.performance.memory_peak,
        current_mem);
  }
}

AdaptationHistoryEntry AdaptivityMetricsCollector::get_current_metrics() const {
  return current_entry_;
}

void AdaptivityMetricsCollector::write_report(const std::string& filename) const {
  std::string report_file = filename.empty() ? "adaptivity_report.txt" : filename;
  std::ofstream report(report_file);

  if (!report.is_open()) return;

  report << "Adaptivity Metrics Report\n";
  report << "========================\n\n";

  // Summary statistics
  auto summary = get_summary();
  report << "Summary Statistics:\n";
  report << "  Total iterations: " << summary.total_iterations << "\n";
  report << "  Total time: " << summary.total_time << " s\n";
  report << "  Total refined: " << summary.total_refined << "\n";
  report << "  Total coarsened: " << summary.total_coarsened << "\n";
  report << "  Avg time/iteration: " << summary.avg_time_per_iteration << " s\n";
  report << "  Avg quality improvement: " << summary.avg_quality_improvement << "\n";
  report << "  Avg error reduction: " << summary.avg_error_reduction << "\n\n";

  // Detailed history
  report << "Iteration History:\n";
  report << std::setw(5) << "Iter"
         << std::setw(12) << "Elements"
         << std::setw(10) << "Refined"
         << std::setw(10) << "Coarsened"
         << std::setw(10) << "Time(s)"
         << std::setw(10) << "Quality"
         << "\n";

  for (const auto& entry : history_) {
    report << std::setw(5) << entry.iteration
           << std::setw(12) << entry.elements_after
           << std::setw(10) << entry.num_refined
           << std::setw(10) << entry.num_coarsened
           << std::setw(10) << std::fixed << std::setprecision(3)
           << entry.performance.total_time
           << std::setw(10) << entry.quality.avg_quality_after
           << "\n";
  }

  // Performance breakdown
  if (!history_.empty()) {
    const auto& last = history_.back();
    report << "\nPerformance Breakdown (last iteration):\n";
    report << "  Error estimation: " << last.performance.error_estimation_time << " s\n";
    report << "  Marking: " << last.performance.marking_time << " s\n";
    report << "  Refinement: " << last.performance.refinement_time << " s\n";
    report << "  Coarsening: " << last.performance.coarsening_time << " s\n";
    report << "  Conformity: " << last.performance.conformity_time << " s\n";
    report << "  Field transfer: " << last.performance.transfer_time << " s\n";
    report << "  Quality check: " << last.performance.quality_time << " s\n";
  }
}

void AdaptivityMetricsCollector::clear_history() {
  history_.clear();
}

AdaptivityMetricsCollector::SummaryStats AdaptivityMetricsCollector::get_summary() const {
  SummaryStats summary;

  summary.total_iterations = history_.size();

  for (const auto& entry : history_) {
    summary.total_time += entry.performance.total_time;
    summary.total_refined += entry.num_refined;
    summary.total_coarsened += entry.num_coarsened;

    if (entry.quality.improvement_factor > 0) {
      summary.avg_quality_improvement += entry.quality.improvement_factor;
    }

    if (entry.error.error_reduction > 0) {
      summary.avg_error_reduction += entry.error.error_reduction;
    }
  }

  if (summary.total_iterations > 0) {
    summary.avg_time_per_iteration = summary.total_time / summary.total_iterations;
    summary.avg_quality_improvement /= summary.total_iterations;
    summary.avg_error_reduction /= summary.total_iterations;
  }

  return summary;
}

PerformanceMetrics AdaptivityMetricsCollector::collect_performance_metrics() {
  PerformanceMetrics metrics = current_entry_.performance;

  // Timer-based metrics are already set via stop_timer
  // Add any additional performance collection here

  return metrics;
}

QualityMetrics AdaptivityMetricsCollector::collect_quality_metrics(const MeshBase& mesh) {
  QualityMetrics metrics;

  // Use quality checker to evaluate mesh
  GeometricQualityChecker checker;
  QualityOptions quality_opts;

  auto mesh_quality = checker.compute_mesh_quality(mesh, quality_opts);

  metrics.min_quality_after = mesh_quality.min_quality;
  metrics.avg_quality_after = mesh_quality.avg_quality;
  metrics.poor_elements_after = mesh_quality.num_poor_elements;

  // Compute improvement (need before metrics)
  if (!history_.empty()) {
    const auto& prev = history_.back();
    metrics.min_quality_before = prev.quality.min_quality_after;
    metrics.avg_quality_before = prev.quality.avg_quality_after;
    metrics.poor_elements_before = prev.quality.poor_elements_after;

    if (metrics.avg_quality_before > 0) {
      metrics.improvement_factor = metrics.avg_quality_after / metrics.avg_quality_before;
    }
  }

  return metrics;
}

ErrorMetrics AdaptivityMetricsCollector::collect_error_metrics(
    const std::vector<double>& error_before,
    const std::vector<double>& error_after) {

  ErrorMetrics metrics;

  if (!error_before.empty()) {
    metrics.max_error_before = *std::max_element(error_before.begin(), error_before.end());
    metrics.total_error_before = std::accumulate(error_before.begin(), error_before.end(), 0.0);
    metrics.distribution_before = compute_distribution(error_before);
  }

  if (!error_after.empty()) {
    metrics.max_error_after = *std::max_element(error_after.begin(), error_after.end());
    metrics.total_error_after = std::accumulate(error_after.begin(), error_after.end(), 0.0);
    metrics.distribution_after = compute_distribution(error_after);
  }

  if (metrics.total_error_before > 0) {
    metrics.error_reduction = 1.0 - (metrics.total_error_after / metrics.total_error_before);
  }

  return metrics;
}

double AdaptivityMetricsCollector::get_memory_usage() const {
  return get_current_memory_mb();
}

void AdaptivityMetricsCollector::write_csv_header() {
  log_file_ << "iteration,timestamp,elements_before,elements_after,"
            << "refined,coarsened,total_time,error_time,marking_time,"
            << "refinement_time,quality_min,quality_avg,memory_mb\n";
}

void AdaptivityMetricsCollector::write_csv_row(const AdaptationHistoryEntry& entry) {
  auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);

  log_file_ << entry.iteration << ","
            << time_t << ","
            << entry.elements_before << ","
            << entry.elements_after << ","
            << entry.num_refined << ","
            << entry.num_coarsened << ","
            << entry.performance.total_time << ","
            << entry.performance.error_estimation_time << ","
            << entry.performance.marking_time << ","
            << entry.performance.refinement_time << ","
            << entry.quality.min_quality_after << ","
            << entry.quality.avg_quality_after << ","
            << entry.performance.memory_after << "\n";

  log_file_.flush();
}

//=============================================================================
// ConvergenceMonitor Implementation
//=============================================================================

ConvergenceMonitor::ConvergenceMonitor(const Config& config)
    : config_(config) {}

void ConvergenceMonitor::update(const AdaptationHistoryEntry& entry) {
  recent_history_.push_back(entry);

  // Keep only recent history
  if (recent_history_.size() > config_.window_size) {
    recent_history_.erase(recent_history_.begin());
  }

  // Check convergence
  converged_ = false;
  convergence_reason_.clear();

  if (recent_history_.size() < 2) {
    return; // Not enough data
  }

  // Check various convergence criteria
  switch (config_.criterion) {
    case Config::Criterion::ERROR_REDUCTION:
      converged_ = check_error_convergence();
      if (converged_) convergence_reason_ = "Error tolerance reached";
      break;

    case Config::Criterion::ELEMENT_CHANGE:
      converged_ = check_element_convergence();
      if (converged_) convergence_reason_ = "Element count stabilized";
      break;

    case Config::Criterion::QUALITY_THRESHOLD:
      converged_ = check_quality_convergence();
      if (converged_) convergence_reason_ = "Quality threshold reached";
      break;

    case Config::Criterion::MAX_ITERATIONS:
      converged_ = (entry.iteration >= config_.max_iterations);
      if (converged_) convergence_reason_ = "Maximum iterations reached";
      break;

    case Config::Criterion::COMBINED:
      if (check_error_convergence()) {
        converged_ = true;
        convergence_reason_ = "Error tolerance reached";
      } else if (check_element_convergence()) {
        converged_ = true;
        convergence_reason_ = "Element count stabilized";
      } else if (check_quality_convergence()) {
        converged_ = true;
        convergence_reason_ = "Quality threshold reached";
      } else if (entry.iteration >= config_.max_iterations) {
        converged_ = true;
        convergence_reason_ = "Maximum iterations reached";
      }
      break;
  }

  // Check for stagnation/oscillation
  if (!converged_ && config_.detect_stagnation && is_stagnating()) {
    converged_ = true;
    convergence_reason_ = "Stagnation detected";
  }

  if (!converged_ && config_.detect_oscillation && is_oscillating()) {
    converged_ = true;
    convergence_reason_ = "Oscillation detected";
  }
}

bool ConvergenceMonitor::is_converged() const {
  return converged_;
}

std::string ConvergenceMonitor::get_convergence_reason() const {
  return convergence_reason_;
}

double ConvergenceMonitor::get_convergence_rate() const {
  if (recent_history_.size() < 2) return 0.0;

  // Compute error reduction rate
  std::vector<double> errors;
  for (const auto& entry : recent_history_) {
    errors.push_back(entry.error.max_error_after);
  }

  return compute_trend(errors);
}

size_t ConvergenceMonitor::predict_iterations_to_convergence() const {
  if (converged_) return 0;

  double rate = get_convergence_rate();
  if (rate <= 0) return config_.max_iterations;

  // Estimate based on current rate
  const auto& current = recent_history_.back();
  double current_error = current.error.max_error_after;

  if (current_error <= 0) return 0;

  size_t iterations = static_cast<size_t>(
      std::log(config_.error_tolerance / current_error) / std::log(rate));

  return std::min(iterations, config_.max_iterations - current.iteration);
}

void ConvergenceMonitor::reset() {
  recent_history_.clear();
  converged_ = false;
  convergence_reason_.clear();
}

bool ConvergenceMonitor::check_error_convergence() const {
  if (recent_history_.empty()) return false;

  const auto& last = recent_history_.back();
  return last.error.max_error_after < config_.error_tolerance;
}

bool ConvergenceMonitor::check_element_convergence() const {
  if (recent_history_.size() < 2) return false;

  const auto& prev = recent_history_[recent_history_.size() - 2];
  const auto& last = recent_history_.back();

  double change = std::abs(static_cast<double>(last.elements_after - prev.elements_after)) /
                  static_cast<double>(prev.elements_after + 1);

  return change < config_.element_tolerance;
}

bool ConvergenceMonitor::check_quality_convergence() const {
  if (recent_history_.empty()) return false;

  const auto& last = recent_history_.back();
  return last.quality.avg_quality_after >= config_.quality_threshold;
}

bool ConvergenceMonitor::is_stagnating() const {
  if (recent_history_.size() < config_.window_size) return false;

  // Check if error is not decreasing
  std::vector<double> errors;
  for (const auto& entry : recent_history_) {
    errors.push_back(entry.error.max_error_after);
  }

  double trend = compute_trend(errors);
  return std::abs(trend) < 1e-6;
}

bool ConvergenceMonitor::is_oscillating() const {
  if (recent_history_.size() < config_.window_size) return false;

  // Check for oscillation in element counts
  int sign_changes = 0;
  int prev_sign = 0;

  for (size_t i = 1; i < recent_history_.size(); ++i) {
    int diff = recent_history_[i].elements_after - recent_history_[i-1].elements_after;
    int sign = (diff > 0) ? 1 : ((diff < 0) ? -1 : 0);

    if (sign != 0 && prev_sign != 0 && sign != prev_sign) {
      sign_changes++;
    }
    prev_sign = sign;
  }

  return sign_changes >= 2;
}

double ConvergenceMonitor::compute_trend(const std::vector<double>& values) const {
  if (values.size() < 2) return 0.0;

  // Simple linear trend
  double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
  size_t n = values.size();

  for (size_t i = 0; i < n; ++i) {
    double x = static_cast<double>(i);
    double y = values[i];
    sum_x += x;
    sum_y += y;
    sum_xy += x * y;
    sum_xx += x * x;
  }

  double denom = n * sum_xx - sum_x * sum_x;
  if (std::abs(denom) < 1e-12) return 0.0;

  return (n * sum_xy - sum_x * sum_y) / denom;
}

//=============================================================================
// AdaptivityProfiler Implementation
//=============================================================================

void AdaptivityProfiler::start_profiling() {
  profiling_ = true;
  events_.clear();
  profile_start_ = std::chrono::steady_clock::now();
}

void AdaptivityProfiler::stop_profiling() {
  profiling_ = false;
}

void AdaptivityProfiler::record_event(const Event& event) {
  if (profiling_) {
    events_.push_back(event);
  }
}

AdaptivityProfiler::ProfileReport AdaptivityProfiler::get_report() const {
  ProfileReport report;

  for (const auto& event : events_) {
    double duration = std::chrono::duration<double>(event.end - event.start).count();

    report.function_times[event.name] += duration;
    report.type_times[event.type] += duration;
    report.total_time += duration;
    report.total_memory += event.memory_allocated;
    report.total_flops += event.flop_count;
  }

  report.hotspots = find_hotspots(10);

  return report;
}

void AdaptivityProfiler::write_profile(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) return;

  auto report = get_report();

  file << "Adaptivity Performance Profile\n";
  file << "==============================\n\n";

  file << "Total time: " << report.total_time << " s\n";
  file << "Total memory: " << report.total_memory / (1024.0 * 1024.0) << " MB\n";
  file << "Total FLOPs: " << report.total_flops << "\n\n";

  file << "Function Times:\n";
  for (const auto& [func, time] : report.function_times) {
    file << "  " << func << ": " << time << " s ("
         << (100.0 * time / report.total_time) << "%)\n";
  }

  file << "\nHotspots:\n";
  for (const auto& event : report.hotspots) {
    double duration = std::chrono::duration<double>(event.end - event.start).count();
    file << "  " << event.name << ": " << duration << " s\n";
  }
}

std::vector<AdaptivityProfiler::Event> AdaptivityProfiler::find_hotspots(size_t count) const {
  std::vector<Event> sorted_events = events_;

  std::sort(sorted_events.begin(), sorted_events.end(),
            [](const Event& a, const Event& b) {
              auto a_duration = a.end - a.start;
              auto b_duration = b.end - b.start;
              return a_duration > b_duration;
            });

  if (sorted_events.size() > count) {
    sorted_events.resize(count);
  }

  return sorted_events;
}

//=============================================================================
// MetricsVisualizer Implementation
//=============================================================================

void MetricsVisualizer::write_convergence_plot(
    const std::vector<AdaptationHistoryEntry>& history,
    const std::string& filename) {

  std::ofstream file(filename);
  if (!file.is_open()) return;

  file << "# Convergence plot data\n";
  file << "# iteration error elements quality time\n";

  for (const auto& entry : history) {
    file << entry.iteration << " "
         << entry.error.max_error_after << " "
         << entry.elements_after << " "
         << entry.quality.avg_quality_after << " "
         << entry.performance.total_time << "\n";
  }
}

void MetricsVisualizer::write_quality_histogram(
    const QualityMetrics& metrics,
    const std::string& filename) {

  std::ofstream file(filename);
  if (!file.is_open()) return;

  file << "# Quality histogram data\n";
  file << "# metric before after\n";

  file << "min_quality " << metrics.min_quality_before << " "
       << metrics.min_quality_after << "\n";
  file << "avg_quality " << metrics.avg_quality_before << " "
       << metrics.avg_quality_after << "\n";
  file << "poor_elements " << metrics.poor_elements_before << " "
       << metrics.poor_elements_after << "\n";
}

void MetricsVisualizer::write_error_distribution(
    const ErrorMetrics& metrics,
    const std::string& filename) {

  std::ofstream file(filename);
  if (!file.is_open()) return;

  file << "# Error distribution data\n";
  file << "# statistic before after\n";

  file << "mean " << metrics.distribution_before.mean << " "
       << metrics.distribution_after.mean << "\n";
  file << "std_dev " << metrics.distribution_before.std_dev << " "
       << metrics.distribution_after.std_dev << "\n";
  file << "min " << metrics.distribution_before.min << " "
       << metrics.distribution_after.min << "\n";
  file << "max " << metrics.distribution_before.max << " "
       << metrics.distribution_after.max << "\n";
  file << "median " << metrics.distribution_before.median << " "
       << metrics.distribution_after.median << "\n";
}

void MetricsVisualizer::write_performance_breakdown(
    const PerformanceMetrics& metrics,
    const std::string& filename) {

  std::ofstream file(filename);
  if (!file.is_open()) return;

  file << "# Performance breakdown\n";
  file << "# phase time percentage\n";

  double total = metrics.total_time;
  if (total <= 0) total = 1.0;

  file << "error_estimation " << metrics.error_estimation_time << " "
       << (100.0 * metrics.error_estimation_time / total) << "\n";
  file << "marking " << metrics.marking_time << " "
       << (100.0 * metrics.marking_time / total) << "\n";
  file << "refinement " << metrics.refinement_time << " "
       << (100.0 * metrics.refinement_time / total) << "\n";
  file << "coarsening " << metrics.coarsening_time << " "
       << (100.0 * metrics.coarsening_time / total) << "\n";
  file << "conformity " << metrics.conformity_time << " "
       << (100.0 * metrics.conformity_time / total) << "\n";
  file << "transfer " << metrics.transfer_time << " "
       << (100.0 * metrics.transfer_time / total) << "\n";
  file << "quality " << metrics.quality_time << " "
       << (100.0 * metrics.quality_time / total) << "\n";
}

void MetricsVisualizer::generate_html_report(
    const std::vector<AdaptationHistoryEntry>& history,
    const std::string& filename) {

  std::ofstream file(filename);
  if (!file.is_open()) return;

  file << "<!DOCTYPE html>\n";
  file << "<html><head><title>Adaptivity Report</title></head>\n";
  file << "<body><h1>Adaptivity Metrics Report</h1>\n";

  // Summary table
  file << "<h2>Summary</h2>\n";
  file << "<table border='1'>\n";
  file << "<tr><th>Metric</th><th>Value</th></tr>\n";
  file << "<tr><td>Total iterations</td><td>" << history.size() << "</td></tr>\n";

  if (!history.empty()) {
    const auto& last = history.back();
    file << "<tr><td>Final elements</td><td>" << last.elements_after << "</td></tr>\n";
    file << "<tr><td>Final quality</td><td>" << last.quality.avg_quality_after << "</td></tr>\n";
    file << "<tr><td>Final error</td><td>" << last.error.max_error_after << "</td></tr>\n";
  }

  file << "</table>\n";

  // Iteration details
  file << "<h2>Iterations</h2>\n";
  file << "<table border='1'>\n";
  file << "<tr><th>Iter</th><th>Elements</th><th>Refined</th>"
       << "<th>Coarsened</th><th>Time(s)</th><th>Quality</th></tr>\n";

  for (const auto& entry : history) {
    file << "<tr>";
    file << "<td>" << entry.iteration << "</td>";
    file << "<td>" << entry.elements_after << "</td>";
    file << "<td>" << entry.num_refined << "</td>";
    file << "<td>" << entry.num_coarsened << "</td>";
    file << "<td>" << std::fixed << std::setprecision(3)
         << entry.performance.total_time << "</td>";
    file << "<td>" << entry.quality.avg_quality_after << "</td>";
    file << "</tr>\n";
  }

  file << "</table>\n";
  file << "</body></html>\n";
}

//=============================================================================
// AdaptivityMetricsUtils Implementation
//=============================================================================

AdaptivityMetricsUtils::EfficiencyMetrics AdaptivityMetricsUtils::compute_efficiency(
    const PerformanceMetrics& serial,
    const PerformanceMetrics& parallel,
    int num_processors) {

  EfficiencyMetrics metrics;

  metrics.speedup = serial.total_time / parallel.total_time;
  metrics.efficiency = metrics.speedup / num_processors;

  // Simplified scalability metric
  metrics.scalability = metrics.efficiency;

  // Load balance (simplified)
  metrics.load_balance = 0.9; // Placeholder

  return metrics;
}

AdaptivityOptions AdaptivityMetricsUtils::estimate_optimal_parameters(
    const std::vector<AdaptationHistoryEntry>& history) {

  AdaptivityOptions options;

  if (history.empty()) return options;

  // Analyze history to find best parameters
  double best_quality = 0.0;
  size_t best_index = 0;

  for (size_t i = 0; i < history.size(); ++i) {
    if (history[i].quality.avg_quality_after > best_quality) {
      best_quality = history[i].quality.avg_quality_after;
      best_index = i;
    }
  }

  // Return parameters from best iteration
  return history[best_index].options;
}

AdaptivityMetricsUtils::ResourcePrediction AdaptivityMetricsUtils::predict_resources(
    const MeshBase& mesh,
    const AdaptivityOptions& options,
    const std::vector<AdaptationHistoryEntry>& history) {

  ResourcePrediction prediction;

  // Simple predictions based on history
  if (!history.empty()) {
    double avg_time = 0.0;
    double avg_memory = 0.0;

    for (const auto& entry : history) {
      avg_time += entry.performance.total_time;
      avg_memory += entry.performance.memory_peak;
    }

    avg_time /= history.size();
    avg_memory /= history.size();

    // Scale based on current mesh size
    double scale_factor = mesh.num_elements() / static_cast<double>(history.back().elements_after);

    prediction.estimated_time = avg_time * scale_factor;
    prediction.estimated_memory = avg_memory * scale_factor;
    prediction.estimated_iterations = options.max_iterations;
  } else {
    // Default predictions
    prediction.estimated_time = 10.0;
    prediction.estimated_memory = 1024.0;
    prediction.estimated_iterations = 10;
  }

  return prediction;
}

void AdaptivityMetricsUtils::compare_strategies(
    const std::map<std::string, std::vector<AdaptationHistoryEntry>>& results,
    const std::string& report_file) {

  std::ofstream report(report_file);
  if (!report.is_open()) return;

  report << "Strategy Comparison Report\n";
  report << "=========================\n\n";

  for (const auto& [strategy, history] : results) {
    report << "Strategy: " << strategy << "\n";

    if (!history.empty()) {
      const auto& last = history.back();
      report << "  Final elements: " << last.elements_after << "\n";
      report << "  Final quality: " << last.quality.avg_quality_after << "\n";
      report << "  Final error: " << last.error.max_error_after << "\n";
      report << "  Total time: " << last.performance.total_time << "\n";
    }

    report << "\n";
  }
}

} // namespace svmp