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

#ifndef SVMP_ADAPTIVITY_METRICS_H
#define SVMP_ADAPTIVITY_METRICS_H

#include "Options.h"
#include "AdaptivityManager.h"
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace svmp {

// Forward declarations
class MeshBase;

/**
 * @brief Performance metrics for adaptivity operations
 */
struct PerformanceMetrics {
  /** Total adaptation time */
  double total_time = 0.0;

  /** Error estimation time */
  double error_estimation_time = 0.0;

  /** Marking time */
  double marking_time = 0.0;

  /** Refinement time */
  double refinement_time = 0.0;

  /** Coarsening time */
  double coarsening_time = 0.0;

  /** Conformity enforcement time */
  double conformity_time = 0.0;

  /** Field transfer time */
  double transfer_time = 0.0;

  /** Quality checking time */
  double quality_time = 0.0;

  /** Mesh update time */
  double update_time = 0.0;

  /** Memory usage before (MB) */
  double memory_before = 0.0;

  /** Memory usage after (MB) */
  double memory_after = 0.0;

  /** Peak memory usage (MB) */
  double memory_peak = 0.0;

  /** Number of cache misses */
  size_t cache_misses = 0;

  /** Floating point operations */
  size_t flop_count = 0;
};

/**
 * @brief Mesh quality metrics
 */
struct QualityMetrics {
  /** Minimum element quality before */
  double min_quality_before = 0.0;

  /** Minimum element quality after */
  double min_quality_after = 0.0;

  /** Average element quality before */
  double avg_quality_before = 0.0;

  /** Average element quality after */
  double avg_quality_after = 0.0;

  /** Number of poor quality elements before */
  size_t poor_elements_before = 0;

  /** Number of poor quality elements after */
  size_t poor_elements_after = 0;

  /** Quality improvement factor */
  double improvement_factor = 0.0;
};

/**
 * @brief Error reduction metrics
 */
struct ErrorMetrics {
  /** Maximum error before */
  double max_error_before = 0.0;

  /** Maximum error after */
  double max_error_after = 0.0;

  /** Total error before */
  double total_error_before = 0.0;

  /** Total error after */
  double total_error_after = 0.0;

  /** Error reduction ratio */
  double error_reduction = 0.0;

  /** Error distribution statistics */
  struct Distribution {
    double mean = 0.0;
    double std_dev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
  };

  Distribution distribution_before;
  Distribution distribution_after;
};

/**
 * @brief Adaptation history entry
 */
struct AdaptationHistoryEntry {
  /** Iteration number */
  size_t iteration = 0;

  /** Timestamp */
  std::chrono::system_clock::time_point timestamp;

  /** Number of elements before */
  size_t elements_before = 0;

  /** Number of elements after */
  size_t elements_after = 0;

  /** Number of refined elements */
  size_t num_refined = 0;

  /** Number of coarsened elements */
  size_t num_coarsened = 0;

  /** Performance metrics */
  PerformanceMetrics performance;

  /** Quality metrics */
  QualityMetrics quality;

  /** Error metrics */
  ErrorMetrics error;

  /** Adaptation parameters used */
  AdaptivityOptions options;
};

/**
 * @brief Metrics collector for adaptivity
 */
class AdaptivityMetricsCollector {
public:
  /**
   * @brief Configuration for metrics collection
   */
  struct Config {
    /** Enable performance profiling */
    bool profile_performance = true;

    /** Enable quality tracking */
    bool track_quality = true;

    /** Enable error tracking */
    bool track_error = true;

    /** Enable memory profiling */
    bool profile_memory = true;

    /** Write metrics to file */
    bool write_to_file = false;

    /** Output filename */
    std::string output_file = "adaptivity_metrics.csv";

    /** Verbose output */
    bool verbose = false;

    /** History size limit */
    size_t max_history = 100;
  };

  explicit AdaptivityMetricsCollector(const Config& config = {});

  /**
   * @brief Start new adaptation iteration
   */
  void begin_iteration(const MeshBase& mesh);

  /**
   * @brief Record adaptation result
   */
  void record_result(
      const MeshBase& mesh,
      const AdaptivityResult& result,
      const AdaptivityOptions& options);

  /**
   * @brief Start timing a phase
   */
  void start_timer(const std::string& phase);

  /**
   * @brief Stop timing a phase
   */
  void stop_timer(const std::string& phase);

  /**
   * @brief Record memory usage
   */
  void record_memory();

  /**
   * @brief Get current metrics
   */
  AdaptationHistoryEntry get_current_metrics() const;

  /**
   * @brief Get full history
   */
  std::vector<AdaptationHistoryEntry> get_history() const { return history_; }

  /**
   * @brief Write metrics report
   */
  void write_report(const std::string& filename = "") const;

  /**
   * @brief Clear history
   */
  void clear_history();

  /**
   * @brief Get summary statistics
   */
  struct SummaryStats {
    size_t total_iterations = 0;
    double total_time = 0.0;
    size_t total_refined = 0;
    size_t total_coarsened = 0;
    double avg_quality_improvement = 0.0;
    double avg_error_reduction = 0.0;
    double avg_time_per_iteration = 0.0;
  };

  SummaryStats get_summary() const;

private:
  Config config_;
  std::vector<AdaptationHistoryEntry> history_;
  AdaptationHistoryEntry current_entry_;
  std::map<std::string, std::chrono::steady_clock::time_point> timers_;
  std::ofstream log_file_;

  /** Collect performance metrics */
  PerformanceMetrics collect_performance_metrics();

  /** Collect quality metrics */
  QualityMetrics collect_quality_metrics(const MeshBase& mesh);

  /** Collect error metrics */
  ErrorMetrics collect_error_metrics(
      const std::vector<double>& error_before,
      const std::vector<double>& error_after);

  /** Get memory usage in MB */
  double get_memory_usage() const;

  /** Write CSV header */
  void write_csv_header();

  /** Write CSV row */
  void write_csv_row(const AdaptationHistoryEntry& entry);
};

/**
 * @brief Convergence monitor for adaptive iterations
 */
class ConvergenceMonitor {
public:
  /**
   * @brief Configuration for convergence monitoring
   */
  struct Config {
    /** Convergence criteria */
    enum class Criterion {
      ERROR_REDUCTION,    // Based on error reduction
      ELEMENT_CHANGE,     // Based on element count change
      QUALITY_THRESHOLD,  // Based on quality improvement
      MAX_ITERATIONS,     // Maximum iteration count
      COMBINED            // Combination of criteria
    };

    Criterion criterion = Criterion::COMBINED;

    /** Error tolerance */
    double error_tolerance = 1e-3;

    /** Element change tolerance */
    double element_tolerance = 0.01;

    /** Quality threshold */
    double quality_threshold = 0.7;

    /** Maximum iterations */
    size_t max_iterations = 20;

    /** Window size for averaging */
    size_t window_size = 3;

    /** Stagnation detection */
    bool detect_stagnation = true;

    /** Oscillation detection */
    bool detect_oscillation = true;
  };

  explicit ConvergenceMonitor(const Config& config = {});

  /**
   * @brief Update monitor with new iteration data
   */
  void update(const AdaptationHistoryEntry& entry);

  /**
   * @brief Check if converged
   */
  bool is_converged() const;

  /**
   * @brief Get convergence reason
   */
  std::string get_convergence_reason() const;

  /**
   * @brief Get convergence rate
   */
  double get_convergence_rate() const;

  /**
   * @brief Predict iterations to convergence
   */
  size_t predict_iterations_to_convergence() const;

  /**
   * @brief Reset monitor
   */
  void reset();

private:
  Config config_;
  std::vector<AdaptationHistoryEntry> recent_history_;
  bool converged_ = false;
  std::string convergence_reason_;

  /** Check error-based convergence */
  bool check_error_convergence() const;

  /** Check element-based convergence */
  bool check_element_convergence() const;

  /** Check quality-based convergence */
  bool check_quality_convergence() const;

  /** Detect stagnation */
  bool is_stagnating() const;

  /** Detect oscillation */
  bool is_oscillating() const;

  /** Compute trend */
  double compute_trend(const std::vector<double>& values) const;
};

/**
 * @brief Performance profiler for adaptivity
 */
class AdaptivityProfiler {
public:
  /**
   * @brief Profile event types
   */
  enum class EventType {
    FUNCTION_CALL,
    MEMORY_ALLOCATION,
    COMMUNICATION,
    FILE_IO,
    COMPUTATION
  };

  /**
   * @brief Profile event
   */
  struct Event {
    EventType type;
    std::string name;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    size_t memory_allocated = 0;
    size_t flop_count = 0;
  };

  /**
   * @brief Start profiling
   */
  void start_profiling();

  /**
   * @brief Stop profiling
   */
  void stop_profiling();

  /**
   * @brief Record event
   */
  void record_event(const Event& event);

  /**
   * @brief Get profile report
   */
  struct ProfileReport {
    std::map<std::string, double> function_times;
    std::map<EventType, double> type_times;
    double total_time = 0.0;
    size_t total_memory = 0;
    size_t total_flops = 0;
    std::vector<Event> hotspots;
  };

  ProfileReport get_report() const;

  /**
   * @brief Write profile to file
   */
  void write_profile(const std::string& filename) const;

private:
  bool profiling_ = false;
  std::vector<Event> events_;
  std::chrono::steady_clock::time_point profile_start_;

  /** Analyze hotspots */
  std::vector<Event> find_hotspots(size_t count = 10) const;

  /** Build call graph */
  void build_call_graph();
};

/**
 * @brief Metrics visualization helper
 */
class MetricsVisualizer {
public:
  /**
   * @brief Write convergence plot data
   */
  static void write_convergence_plot(
      const std::vector<AdaptationHistoryEntry>& history,
      const std::string& filename);

  /**
   * @brief Write quality histogram data
   */
  static void write_quality_histogram(
      const QualityMetrics& metrics,
      const std::string& filename);

  /**
   * @brief Write error distribution data
   */
  static void write_error_distribution(
      const ErrorMetrics& metrics,
      const std::string& filename);

  /**
   * @brief Write performance breakdown
   */
  static void write_performance_breakdown(
      const PerformanceMetrics& metrics,
      const std::string& filename);

  /**
   * @brief Generate HTML report
   */
  static void generate_html_report(
      const std::vector<AdaptationHistoryEntry>& history,
      const std::string& filename);
};

/**
 * @brief Adaptivity metrics utilities
 */
class AdaptivityMetricsUtils {
public:
  /**
   * @brief Compute efficiency metrics
   */
  struct EfficiencyMetrics {
    double speedup;           // Parallel speedup
    double efficiency;        // Parallel efficiency
    double scalability;       // Weak/strong scaling
    double load_balance;      // Load balance factor
  };

  static EfficiencyMetrics compute_efficiency(
      const PerformanceMetrics& serial,
      const PerformanceMetrics& parallel,
      int num_processors);

  /**
   * @brief Estimate optimal parameters
   */
  static AdaptivityOptions estimate_optimal_parameters(
      const std::vector<AdaptationHistoryEntry>& history);

  /**
   * @brief Predict resource requirements
   */
  struct ResourcePrediction {
    double estimated_time;
    double estimated_memory;
    size_t estimated_iterations;
  };

  static ResourcePrediction predict_resources(
      const MeshBase& mesh,
      const AdaptivityOptions& options,
      const std::vector<AdaptationHistoryEntry>& history);

  /**
   * @brief Compare adaptation strategies
   */
  static void compare_strategies(
      const std::map<std::string, std::vector<AdaptationHistoryEntry>>& results,
      const std::string& report_file);
};

} // namespace svmp

#endif // SVMP_ADAPTIVITY_METRICS_H