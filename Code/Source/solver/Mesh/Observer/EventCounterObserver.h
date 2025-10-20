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

#ifndef SVMP_EVENT_COUNTER_OBSERVER_H
#define SVMP_EVENT_COUNTER_OBSERVER_H

#include "MeshObserver.h"
#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace svmp {

/**
 * @brief Extended event counter with histograms and timing statistics
 *
 * Provides comprehensive event counting and analysis:
 * - Count events by type
 * - Track timing between events
 * - Generate histograms of event frequency
 * - Create snapshots for comparison
 * - Export statistics for benchmarking
 */
class EventCounterObserver : public MeshObserver {
public:
  /**
   * @brief Event statistics for a single event type
   */
  struct EventStats {
    size_t count = 0;
    std::chrono::steady_clock::time_point first_time;
    std::chrono::steady_clock::time_point last_time;
    std::chrono::milliseconds total_interval{0};
    std::chrono::milliseconds min_interval{std::chrono::milliseconds::max()};
    std::chrono::milliseconds max_interval{0};
    std::deque<std::chrono::steady_clock::time_point> recent_times;

    void reset();
    double average_interval_ms() const;
    double events_per_second() const;
  };

  /**
   * @brief Histogram bin for event frequency analysis
   */
  struct HistogramBin {
    std::chrono::milliseconds lower_bound;
    std::chrono::milliseconds upper_bound;
    size_t count = 0;
  };

  /**
   * @brief Snapshot of counter state
   */
  struct Snapshot {
    std::string name;
    std::chrono::steady_clock::time_point timestamp;
    std::map<MeshEvent, EventStats> stats;
    size_t total_events = 0;

    std::string summary() const;
  };

  /**
   * @brief Configuration for the counter
   */
  struct Config {
    size_t max_recent_events = 100;  // Max events to track for recent history
    bool track_intervals = true;      // Track timing between events
    bool enable_histograms = false;   // Generate interval histograms
    size_t histogram_bins = 10;       // Number of histogram bins
  };

  /**
   * @brief Default constructor
   */
  EventCounterObserver();

  /**
   * @brief Constructor with configuration
   */
  explicit EventCounterObserver(const Config& config);

  /**
   * @brief Handle mesh event
   */
  void on_mesh_event(MeshEvent event) override;

  /**
   * @brief Get observer name
   */
  const char* observer_name() const override { return "EventCounterObserver"; }

  // Query methods

  /**
   * @brief Get total count across all events
   */
  size_t total_count() const { return total_events_; }

  /**
   * @brief Get count for specific event
   */
  size_t count(MeshEvent event) const;

  /**
   * @brief Get statistics for specific event
   */
  const EventStats& stats(MeshEvent event) const;

  /**
   * @brief Get all statistics
   */
  const std::map<MeshEvent, EventStats>& all_stats() const { return event_stats_; }

  /**
   * @brief Check if any events have been recorded
   */
  bool has_events() const { return total_events_ > 0; }

  // Histogram methods

  /**
   * @brief Generate histogram for event intervals
   *
   * @param event Event type to analyze
   * @param num_bins Number of histogram bins
   * @return Vector of histogram bins
   */
  std::vector<HistogramBin> generate_histogram(MeshEvent event, size_t num_bins = 10) const;

  /**
   * @brief Generate combined histogram for all events
   */
  std::vector<HistogramBin> generate_combined_histogram(size_t num_bins = 10) const;

  /**
   * @brief Format histogram as string
   */
  std::string format_histogram(const std::vector<HistogramBin>& histogram,
                              const std::string& title = "Event Interval Histogram") const;

  // Snapshot methods

  /**
   * @brief Create a snapshot of current state
   *
   * @param name Optional name for the snapshot
   * @return Snapshot object
   */
  Snapshot create_snapshot(const std::string& name = "") const;

  /**
   * @brief Store a snapshot for later comparison
   */
  void store_snapshot(const std::string& name);

  /**
   * @brief Get stored snapshot by name
   */
  const Snapshot* get_snapshot(const std::string& name) const;

  /**
   * @brief List all stored snapshot names
   */
  std::vector<std::string> list_snapshots() const;

  /**
   * @brief Compare two snapshots
   */
  std::string compare_snapshots(const Snapshot& s1, const Snapshot& s2) const;

  /**
   * @brief Compare current state with a snapshot
   */
  std::string compare_with_snapshot(const std::string& snapshot_name) const;

  // Reset and reporting

  /**
   * @brief Reset all counters
   */
  void reset();

  /**
   * @brief Reset counters for specific event
   */
  void reset_event(MeshEvent event);

  /**
   * @brief Generate detailed report
   */
  std::string detailed_report() const;

  /**
   * @brief Generate summary report
   */
  std::string summary_report() const;

  /**
   * @brief Export statistics as CSV
   */
  std::string export_csv() const;

  /**
   * @brief Export statistics as JSON
   */
  std::string export_json() const;

  // Configuration

  /**
   * @brief Set maximum recent events to track
   */
  void set_max_recent_events(size_t max) {
    config_.max_recent_events = max;
    trim_recent_events();
  }

  /**
   * @brief Enable/disable interval tracking
   */
  void set_track_intervals(bool enabled) {
    config_.track_intervals = enabled;
  }

  /**
   * @brief Enable/disable histogram generation
   */
  void set_enable_histograms(bool enabled) {
    config_.enable_histograms = enabled;
  }

private:
  /**
   * @brief Update statistics for an event
   */
  void update_stats(MeshEvent event);

  /**
   * @brief Trim recent event lists to max size
   */
  void trim_recent_events();

  /**
   * @brief Calculate percentile from sorted intervals
   */
  double calculate_percentile(const std::vector<double>& sorted_intervals,
                             double percentile) const;

  Config config_;
  std::map<MeshEvent, EventStats> event_stats_;
  size_t total_events_ = 0;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_event_time_;

  // Stored snapshots
  std::map<std::string, Snapshot> snapshots_;

  // All intervals for histogram generation (when enabled)
  std::map<MeshEvent, std::vector<std::chrono::milliseconds>> all_intervals_;
};

/**
 * @brief Helper to track event patterns
 */
class EventPatternTracker {
public:
  /**
   * @brief Pattern entry
   */
  struct Pattern {
    std::vector<MeshEvent> sequence;
    size_t count = 0;
    std::chrono::milliseconds total_duration{0};
    std::chrono::milliseconds min_duration{std::chrono::milliseconds::max()};
    std::chrono::milliseconds max_duration{0};
  };

  /**
   * @brief Track an event sequence
   */
  void track_sequence(const std::vector<MeshEvent>& sequence,
                      std::chrono::milliseconds duration);

  /**
   * @brief Get all tracked patterns
   */
  const std::map<std::vector<MeshEvent>, Pattern>& patterns() const {
    return patterns_;
  }

  /**
   * @brief Get most common patterns
   */
  std::vector<Pattern> most_common_patterns(size_t n = 10) const;

  /**
   * @brief Generate pattern report
   */
  std::string pattern_report() const;

  /**
   * @brief Reset all patterns
   */
  void reset() { patterns_.clear(); }

private:
  std::map<std::vector<MeshEvent>, Pattern> patterns_;
};

} // namespace svmp

#endif // SVMP_EVENT_COUNTER_OBSERVER_H