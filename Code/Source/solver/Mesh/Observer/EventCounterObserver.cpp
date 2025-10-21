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

#include "EventCounterObserver.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>

namespace svmp {

// ====================
// EventStats Implementation
// ====================

void EventCounterObserver::EventStats::reset() {
  count = 0;
  first_time = {};
  last_time = {};
  total_interval = std::chrono::milliseconds{0};
  min_interval = std::chrono::milliseconds::max();
  max_interval = std::chrono::milliseconds{0};
  recent_times.clear();
}

double EventCounterObserver::EventStats::average_interval_ms() const {
  if (count <= 1) return 0.0;
  return static_cast<double>(total_interval.count()) / (count - 1);
}

double EventCounterObserver::EventStats::events_per_second() const {
  if (count == 0) return 0.0;
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      last_time - first_time);
  if (duration.count() == 0) return 0.0;
  return (static_cast<double>(count) * 1000.0) / duration.count();
}

// ====================
// Snapshot Implementation
// ====================

std::string EventCounterObserver::Snapshot::summary() const {
  std::stringstream ss;
  ss << "Snapshot: " << (name.empty() ? "<unnamed>" : name) << std::endl;
  ss << "Total events: " << total_events << std::endl;
  ss << "Event breakdown:" << std::endl;

  for (const auto& [event, stat] : stats) {
    ss << "  " << std::left << std::setw(20) << event_name(event)
       << ": " << std::right << std::setw(8) << stat.count;
    if (stat.count > 1) {
      ss << " (avg interval: " << std::fixed << std::setprecision(2)
         << stat.average_interval_ms() << "ms, "
         << std::fixed << std::setprecision(2)
         << stat.events_per_second() << " events/s)";
    }
    ss << std::endl;
  }

  return ss.str();
}

// ====================
// EventCounterObserver Implementation
// ====================

EventCounterObserver::EventCounterObserver()
    : EventCounterObserver(Config{}) {
}

EventCounterObserver::EventCounterObserver(const Config& config)
    : config_(config),
      start_time_(std::chrono::steady_clock::now()),
      last_event_time_(start_time_) {
}

void EventCounterObserver::on_mesh_event(MeshEvent event) {
  update_stats(event);
  total_events_++;
  last_event_time_ = std::chrono::steady_clock::now();
}

size_t EventCounterObserver::count(MeshEvent event) const {
  auto it = event_stats_.find(event);
  return (it != event_stats_.end()) ? it->second.count : 0;
}

const EventCounterObserver::EventStats& EventCounterObserver::stats(MeshEvent event) const {
  static EventStats empty_stats;
  auto it = event_stats_.find(event);
  return (it != event_stats_.end()) ? it->second : empty_stats;
}

void EventCounterObserver::update_stats(MeshEvent event) {
  auto now = std::chrono::steady_clock::now();
  auto& stats = event_stats_[event];

  if (stats.count == 0) {
    // First occurrence of this event
    stats.first_time = now;
  } else if (config_.track_intervals) {
    // Calculate interval from last occurrence
    auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - stats.last_time);
    stats.total_interval += interval;
    stats.min_interval = std::min(stats.min_interval, interval);
    stats.max_interval = std::max(stats.max_interval, interval);

    // Store interval for histogram if enabled
    if (config_.enable_histograms) {
      all_intervals_[event].push_back(interval);
    }
  }

  // Update recent times
  stats.recent_times.push_back(now);
  if (stats.recent_times.size() > config_.max_recent_events) {
    stats.recent_times.pop_front();
  }

  stats.last_time = now;
  stats.count++;
}

void EventCounterObserver::trim_recent_events() {
  for (auto& [event, stats] : event_stats_) {
    while (stats.recent_times.size() > config_.max_recent_events) {
      stats.recent_times.pop_front();
    }
  }
}

std::vector<EventCounterObserver::HistogramBin>
EventCounterObserver::generate_histogram(MeshEvent event, size_t num_bins) const {
  std::vector<HistogramBin> histogram;

  auto it = event_stats_.find(event);
  if (it == event_stats_.end() || it->second.count <= 1) {
    return histogram;  // No data or not enough for intervals
  }

  const auto& stats = it->second;
  auto min_ms = stats.min_interval.count();
  auto max_ms = stats.max_interval.count();

  if (min_ms == max_ms) {
    // All intervals are the same
    histogram.push_back({stats.min_interval, stats.max_interval, stats.count - 1});
    return histogram;
  }

  // Create bins
  auto bin_width = (max_ms - min_ms) / static_cast<double>(num_bins);
  for (size_t i = 0; i < num_bins; ++i) {
    HistogramBin bin;
    bin.lower_bound = std::chrono::milliseconds(
        static_cast<int64_t>(min_ms + i * bin_width));
    bin.upper_bound = std::chrono::milliseconds(
        static_cast<int64_t>(min_ms + (i + 1) * bin_width));
    bin.count = 0;
    histogram.push_back(bin);
  }

  // Fill bins
  if (config_.enable_histograms) {
    auto intervals_it = all_intervals_.find(event);
    if (intervals_it != all_intervals_.end()) {
      for (const auto& interval : intervals_it->second) {
        auto ms = interval.count();
        size_t bin_idx = static_cast<size_t>((ms - min_ms) / bin_width);
        if (bin_idx >= num_bins) bin_idx = num_bins - 1;
        histogram[bin_idx].count++;
      }
    }
  }

  return histogram;
}

std::vector<EventCounterObserver::HistogramBin>
EventCounterObserver::generate_combined_histogram(size_t num_bins) const {
  std::vector<HistogramBin> histogram;

  // Collect all intervals
  std::vector<std::chrono::milliseconds> all_combined;
  for (const auto& [event, intervals] : all_intervals_) {
    all_combined.insert(all_combined.end(), intervals.begin(), intervals.end());
  }

  if (all_combined.empty()) {
    return histogram;
  }

  // Find min and max
  auto [min_it, max_it] = std::minmax_element(all_combined.begin(), all_combined.end());
  auto min_ms = min_it->count();
  auto max_ms = max_it->count();

  if (min_ms == max_ms) {
    histogram.push_back({*min_it, *max_it, all_combined.size()});
    return histogram;
  }

  // Create and fill bins
  auto bin_width = (max_ms - min_ms) / static_cast<double>(num_bins);
  for (size_t i = 0; i < num_bins; ++i) {
    HistogramBin bin;
    bin.lower_bound = std::chrono::milliseconds(
        static_cast<int64_t>(min_ms + i * bin_width));
    bin.upper_bound = std::chrono::milliseconds(
        static_cast<int64_t>(min_ms + (i + 1) * bin_width));
    bin.count = 0;
    histogram.push_back(bin);
  }

  for (const auto& interval : all_combined) {
    auto ms = interval.count();
    size_t bin_idx = static_cast<size_t>((ms - min_ms) / bin_width);
    if (bin_idx >= num_bins) bin_idx = num_bins - 1;
    histogram[bin_idx].count++;
  }

  return histogram;
}

std::string EventCounterObserver::format_histogram(
    const std::vector<HistogramBin>& histogram,
    const std::string& title) const {
  if (histogram.empty()) {
    return title + ": No data\n";
  }

  std::stringstream ss;
  ss << title << std::endl;

  // Find max count for scaling
  size_t max_count = 0;
  for (const auto& bin : histogram) {
    max_count = std::max(max_count, bin.count);
  }

  const size_t bar_width = 40;

  for (const auto& bin : histogram) {
    ss << std::setw(6) << bin.lower_bound.count() << "-"
       << std::setw(6) << bin.upper_bound.count() << "ms | ";

    // Draw bar
    size_t bar_len = (max_count > 0) ? (bin.count * bar_width / max_count) : 0;
    ss << std::string(bar_len, '#')
       << std::string(bar_width - bar_len, ' ')
       << " | " << bin.count << std::endl;
  }

  return ss.str();
}

EventCounterObserver::Snapshot EventCounterObserver::create_snapshot(
    const std::string& name) const {
  Snapshot snap;
  snap.name = name;
  snap.timestamp = std::chrono::steady_clock::now();
  snap.stats = event_stats_;
  snap.total_events = total_events_;
  return snap;
}

void EventCounterObserver::store_snapshot(const std::string& name) {
  snapshots_[name] = create_snapshot(name);
}

const EventCounterObserver::Snapshot* EventCounterObserver::get_snapshot(
    const std::string& name) const {
  auto it = snapshots_.find(name);
  return (it != snapshots_.end()) ? &it->second : nullptr;
}

std::vector<std::string> EventCounterObserver::list_snapshots() const {
  std::vector<std::string> names;
  names.reserve(snapshots_.size());
  for (const auto& [name, snap] : snapshots_) {
    names.push_back(name);
  }
  return names;
}

std::string EventCounterObserver::compare_snapshots(
    const Snapshot& s1, const Snapshot& s2) const {
  std::stringstream ss;
  ss << "Snapshot Comparison: " << s1.name << " vs " << s2.name << std::endl;
  ss << "Total events: " << s1.total_events << " -> " << s2.total_events
     << " (" << std::showpos << static_cast<int64_t>(s2.total_events - s1.total_events)
     << std::noshowpos << ")" << std::endl;

  // Collect all event types
  std::set<MeshEvent> all_events;
  for (const auto& [evt, _] : s1.stats) all_events.insert(evt);
  for (const auto& [evt, _] : s2.stats) all_events.insert(evt);

  ss << "Event changes:" << std::endl;
  for (const auto& event : all_events) {
    auto it1 = s1.stats.find(event);
    auto it2 = s2.stats.find(event);

    size_t count1 = (it1 != s1.stats.end()) ? it1->second.count : 0;
    size_t count2 = (it2 != s2.stats.end()) ? it2->second.count : 0;

    ss << "  " << event_name(event)
       << ": " << count1
       << " -> " << count2
       << " (" << std::showpos << static_cast<int64_t>(count2 - count1)
       << std::noshowpos << ")" << std::endl;
  }

  return ss.str();
}

std::string EventCounterObserver::compare_with_snapshot(
    const std::string& snapshot_name) const {
  auto snap = get_snapshot(snapshot_name);
  if (!snap) {
    return "Snapshot '" + snapshot_name + "' not found";
  }

  auto current = create_snapshot("current");
  return compare_snapshots(*snap, current);
}

void EventCounterObserver::reset() {
  event_stats_.clear();
  total_events_ = 0;
  start_time_ = std::chrono::steady_clock::now();
  last_event_time_ = start_time_;
  all_intervals_.clear();
}

void EventCounterObserver::reset_event(MeshEvent event) {
  event_stats_.erase(event);
  all_intervals_.erase(event);

  // Recalculate total
  total_events_ = 0;
  for (const auto& [evt, stats] : event_stats_) {
    total_events_ += stats.count;
  }
}

std::string EventCounterObserver::detailed_report() const {
  std::stringstream report;
  auto now = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(
      now - start_time_);

  report << "=== EventCounterObserver Detailed Report ===" << std::endl;
  report << "Duration: " << duration.count() << " seconds" << std::endl;
  report << "Total events: " << total_events_ << std::endl;

  if (duration.count() > 0) {
    report << "Overall rate: " << std::fixed << std::setprecision(2)
           << (static_cast<double>(total_events_) / duration.count())
           << " events/second" << std::endl;
  }

  report << std::endl << "Event Statistics:" << std::endl;

  for (const auto& [event, stats] : event_stats_) {
    report << std::endl << event_name(event) << ":" << std::endl;
    report << "  Count: " << stats.count << std::endl;

    if (stats.count > 1 && config_.track_intervals) {
      report << "  Intervals:" << std::endl;
      report << "    Average: " << std::fixed << std::setprecision(2)
             << stats.average_interval_ms() << "ms" << std::endl;
      report << "    Min: " << stats.min_interval.count() << "ms" << std::endl;
      report << "    Max: " << stats.max_interval.count() << "ms" << std::endl;
      report << "  Rate: " << std::fixed << std::setprecision(2)
             << stats.events_per_second() << " events/second" << std::endl;
    }

    if (!stats.recent_times.empty()) {
      auto recent_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          stats.recent_times.back() - stats.recent_times.front());
      if (recent_duration.count() > 0) {
        report << "  Recent rate (" << stats.recent_times.size() << " events): "
               << std::fixed << std::setprecision(2)
               << (stats.recent_times.size() * 1000.0 / recent_duration.count())
               << " events/second" << std::endl;
      }
    }
  }

  if (config_.enable_histograms && !all_intervals_.empty()) {
    report << std::endl << "Interval Histograms:" << std::endl;
    for (const auto& [event, intervals] : all_intervals_) {
      if (!intervals.empty()) {
        auto histogram = generate_histogram(event, config_.histogram_bins);
        report << std::endl << format_histogram(histogram,
            std::string(event_name(event)) + " Intervals");
      }
    }
  }

  return report.str();
}

std::string EventCounterObserver::summary_report() const {
  std::stringstream report;
  report << "=== EventCounterObserver Summary ===" << std::endl;
  report << "Total events: " << total_events_ << std::endl;

  if (total_events_ == 0) {
    report << "No events recorded" << std::endl;
    return report.str();
  }

  report << "Event counts:" << std::endl;
  for (const auto& [event, stats] : event_stats_) {
    double percentage = (100.0 * stats.count) / total_events_;
    report << "  " << std::left << std::setw(20) << event_name(event)
           << ": " << std::right << std::setw(8) << stats.count
           << " (" << std::fixed << std::setprecision(1) << percentage << "%)"
           << std::endl;
  }

  return report.str();
}

std::string EventCounterObserver::export_csv() const {
  std::stringstream csv;
  csv << "Event,Count,Percentage,AvgInterval_ms,MinInterval_ms,MaxInterval_ms,EventsPerSecond"
      << std::endl;

  for (const auto& [event, stats] : event_stats_) {
    double percentage = (total_events_ > 0) ? (100.0 * stats.count / total_events_) : 0.0;
    csv << event_name(event) << ","
        << stats.count << ","
        << std::fixed << std::setprecision(2) << percentage << ",";

    if (stats.count > 1) {
      csv << stats.average_interval_ms() << ","
          << stats.min_interval.count() << ","
          << stats.max_interval.count() << ","
          << stats.events_per_second();
    } else {
      csv << "N/A,N/A,N/A,N/A";
    }
    csv << std::endl;
  }

  return csv.str();
}

std::string EventCounterObserver::export_json() const {
  std::stringstream json;
  json << "{" << std::endl;
  json << "  \"total_events\": " << total_events_ << "," << std::endl;
  json << "  \"events\": {" << std::endl;

  bool first = true;
  for (const auto& [event, stats] : event_stats_) {
    if (!first) json << "," << std::endl;
    first = false;

    json << "    \"" << event_name(event) << "\": {" << std::endl;
    json << "      \"count\": " << stats.count << "," << std::endl;
    json << "      \"percentage\": " << std::fixed << std::setprecision(2)
         << ((total_events_ > 0) ? (100.0 * stats.count / total_events_) : 0.0) << "," << std::endl;

    if (stats.count > 1) {
      json << "      \"average_interval_ms\": " << stats.average_interval_ms() << "," << std::endl;
      json << "      \"min_interval_ms\": " << stats.min_interval.count() << "," << std::endl;
      json << "      \"max_interval_ms\": " << stats.max_interval.count() << "," << std::endl;
      json << "      \"events_per_second\": " << stats.events_per_second() << std::endl;
    } else {
      json << "      \"average_interval_ms\": null," << std::endl;
      json << "      \"min_interval_ms\": null," << std::endl;
      json << "      \"max_interval_ms\": null," << std::endl;
      json << "      \"events_per_second\": null" << std::endl;
    }
    json << "    }";
  }

  json << std::endl << "  }" << std::endl;
  json << "}" << std::endl;

  return json.str();
}

// ====================
// EventPatternTracker Implementation
// ====================

void EventPatternTracker::track_sequence(const std::vector<MeshEvent>& sequence,
                                        std::chrono::milliseconds duration) {
  auto& pattern = patterns_[sequence];
  pattern.sequence = sequence;
  pattern.count++;
  pattern.total_duration += duration;
  pattern.min_duration = std::min(pattern.min_duration, duration);
  pattern.max_duration = std::max(pattern.max_duration, duration);
}

std::vector<EventPatternTracker::Pattern> EventPatternTracker::most_common_patterns(
    size_t n) const {
  std::vector<Pattern> sorted_patterns;
  sorted_patterns.reserve(patterns_.size());

  for (const auto& [seq, pattern] : patterns_) {
    sorted_patterns.push_back(pattern);
  }

  std::sort(sorted_patterns.begin(), sorted_patterns.end(),
           [](const Pattern& a, const Pattern& b) {
             return a.count > b.count;
           });

  if (sorted_patterns.size() > n) {
    sorted_patterns.resize(n);
  }

  return sorted_patterns;
}

std::string EventPatternTracker::pattern_report() const {
  std::stringstream report;
  report << "=== Event Pattern Report ===" << std::endl;
  report << "Total unique patterns: " << patterns_.size() << std::endl;

  auto common = most_common_patterns(10);
  report << std::endl << "Top " << common.size() << " patterns:" << std::endl;

  for (size_t i = 0; i < common.size(); ++i) {
    const auto& pattern = common[i];
    report << std::endl << i + 1 << ". Pattern (occurred " << pattern.count << " times):" << std::endl;
    report << "   Sequence: ";

    for (size_t j = 0; j < pattern.sequence.size(); ++j) {
      if (j > 0) report << " -> ";
      report << event_name(pattern.sequence[j]);
    }

    auto avg_duration = pattern.total_duration.count() / pattern.count;
    report << std::endl << "   Duration: avg=" << avg_duration << "ms"
           << ", min=" << pattern.min_duration.count() << "ms"
           << ", max=" << pattern.max_duration.count() << "ms" << std::endl;
  }

  return report.str();
}

} // namespace svmp
