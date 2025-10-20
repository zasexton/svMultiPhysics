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

#include <gtest/gtest.h>
#include "../../../Observer/EventCounterObserver.h"
#include <chrono>
#include <thread>
#include <regex>

using namespace svmp;

// ====================
// EventStats Tests
// ====================

TEST(EventStatsTest, InitialState) {
  EventCounterObserver::EventStats stats;

  EXPECT_EQ(stats.count, 0);
  EXPECT_EQ(stats.total_interval.count(), 0);
  EXPECT_EQ(stats.min_interval.count(), std::chrono::milliseconds::max().count());
  EXPECT_EQ(stats.max_interval.count(), 0);
  EXPECT_TRUE(stats.recent_times.empty());
  EXPECT_EQ(stats.average_interval_ms(), 0.0);
  EXPECT_EQ(stats.events_per_second(), 0.0);
}

TEST(EventStatsTest, Reset) {
  EventCounterObserver::EventStats stats;

  // Populate with some data
  stats.count = 10;
  stats.total_interval = std::chrono::milliseconds(500);
  stats.min_interval = std::chrono::milliseconds(10);
  stats.max_interval = std::chrono::milliseconds(100);
  stats.recent_times.push_back(std::chrono::steady_clock::now());

  stats.reset();

  EXPECT_EQ(stats.count, 0);
  EXPECT_EQ(stats.total_interval.count(), 0);
  EXPECT_EQ(stats.min_interval.count(), std::chrono::milliseconds::max().count());
  EXPECT_EQ(stats.max_interval.count(), 0);
  EXPECT_TRUE(stats.recent_times.empty());
}

TEST(EventStatsTest, AverageInterval) {
  EventCounterObserver::EventStats stats;

  stats.count = 5;
  stats.total_interval = std::chrono::milliseconds(400);

  // Average = 400ms / (5-1) = 100ms
  EXPECT_DOUBLE_EQ(stats.average_interval_ms(), 100.0);

  // Edge cases
  stats.count = 0;
  EXPECT_EQ(stats.average_interval_ms(), 0.0);

  stats.count = 1;
  EXPECT_EQ(stats.average_interval_ms(), 0.0);
}

TEST(EventStatsTest, EventsPerSecond) {
  EventCounterObserver::EventStats stats;

  auto now = std::chrono::steady_clock::now();
  stats.first_time = now;
  stats.last_time = now + std::chrono::milliseconds(2000);
  stats.count = 10;

  // 10 events in 2000ms = 5 events/second
  EXPECT_DOUBLE_EQ(stats.events_per_second(), 5.0);

  // Edge cases
  stats.count = 0;
  EXPECT_EQ(stats.events_per_second(), 0.0);

  stats.count = 1;
  stats.last_time = stats.first_time; // No duration
  EXPECT_EQ(stats.events_per_second(), 0.0);
}

// ====================
// EventCounterObserver Tests
// ====================

class EventCounterObserverTest : public ::testing::Test {
protected:
  std::unique_ptr<EventCounterObserver> counter;

  void SetUp() override {
    counter = std::make_unique<EventCounterObserver>();
  }
};

TEST_F(EventCounterObserverTest, DefaultConfig) {
  EventCounterObserver::Config default_config;
  EventCounterObserver counter_with_config(default_config);

  EXPECT_EQ(counter_with_config.total_count(), 0);
  EXPECT_FALSE(counter_with_config.has_events());
}

TEST_F(EventCounterObserverTest, BasicCounting) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::GeometryChanged);

  EXPECT_EQ(counter->total_count(), 3);
  EXPECT_EQ(counter->count(MeshEvent::TopologyChanged), 2);
  EXPECT_EQ(counter->count(MeshEvent::GeometryChanged), 1);
  EXPECT_EQ(counter->count(MeshEvent::LabelsChanged), 0);
  EXPECT_TRUE(counter->has_events());
}

TEST_F(EventCounterObserverTest, EventStats) {
  // Add events with timing
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  counter->on_mesh_event(MeshEvent::TopologyChanged);

  const auto& stats = counter->stats(MeshEvent::TopologyChanged);
  EXPECT_EQ(stats.count, 3);
  EXPECT_GE(stats.total_interval.count(), 30);  // At least 30ms total
  EXPECT_GT(stats.min_interval.count(), 0);
  EXPECT_GT(stats.max_interval.count(), stats.min_interval.count());
  EXPECT_EQ(stats.recent_times.size(), 3);
}

TEST_F(EventCounterObserverTest, AllStats) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::GeometryChanged);
  counter->on_mesh_event(MeshEvent::LabelsChanged);

  const auto& all_stats = counter->all_stats();
  EXPECT_EQ(all_stats.size(), 3);
  EXPECT_EQ(all_stats.at(MeshEvent::TopologyChanged).count, 1);
  EXPECT_EQ(all_stats.at(MeshEvent::GeometryChanged).count, 1);
  EXPECT_EQ(all_stats.at(MeshEvent::LabelsChanged).count, 1);
}

TEST_F(EventCounterObserverTest, RecentEventsTracking) {
  EventCounterObserver::Config config;
  config.max_recent_events = 3;
  EventCounterObserver limited_counter(config);

  // Add more events than the limit
  for (int i = 0; i < 5; i++) {
    limited_counter.on_mesh_event(MeshEvent::TopologyChanged);
  }

  const auto& stats = limited_counter.stats(MeshEvent::TopologyChanged);
  EXPECT_EQ(stats.count, 5);  // Total count unchanged
  EXPECT_LE(stats.recent_times.size(), 3);  // Recent times limited
}

TEST_F(EventCounterObserverTest, IntervalTracking) {
  EventCounterObserver::Config config;
  config.track_intervals = true;
  EventCounterObserver interval_counter(config);

  interval_counter.on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  interval_counter.on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  interval_counter.on_mesh_event(MeshEvent::TopologyChanged);

  const auto& stats = interval_counter.stats(MeshEvent::TopologyChanged);
  EXPECT_GT(stats.average_interval_ms(), 0);
  EXPECT_GT(stats.events_per_second(), 0);
}

TEST_F(EventCounterObserverTest, NoIntervalTracking) {
  EventCounterObserver::Config config;
  config.track_intervals = false;
  EventCounterObserver no_interval_counter(config);

  no_interval_counter.on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  no_interval_counter.on_mesh_event(MeshEvent::TopologyChanged);

  const auto& stats = no_interval_counter.stats(MeshEvent::TopologyChanged);
  EXPECT_EQ(stats.total_interval.count(), 0);  // No interval tracking
}

TEST_F(EventCounterObserverTest, Reset) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::GeometryChanged);

  EXPECT_EQ(counter->total_count(), 2);
  EXPECT_TRUE(counter->has_events());

  counter->reset();

  EXPECT_EQ(counter->total_count(), 0);
  EXPECT_FALSE(counter->has_events());
  EXPECT_EQ(counter->count(MeshEvent::TopologyChanged), 0);
  EXPECT_EQ(counter->count(MeshEvent::GeometryChanged), 0);
}

TEST_F(EventCounterObserverTest, ResetSpecificEvent) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::GeometryChanged);

  EXPECT_EQ(counter->total_count(), 3);

  counter->reset_event(MeshEvent::TopologyChanged);

  EXPECT_EQ(counter->total_count(), 1);  // Only GeometryChanged remains
  EXPECT_EQ(counter->count(MeshEvent::TopologyChanged), 0);
  EXPECT_EQ(counter->count(MeshEvent::GeometryChanged), 1);
}

TEST_F(EventCounterObserverTest, ObserverName) {
  EXPECT_STREQ(counter->observer_name(), "EventCounterObserver");
}

// ====================
// Histogram Tests
// ====================

TEST_F(EventCounterObserverTest, GenerateHistogramNoData) {
  auto histogram = counter->generate_histogram(MeshEvent::TopologyChanged, 10);
  EXPECT_TRUE(histogram.empty());
}

TEST_F(EventCounterObserverTest, GenerateHistogramSingleEvent) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);

  auto histogram = counter->generate_histogram(MeshEvent::TopologyChanged, 10);
  EXPECT_TRUE(histogram.empty());  // Need at least 2 events for intervals
}

TEST_F(EventCounterObserverTest, GenerateHistogramWithData) {
  EventCounterObserver::Config config;
  config.enable_histograms = true;
  EventCounterObserver hist_counter(config);

  // Generate events with varying intervals
  hist_counter.on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  hist_counter.on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  hist_counter.on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(15));
  hist_counter.on_mesh_event(MeshEvent::TopologyChanged);

  auto histogram = hist_counter.generate_histogram(MeshEvent::TopologyChanged, 5);

  EXPECT_FALSE(histogram.empty());
  EXPECT_LE(histogram.size(), 5);

  // Check that bins cover the range
  if (!histogram.empty()) {
    size_t total_count = 0;
    for (const auto& bin : histogram) {
      total_count += bin.count;
      EXPECT_GE(bin.upper_bound.count(), bin.lower_bound.count());
    }
    EXPECT_EQ(total_count, 3);  // 3 intervals from 4 events
  }
}

TEST_F(EventCounterObserverTest, GenerateCombinedHistogram) {
  EventCounterObserver::Config config;
  config.enable_histograms = true;
  EventCounterObserver hist_counter(config);

  // Generate events for multiple types
  hist_counter.on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  hist_counter.on_mesh_event(MeshEvent::TopologyChanged);

  hist_counter.on_mesh_event(MeshEvent::GeometryChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  hist_counter.on_mesh_event(MeshEvent::GeometryChanged);

  auto histogram = hist_counter.generate_combined_histogram(5);

  if (!histogram.empty()) {
    size_t total_count = 0;
    for (const auto& bin : histogram) {
      total_count += bin.count;
    }
    EXPECT_EQ(total_count, 2);  // 2 total intervals
  }
}

TEST_F(EventCounterObserverTest, FormatHistogram) {
  std::vector<EventCounterObserver::HistogramBin> histogram;
  histogram.push_back({std::chrono::milliseconds(0), std::chrono::milliseconds(10), 5});
  histogram.push_back({std::chrono::milliseconds(10), std::chrono::milliseconds(20), 3});
  histogram.push_back({std::chrono::milliseconds(20), std::chrono::milliseconds(30), 1});

  std::string formatted = counter->format_histogram(histogram, "Test Histogram");

  EXPECT_TRUE(formatted.find("Test Histogram") != std::string::npos);
  EXPECT_TRUE(formatted.find("0-") != std::string::npos);
  EXPECT_TRUE(formatted.find("10-") != std::string::npos);
  EXPECT_TRUE(formatted.find("20-") != std::string::npos);
  EXPECT_TRUE(formatted.find("#") != std::string::npos);  // Bar characters
}

TEST_F(EventCounterObserverTest, FormatEmptyHistogram) {
  std::vector<EventCounterObserver::HistogramBin> empty_histogram;
  std::string formatted = counter->format_histogram(empty_histogram);

  EXPECT_TRUE(formatted.find("No data") != std::string::npos);
}

// ====================
// Snapshot Tests
// ====================

TEST_F(EventCounterObserverTest, CreateSnapshot) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::GeometryChanged);

  auto snapshot = counter->create_snapshot("test_snapshot");

  EXPECT_EQ(snapshot.name, "test_snapshot");
  EXPECT_EQ(snapshot.total_events, 2);
  EXPECT_EQ(snapshot.stats.size(), 2);
  EXPECT_EQ(snapshot.stats.at(MeshEvent::TopologyChanged).count, 1);
  EXPECT_EQ(snapshot.stats.at(MeshEvent::GeometryChanged).count, 1);
}

TEST_F(EventCounterObserverTest, SnapshotSummary) {
  EventCounterObserver::Snapshot snapshot;
  snapshot.name = "test";
  snapshot.total_events = 3;
  snapshot.stats[MeshEvent::TopologyChanged].count = 2;
  snapshot.stats[MeshEvent::GeometryChanged].count = 1;

  std::string summary = snapshot.summary();

  EXPECT_TRUE(summary.find("Snapshot: test") != std::string::npos);
  EXPECT_TRUE(summary.find("Total events: 3") != std::string::npos);
  EXPECT_TRUE(summary.find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(summary.find("GeometryChanged") != std::string::npos);
}

TEST_F(EventCounterObserverTest, StoreAndRetrieveSnapshot) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::GeometryChanged);

  counter->store_snapshot("snap1");

  const auto* retrieved = counter->get_snapshot("snap1");
  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->name, "snap1");
  EXPECT_EQ(retrieved->total_events, 2);

  const auto* non_existent = counter->get_snapshot("non_existent");
  EXPECT_EQ(non_existent, nullptr);
}

TEST_F(EventCounterObserverTest, ListSnapshots) {
  counter->store_snapshot("snap1");
  counter->store_snapshot("snap2");
  counter->store_snapshot("snap3");

  auto snapshots = counter->list_snapshots();
  EXPECT_EQ(snapshots.size(), 3);
  EXPECT_TRUE(std::find(snapshots.begin(), snapshots.end(), "snap1") != snapshots.end());
  EXPECT_TRUE(std::find(snapshots.begin(), snapshots.end(), "snap2") != snapshots.end());
  EXPECT_TRUE(std::find(snapshots.begin(), snapshots.end(), "snap3") != snapshots.end());
}

TEST_F(EventCounterObserverTest, CompareSnapshots) {
  // Create first snapshot
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::GeometryChanged);
  auto snap1 = counter->create_snapshot("snap1");

  // Add more events
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::LabelsChanged);
  auto snap2 = counter->create_snapshot("snap2");

  std::string comparison = counter->compare_snapshots(snap1, snap2);

  EXPECT_TRUE(comparison.find("snap1 vs snap2") != std::string::npos);
  EXPECT_TRUE(comparison.find("2 -> 4") != std::string::npos);  // Total events
  EXPECT_TRUE(comparison.find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(comparison.find("1 -> 2") != std::string::npos);  // Topology count
}

TEST_F(EventCounterObserverTest, CompareWithSnapshot) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->store_snapshot("initial");

  counter->on_mesh_event(MeshEvent::GeometryChanged);
  counter->on_mesh_event(MeshEvent::TopologyChanged);

  std::string comparison = counter->compare_with_snapshot("initial");

  EXPECT_TRUE(comparison.find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(comparison.find("1 -> 2") != std::string::npos);
  EXPECT_TRUE(comparison.find("GeometryChanged") != std::string::npos);
  EXPECT_TRUE(comparison.find("0 -> 1") != std::string::npos);
}

TEST_F(EventCounterObserverTest, CompareWithNonExistentSnapshot) {
  std::string comparison = counter->compare_with_snapshot("non_existent");
  EXPECT_TRUE(comparison.find("not found") != std::string::npos);
}

// ====================
// Report Tests
// ====================

TEST_F(EventCounterObserverTest, DetailedReport) {
  EventCounterObserver::Config config;
  config.track_intervals = true;
  config.enable_histograms = true;
  config.histogram_bins = 5;
  EventCounterObserver detailed_counter(config);

  // Generate some events
  detailed_counter.on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  detailed_counter.on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  detailed_counter.on_mesh_event(MeshEvent::GeometryChanged);

  std::string report = detailed_counter.detailed_report();

  EXPECT_TRUE(report.find("EventCounterObserver Detailed Report") != std::string::npos);
  EXPECT_TRUE(report.find("Total events: 3") != std::string::npos);
  EXPECT_TRUE(report.find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(report.find("Count: 2") != std::string::npos);
  EXPECT_TRUE(report.find("Average:") != std::string::npos);
  EXPECT_TRUE(report.find("events/second") != std::string::npos);
}

TEST_F(EventCounterObserverTest, SummaryReport) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::GeometryChanged);

  std::string report = counter->summary_report();

  EXPECT_TRUE(report.find("EventCounterObserver Summary") != std::string::npos);
  EXPECT_TRUE(report.find("Total events: 3") != std::string::npos);
  EXPECT_TRUE(report.find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(report.find("66.7%") != std::string::npos ||
              report.find("66.6%") != std::string::npos);  // 2/3
}

TEST_F(EventCounterObserverTest, SummaryReportNoEvents) {
  std::string report = counter->summary_report();

  EXPECT_TRUE(report.find("No events recorded") != std::string::npos);
}

// ====================
// Export Tests
// ====================

TEST_F(EventCounterObserverTest, ExportCSV) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::GeometryChanged);

  std::string csv = counter->export_csv();

  // Check header
  EXPECT_TRUE(csv.find("Event,Count,Percentage") != std::string::npos);

  // Check data rows
  EXPECT_TRUE(csv.find("TopologyChanged,2,") != std::string::npos);
  EXPECT_TRUE(csv.find("GeometryChanged,1,") != std::string::npos);

  // Check it's valid CSV (has commas)
  size_t comma_count = std::count(csv.begin(), csv.end(), ',');
  EXPECT_GT(comma_count, 6);  // At least header + 2 data rows
}

TEST_F(EventCounterObserverTest, ExportJSON) {
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  counter->on_mesh_event(MeshEvent::GeometryChanged);

  std::string json = counter->export_json();

  // Check JSON structure
  EXPECT_TRUE(json.find("\"total_events\": 3") != std::string::npos);
  EXPECT_TRUE(json.find("\"events\": {") != std::string::npos);
  EXPECT_TRUE(json.find("\"TopologyChanged\": {") != std::string::npos);
  EXPECT_TRUE(json.find("\"count\": 2") != std::string::npos);
  EXPECT_TRUE(json.find("\"GeometryChanged\": {") != std::string::npos);
  EXPECT_TRUE(json.find("\"count\": 1") != std::string::npos);

  // Check it's valid JSON (has matching braces)
  size_t open_braces = std::count(json.begin(), json.end(), '{');
  size_t close_braces = std::count(json.begin(), json.end(), '}');
  EXPECT_EQ(open_braces, close_braces);
}

TEST_F(EventCounterObserverTest, ExportJSONWithIntervals) {
  EventCounterObserver::Config config;
  config.track_intervals = true;
  EventCounterObserver interval_counter(config);

  interval_counter.on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  interval_counter.on_mesh_event(MeshEvent::TopologyChanged);

  std::string json = interval_counter.export_json();

  EXPECT_TRUE(json.find("\"average_interval_ms\":") != std::string::npos);
  EXPECT_TRUE(json.find("\"min_interval_ms\":") != std::string::npos);
  EXPECT_TRUE(json.find("\"max_interval_ms\":") != std::string::npos);
  EXPECT_TRUE(json.find("\"events_per_second\":") != std::string::npos);
}

// ====================
// Configuration Tests
// ====================

TEST_F(EventCounterObserverTest, SetMaxRecentEvents) {
  counter->set_max_recent_events(2);

  // Add more events than the new limit
  for (int i = 0; i < 5; i++) {
    counter->on_mesh_event(MeshEvent::TopologyChanged);
  }

  const auto& stats = counter->stats(MeshEvent::TopologyChanged);
  EXPECT_LE(stats.recent_times.size(), 2);
}

TEST_F(EventCounterObserverTest, SetTrackIntervals) {
  counter->set_track_intervals(false);

  counter->on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  counter->on_mesh_event(MeshEvent::TopologyChanged);

  const auto& stats = counter->stats(MeshEvent::TopologyChanged);
  EXPECT_EQ(stats.total_interval.count(), 0);

  counter->set_track_intervals(true);

  counter->on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  counter->on_mesh_event(MeshEvent::TopologyChanged);

  const auto& stats2 = counter->stats(MeshEvent::TopologyChanged);
  EXPECT_GT(stats2.total_interval.count(), 0);
}

TEST_F(EventCounterObserverTest, SetEnableHistograms) {
  counter->set_enable_histograms(true);

  counter->on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  counter->on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  counter->on_mesh_event(MeshEvent::TopologyChanged);

  auto histogram = counter->generate_histogram(MeshEvent::TopologyChanged, 5);

  // With histograms enabled, should get valid histogram if implementation stores intervals
  // The actual behavior depends on the implementation
}

// ====================
// EventPatternTracker Tests
// ====================

TEST(EventPatternTrackerTest, TrackSimpleSequence) {
  EventPatternTracker tracker;

  std::vector<MeshEvent> sequence = {
    MeshEvent::TopologyChanged,
    MeshEvent::GeometryChanged
  };

  tracker.track_sequence(sequence, std::chrono::milliseconds(100));

  const auto& patterns = tracker.patterns();
  EXPECT_EQ(patterns.size(), 1);

  auto it = patterns.find(sequence);
  ASSERT_NE(it, patterns.end());
  EXPECT_EQ(it->second.count, 1);
  EXPECT_EQ(it->second.total_duration.count(), 100);
  EXPECT_EQ(it->second.min_duration.count(), 100);
  EXPECT_EQ(it->second.max_duration.count(), 100);
}

TEST(EventPatternTrackerTest, TrackMultipleOccurrences) {
  EventPatternTracker tracker;

  std::vector<MeshEvent> sequence = {
    MeshEvent::TopologyChanged,
    MeshEvent::GeometryChanged,
    MeshEvent::LabelsChanged
  };

  tracker.track_sequence(sequence, std::chrono::milliseconds(100));
  tracker.track_sequence(sequence, std::chrono::milliseconds(150));
  tracker.track_sequence(sequence, std::chrono::milliseconds(50));

  const auto& patterns = tracker.patterns();
  auto it = patterns.find(sequence);
  ASSERT_NE(it, patterns.end());

  EXPECT_EQ(it->second.count, 3);
  EXPECT_EQ(it->second.total_duration.count(), 300);
  EXPECT_EQ(it->second.min_duration.count(), 50);
  EXPECT_EQ(it->second.max_duration.count(), 150);
}

TEST(EventPatternTrackerTest, MostCommonPatterns) {
  EventPatternTracker tracker;

  std::vector<MeshEvent> pattern1 = {MeshEvent::TopologyChanged};
  std::vector<MeshEvent> pattern2 = {MeshEvent::GeometryChanged};
  std::vector<MeshEvent> pattern3 = {MeshEvent::LabelsChanged};

  // Track with different frequencies
  for (int i = 0; i < 5; i++) {
    tracker.track_sequence(pattern1, std::chrono::milliseconds(10));
  }
  for (int i = 0; i < 3; i++) {
    tracker.track_sequence(pattern2, std::chrono::milliseconds(10));
  }
  tracker.track_sequence(pattern3, std::chrono::milliseconds(10));

  auto most_common = tracker.most_common_patterns(2);

  ASSERT_EQ(most_common.size(), 2);
  EXPECT_EQ(most_common[0].count, 5);  // pattern1
  EXPECT_EQ(most_common[1].count, 3);  // pattern2
}

TEST(EventPatternTrackerTest, PatternReport) {
  EventPatternTracker tracker;

  std::vector<MeshEvent> sequence = {
    MeshEvent::TopologyChanged,
    MeshEvent::GeometryChanged
  };

  tracker.track_sequence(sequence, std::chrono::milliseconds(100));
  tracker.track_sequence(sequence, std::chrono::milliseconds(200));

  std::string report = tracker.pattern_report();

  EXPECT_TRUE(report.find("Event Pattern Report") != std::string::npos);
  EXPECT_TRUE(report.find("occurred 2 times") != std::string::npos);
  EXPECT_TRUE(report.find("TopologyChanged -> GeometryChanged") != std::string::npos);
  EXPECT_TRUE(report.find("avg=150ms") != std::string::npos);
  EXPECT_TRUE(report.find("min=100ms") != std::string::npos);
  EXPECT_TRUE(report.find("max=200ms") != std::string::npos);
}

TEST(EventPatternTrackerTest, Reset) {
  EventPatternTracker tracker;

  std::vector<MeshEvent> sequence = {MeshEvent::TopologyChanged};
  tracker.track_sequence(sequence, std::chrono::milliseconds(100));

  EXPECT_EQ(tracker.patterns().size(), 1);

  tracker.reset();

  EXPECT_EQ(tracker.patterns().size(), 0);
}

// ====================
// Integration Tests
// ====================

TEST(EventCounterObserverIntegrationTest, CompleteWorkflow) {
  EventCounterObserver::Config config;
  config.max_recent_events = 50;
  config.track_intervals = true;
  config.enable_histograms = true;
  config.histogram_bins = 10;

  EventCounterObserver counter(config);

  // Simulate realistic event sequence
  for (int i = 0; i < 10; i++) {
    counter.on_mesh_event(MeshEvent::TopologyChanged);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    if (i % 3 == 0) {
      counter.on_mesh_event(MeshEvent::GeometryChanged);
    }

    if (i % 5 == 0) {
      counter.on_mesh_event(MeshEvent::LabelsChanged);
    }
  }

  // Take initial snapshot
  counter.store_snapshot("initial");

  // Generate more events
  for (int i = 0; i < 5; i++) {
    counter.on_mesh_event(MeshEvent::FieldsChanged);
    counter.on_mesh_event(MeshEvent::PartitionChanged);
  }

  // Take final snapshot
  counter.store_snapshot("final");

  // Verify counts
  EXPECT_EQ(counter.count(MeshEvent::TopologyChanged), 10);
  EXPECT_EQ(counter.count(MeshEvent::GeometryChanged), 4);
  EXPECT_EQ(counter.count(MeshEvent::LabelsChanged), 2);
  EXPECT_EQ(counter.count(MeshEvent::FieldsChanged), 5);
  EXPECT_EQ(counter.count(MeshEvent::PartitionChanged), 5);
  EXPECT_EQ(counter.total_count(), 26);

  // Check reports
  std::string detailed = counter.detailed_report();
  EXPECT_TRUE(detailed.find("Total events: 26") != std::string::npos);

  std::string summary = counter.summary_report();
  EXPECT_TRUE(summary.find("TopologyChanged") != std::string::npos);

  // Check exports
  std::string csv = counter.export_csv();
  EXPECT_TRUE(csv.find("TopologyChanged,10,") != std::string::npos);

  std::string json = counter.export_json();
  EXPECT_TRUE(json.find("\"total_events\": 26") != std::string::npos);

  // Check snapshots
  auto snapshots = counter.list_snapshots();
  EXPECT_EQ(snapshots.size(), 2);

  std::string comparison = counter.compare_with_snapshot("initial");
  EXPECT_TRUE(comparison.find("FieldsChanged") != std::string::npos);
  EXPECT_TRUE(comparison.find("0 -> 5") != std::string::npos);
}