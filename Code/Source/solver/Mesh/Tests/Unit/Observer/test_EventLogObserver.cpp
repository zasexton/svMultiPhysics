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
#include "../../../Observer/EventLogObserver.h"
#include <chrono>
#include <fstream>
#include <memory>
#include <sstream>
#include <thread>

using namespace svmp;

// ====================
// EventLogObserver Tests
// ====================

class EventLogObserverTest : public ::testing::Test {
protected:
  std::shared_ptr<std::stringstream> output_stream;
  std::unique_ptr<EventLogObserver> logger;

  void SetUp() override {
    output_stream = std::make_shared<std::stringstream>();
  }

  void CreateLoggerWithStringOutput(const EventLogObserver::Config& config = {}) {
    logger = std::make_unique<EventLogObserver>(config);
    logger->set_string_output(output_stream);
  }

  std::string GetOutput() {
    return output_stream->str();
  }

  void ClearOutput() {
    output_stream->str("");
    output_stream->clear();
  }
};

TEST_F(EventLogObserverTest, BasicLogging) {
  CreateLoggerWithStringOutput();

  logger->on_mesh_event(MeshEvent::TopologyChanged);

  std::string output = GetOutput();
  EXPECT_TRUE(output.find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(output.find("Mesh") != std::string::npos);  // Default prefix
}

TEST_F(EventLogObserverTest, CustomPrefix) {
  EventLogObserver::Config config;
  config.prefix = "TestMesh";
  CreateLoggerWithStringOutput(config);

  logger->on_mesh_event(MeshEvent::GeometryChanged);

  std::string output = GetOutput();
  EXPECT_TRUE(output.find("TestMesh") != std::string::npos);
  EXPECT_TRUE(output.find("GeometryChanged") != std::string::npos);
}

TEST_F(EventLogObserverTest, LogLevelMinimal) {
  EventLogObserver::Config config;
  config.level = EventLogObserver::LogLevel::MINIMAL;
  config.include_timestamp = false;
  CreateLoggerWithStringOutput(config);

  logger->on_mesh_event(MeshEvent::TopologyChanged);

  std::string output = GetOutput();
  EXPECT_TRUE(output.find("TopologyChanged") != std::string::npos);
  EXPECT_FALSE(output.find("#") != std::string::npos);  // No count
}

TEST_F(EventLogObserverTest, LogLevelNormal) {
  EventLogObserver::Config config;
  config.level = EventLogObserver::LogLevel::NORMAL;
  config.include_timestamp = false;
  CreateLoggerWithStringOutput(config);

  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::TopologyChanged);

  std::string output = GetOutput();
  EXPECT_TRUE(output.find("#2") != std::string::npos);  // Should show count
}

TEST_F(EventLogObserverTest, LogLevelDetailed) {
  EventLogObserver::Config config;
  config.level = EventLogObserver::LogLevel::DETAILED;
  config.include_timestamp = false;
  CreateLoggerWithStringOutput(config);

  logger->on_mesh_event(MeshEvent::TopologyChanged);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  logger->on_mesh_event(MeshEvent::GeometryChanged);

  std::string output = GetOutput();
  EXPECT_TRUE(output.find("ms]") != std::string::npos);  // Time since last
}

TEST_F(EventLogObserverTest, LogLevelVerbose) {
  EventLogObserver::Config config;
  config.level = EventLogObserver::LogLevel::VERBOSE;
  config.include_timestamp = false;
  CreateLoggerWithStringOutput(config);

  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::GeometryChanged);

  std::string output = GetOutput();
  EXPECT_TRUE(output.find("total:") != std::string::npos);
  EXPECT_TRUE(output.find("#1/1") != std::string::npos ||
              output.find("#1/2") != std::string::npos);  // Count/total
}

TEST_F(EventLogObserverTest, TimestampIncluded) {
  EventLogObserver::Config config;
  config.include_timestamp = true;
  CreateLoggerWithStringOutput(config);

  logger->on_mesh_event(MeshEvent::TopologyChanged);

  std::string output = GetOutput();
  // Check for timestamp format (YYYY-MM-DD HH:MM:SS.mmm)
  EXPECT_TRUE(output.find("-") != std::string::npos);  // Date separator
  EXPECT_TRUE(output.find(":") != std::string::npos);  // Time separator
}

TEST_F(EventLogObserverTest, NoTimestamp) {
  EventLogObserver::Config config;
  config.include_timestamp = false;
  CreateLoggerWithStringOutput(config);

  logger->on_mesh_event(MeshEvent::TopologyChanged);

  std::string output = GetOutput();
  // Should not have date format
  EXPECT_FALSE(output.find("2024-") != std::string::npos &&
               output.find(":") != std::string::npos);
}

TEST_F(EventLogObserverTest, EventFiltering_IncludeMode) {
  EventLogObserver::Config config;
  config.include_timestamp = false;
  config.filter_events = {MeshEvent::TopologyChanged, MeshEvent::GeometryChanged};
  config.filter_mode_include = true;
  CreateLoggerWithStringOutput(config);

  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::GeometryChanged);
  logger->on_mesh_event(MeshEvent::LabelsChanged);
  logger->on_mesh_event(MeshEvent::FieldsChanged);

  std::string output = GetOutput();
  EXPECT_TRUE(output.find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(output.find("GeometryChanged") != std::string::npos);
  EXPECT_FALSE(output.find("LabelsChanged") != std::string::npos);
  EXPECT_FALSE(output.find("FieldsChanged") != std::string::npos);
}

TEST_F(EventLogObserverTest, EventFiltering_ExcludeMode) {
  EventLogObserver::Config config;
  config.include_timestamp = false;
  config.filter_events = {MeshEvent::TopologyChanged, MeshEvent::GeometryChanged};
  config.filter_mode_include = false;  // Exclude mode
  CreateLoggerWithStringOutput(config);

  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::GeometryChanged);
  logger->on_mesh_event(MeshEvent::LabelsChanged);
  logger->on_mesh_event(MeshEvent::FieldsChanged);

  std::string output = GetOutput();
  EXPECT_FALSE(output.find("TopologyChanged") != std::string::npos);
  EXPECT_FALSE(output.find("GeometryChanged") != std::string::npos);
  EXPECT_TRUE(output.find("LabelsChanged") != std::string::npos);
  EXPECT_TRUE(output.find("FieldsChanged") != std::string::npos);
}

TEST_F(EventLogObserverTest, Throttling) {
  EventLogObserver::Config config;
  config.include_timestamp = false;
  config.throttle_enabled = true;
  config.throttle_interval = std::chrono::milliseconds(50);
  CreateLoggerWithStringOutput(config);

  // Fire events rapidly
  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::TopologyChanged);  // Should be throttled
  logger->on_mesh_event(MeshEvent::TopologyChanged);  // Should be throttled

  std::string output = GetOutput();
  // Should only have one TopologyChanged logged
  size_t count = 0;
  size_t pos = 0;
  while ((pos = output.find("TopologyChanged", pos)) != std::string::npos) {
    count++;
    pos += strlen("TopologyChanged");
  }
  EXPECT_EQ(count, 1);
  EXPECT_EQ(logger->throttled_events(), 2);

  // Wait for throttle interval to pass
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  ClearOutput();

  logger->on_mesh_event(MeshEvent::TopologyChanged);  // Should not be throttled

  output = GetOutput();
  EXPECT_TRUE(output.find("TopologyChanged") != std::string::npos);
}

TEST_F(EventLogObserverTest, ThrottlingPerEvent) {
  EventLogObserver::Config config;
  config.include_timestamp = false;
  config.throttle_enabled = true;
  config.throttle_interval = std::chrono::milliseconds(50);
  CreateLoggerWithStringOutput(config);

  // Different events should have independent throttling
  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::TopologyChanged);  // Throttled
  logger->on_mesh_event(MeshEvent::GeometryChanged);  // Not throttled (different event)
  logger->on_mesh_event(MeshEvent::GeometryChanged);  // Throttled

  std::string output = GetOutput();
  size_t topology_count = 0;
  size_t geometry_count = 0;
  size_t pos = 0;

  while ((pos = output.find("TopologyChanged", pos)) != std::string::npos) {
    topology_count++;
    pos += strlen("TopologyChanged");
  }

  pos = 0;
  while ((pos = output.find("GeometryChanged", pos)) != std::string::npos) {
    geometry_count++;
    pos += strlen("GeometryChanged");
  }

  EXPECT_EQ(topology_count, 1);
  EXPECT_EQ(geometry_count, 1);
  EXPECT_EQ(logger->throttled_events(), 2);
}

TEST_F(EventLogObserverTest, SetFilter) {
  EventLogObserver::Config config;
  config.include_timestamp = false;
  CreateLoggerWithStringOutput(config);

  // Initially no filter
  logger->on_mesh_event(MeshEvent::TopologyChanged);
  EXPECT_TRUE(GetOutput().find("TopologyChanged") != std::string::npos);

  ClearOutput();

  // Set filter
  logger->set_filter({MeshEvent::GeometryChanged}, true);
  logger->on_mesh_event(MeshEvent::TopologyChanged);  // Filtered out
  logger->on_mesh_event(MeshEvent::GeometryChanged);  // Allowed

  std::string output = GetOutput();
  EXPECT_FALSE(output.find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(output.find("GeometryChanged") != std::string::npos);
}

TEST_F(EventLogObserverTest, ClearFilter) {
  EventLogObserver::Config config;
  config.include_timestamp = false;
  config.filter_events = {MeshEvent::TopologyChanged};
  config.filter_mode_include = true;
  CreateLoggerWithStringOutput(config);

  logger->on_mesh_event(MeshEvent::GeometryChanged);  // Filtered out
  EXPECT_TRUE(GetOutput().empty());

  logger->clear_filter();
  logger->on_mesh_event(MeshEvent::GeometryChanged);  // Now allowed

  EXPECT_TRUE(GetOutput().find("GeometryChanged") != std::string::npos);
}

TEST_F(EventLogObserverTest, EventCounting) {
  CreateLoggerWithStringOutput();

  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::GeometryChanged);

  EXPECT_EQ(logger->total_events(), 3);
  EXPECT_EQ(logger->event_count(MeshEvent::TopologyChanged), 2);
  EXPECT_EQ(logger->event_count(MeshEvent::GeometryChanged), 1);
  EXPECT_EQ(logger->event_count(MeshEvent::LabelsChanged), 0);
}

TEST_F(EventLogObserverTest, ResetCounters) {
  CreateLoggerWithStringOutput();

  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::GeometryChanged);
  EXPECT_EQ(logger->total_events(), 2);

  logger->reset_counters();
  EXPECT_EQ(logger->total_events(), 0);
  EXPECT_EQ(logger->event_count(MeshEvent::TopologyChanged), 0);
  EXPECT_EQ(logger->throttled_events(), 0);
}

TEST_F(EventLogObserverTest, SummaryReport) {
  CreateLoggerWithStringOutput();

  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::GeometryChanged);
  logger->on_mesh_event(MeshEvent::LabelsChanged);

  std::string report = logger->summary_report();

  EXPECT_TRUE(report.find("EventLogObserver Summary") != std::string::npos);
  EXPECT_TRUE(report.find("Total events: 4") != std::string::npos);
  EXPECT_TRUE(report.find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(report.find("50.0%") != std::string::npos);  // 2/4 = 50%
  EXPECT_TRUE(report.find("events/second") != std::string::npos);
}

TEST_F(EventLogObserverTest, FileOutput) {
  // Create a temporary file
  std::string temp_file = "test_event_log.txt";

  {
    EventLogObserver::Config config;
    config.include_timestamp = false;
    EventLogObserver file_logger(config);
    file_logger.set_file_output(temp_file);

    file_logger.on_mesh_event(MeshEvent::TopologyChanged);
    file_logger.on_mesh_event(MeshEvent::GeometryChanged);
  }

  // Read the file
  std::ifstream file(temp_file);
  ASSERT_TRUE(file.is_open());

  std::string line;
  bool found_topology = false;
  bool found_geometry = false;

  while (std::getline(file, line)) {
    if (line.find("TopologyChanged") != std::string::npos) {
      found_topology = true;
    }
    if (line.find("GeometryChanged") != std::string::npos) {
      found_geometry = true;
    }
  }

  EXPECT_TRUE(found_topology);
  EXPECT_TRUE(found_geometry);

  file.close();
  std::remove(temp_file.c_str());
}

TEST_F(EventLogObserverTest, CustomSink) {
  std::vector<std::string> messages;

  EventLogObserver::Config config;
  config.include_timestamp = false;
  EventLogObserver::LogSink custom_sink = [&messages](const std::string& msg) {
    messages.push_back(msg);
  };

  EventLogObserver custom_logger(config, custom_sink);

  custom_logger.on_mesh_event(MeshEvent::TopologyChanged);
  custom_logger.on_mesh_event(MeshEvent::GeometryChanged);

  ASSERT_EQ(messages.size(), 2);
  EXPECT_TRUE(messages[0].find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(messages[1].find("GeometryChanged") != std::string::npos);
}

TEST_F(EventLogObserverTest, SetStdoutOutput) {
  CreateLoggerWithStringOutput();

  // Initially using string output
  logger->on_mesh_event(MeshEvent::TopologyChanged);
  EXPECT_FALSE(GetOutput().empty());

  // Switch to stdout
  logger->set_stdout_output();
  ClearOutput();

  logger->on_mesh_event(MeshEvent::GeometryChanged);
  // String output should be empty now
  EXPECT_TRUE(GetOutput().empty());
}

TEST_F(EventLogObserverTest, ObserverName) {
  CreateLoggerWithStringOutput();
  EXPECT_STREQ(logger->observer_name(), "EventLogObserver");
}

// ====================
// EventLogBuilder Tests
// ====================

TEST(EventLogBuilderTest, BasicBuilder) {
  auto logger = EventLogBuilder()
    .with_prefix("TestMesh")
    .with_level(EventLogObserver::LogLevel::DETAILED)
    .with_timestamps(false)
    .to_stdout()
    .build();

  ASSERT_NE(logger, nullptr);

  // Test that settings were applied
  logger->on_mesh_event(MeshEvent::TopologyChanged);
  EXPECT_EQ(logger->total_events(), 1);
}

TEST(EventLogBuilderTest, BuilderWithThrottling) {
  auto logger = EventLogBuilder()
    .with_prefix("ThrottledMesh")
    .with_throttling(std::chrono::milliseconds(100))
    .build();

  ASSERT_NE(logger, nullptr);

  // Test throttling
  logger->on_mesh_event(MeshEvent::TopologyChanged);
  logger->on_mesh_event(MeshEvent::TopologyChanged);  // Should be throttled

  EXPECT_EQ(logger->total_events(), 2);
  EXPECT_EQ(logger->throttled_events(), 1);
}

TEST(EventLogBuilderTest, BuilderWithFilter) {
  std::set<MeshEvent> filter = {MeshEvent::TopologyChanged, MeshEvent::GeometryChanged};

  auto logger = EventLogBuilder()
    .with_filter(filter, true)
    .build();

  ASSERT_NE(logger, nullptr);

  // Set string output for testing
  auto stream = std::make_shared<std::stringstream>();
  logger->set_string_output(stream);

  logger->on_mesh_event(MeshEvent::TopologyChanged);  // Allowed
  logger->on_mesh_event(MeshEvent::LabelsChanged);    // Filtered

  std::string output = stream->str();
  EXPECT_TRUE(output.find("TopologyChanged") != std::string::npos);
  EXPECT_FALSE(output.find("LabelsChanged") != std::string::npos);
}

TEST(EventLogBuilderTest, BuilderToFile) {
  std::string temp_file = "test_builder_log.txt";

  auto logger = EventLogBuilder()
    .with_prefix("FileMesh")
    .to_file(temp_file)
    .build();

  ASSERT_NE(logger, nullptr);

  logger->on_mesh_event(MeshEvent::TopologyChanged);

  // Verify file was created and written
  std::ifstream file(temp_file);
  ASSERT_TRUE(file.is_open());

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  EXPECT_TRUE(content.find("FileMesh") != std::string::npos);
  EXPECT_TRUE(content.find("TopologyChanged") != std::string::npos);

  file.close();
  std::remove(temp_file.c_str());
}

// ====================
// Integration Tests
// ====================

TEST(EventLogObserverIntegrationTest, CompleteWorkflow) {
  // Create logger with all features enabled
  EventLogObserver::Config config;
  config.prefix = "IntegrationTest";
  config.level = EventLogObserver::LogLevel::VERBOSE;
  config.include_timestamp = true;
  config.throttle_enabled = true;
  config.throttle_interval = std::chrono::milliseconds(20);
  config.filter_events = {MeshEvent::PartitionChanged, MeshEvent::AdaptivityApplied};
  config.filter_mode_include = false;  // Exclude these

  auto output = std::make_shared<std::stringstream>();
  EventLogObserver logger(config);
  logger.set_string_output(output);

  // Generate events
  logger.on_mesh_event(MeshEvent::TopologyChanged);
  logger.on_mesh_event(MeshEvent::TopologyChanged);  // Throttled
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  logger.on_mesh_event(MeshEvent::TopologyChanged);  // Not throttled

  logger.on_mesh_event(MeshEvent::GeometryChanged);
  logger.on_mesh_event(MeshEvent::LabelsChanged);
  logger.on_mesh_event(MeshEvent::FieldsChanged);

  logger.on_mesh_event(MeshEvent::PartitionChanged);   // Filtered
  logger.on_mesh_event(MeshEvent::AdaptivityApplied);  // Filtered

  // Check statistics
  EXPECT_EQ(logger.total_events(), 8);
  EXPECT_EQ(logger.throttled_events(), 1);
  EXPECT_EQ(logger.event_count(MeshEvent::TopologyChanged), 3);
  EXPECT_EQ(logger.event_count(MeshEvent::PartitionChanged), 1);

  // Check output
  std::string log_output = output->str();

  // Should have logged events (not filtered)
  EXPECT_TRUE(log_output.find("TopologyChanged") != std::string::npos);
  EXPECT_TRUE(log_output.find("GeometryChanged") != std::string::npos);
  EXPECT_TRUE(log_output.find("LabelsChanged") != std::string::npos);
  EXPECT_TRUE(log_output.find("FieldsChanged") != std::string::npos);

  // Should not have logged filtered events
  EXPECT_FALSE(log_output.find("PartitionChanged") != std::string::npos);
  EXPECT_FALSE(log_output.find("AdaptivityApplied") != std::string::npos);

  // Generate summary
  std::string summary = logger.summary_report();
  EXPECT_TRUE(summary.find("Total events: 8") != std::string::npos);
}