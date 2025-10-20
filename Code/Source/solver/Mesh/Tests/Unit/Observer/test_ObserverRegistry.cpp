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
#include "../../../Observer/ObserverRegistry.h"
#include <memory>
#include <sstream>

using namespace svmp;

// Test cache class with invalidation tracking
class TestCache {
public:
  bool valid = true;
  size_t invalidation_count = 0;

  void invalidate() {
    valid = false;
    invalidation_count++;
  }

  bool is_valid() const { return valid; }

  void reset() {
    valid = true;
    invalidation_count = 0;
  }
};

// Test search accelerator
class TestSearchAccel {
public:
  bool valid = true;
  size_t invalidation_count = 0;

  void invalidate() {
    valid = false;
    invalidation_count++;
  }

  bool is_valid() const { return valid; }

  void reset() {
    valid = true;
    invalidation_count = 0;
  }
};

// ====================
// ObserverRegistry Tests
// ====================

class ObserverRegistryTest : public ::testing::Test {
protected:
  MeshEventBus bus;
  TestCache cache;
  TestSearchAccel search_accel;

  void SetUp() override {
    // Clear registry before each test
    ObserverRegistry::instance().cleanup_expired();
  }

  void TearDown() override {
    // Clean up after test
    bus.clear();
    ObserverRegistry::instance().cleanup_expired();
  }
};

TEST_F(ObserverRegistryTest, AttachSearchInvalidator) {
  auto observer = ObserverRegistry::attach_search_invalidator(bus, search_accel);
  ASSERT_NE(observer, nullptr);
  EXPECT_EQ(bus.num_observers(), 1);

  // Search invalidator should respond to topology, geometry, and partition changes
  EXPECT_TRUE(search_accel.is_valid());

  bus.notify(MeshEvent::TopologyChanged);
  EXPECT_FALSE(search_accel.is_valid());
  EXPECT_EQ(search_accel.invalidation_count, 1);

  search_accel.reset();
  bus.notify(MeshEvent::GeometryChanged);
  EXPECT_FALSE(search_accel.is_valid());
  EXPECT_EQ(search_accel.invalidation_count, 1);

  search_accel.reset();
  bus.notify(MeshEvent::PartitionChanged);
  EXPECT_FALSE(search_accel.is_valid());
  EXPECT_EQ(search_accel.invalidation_count, 1);

  // Should not respond to other events
  search_accel.reset();
  bus.notify(MeshEvent::LabelsChanged);
  EXPECT_TRUE(search_accel.is_valid());
  EXPECT_EQ(search_accel.invalidation_count, 0);
}

TEST_F(ObserverRegistryTest, AttachGeometryCacheInvalidator) {
  auto observer = ObserverRegistry::attach_geometry_cache_invalidator(bus, cache);
  ASSERT_NE(observer, nullptr);
  EXPECT_EQ(bus.num_observers(), 1);

  // Geometry cache invalidator responds to topology and geometry changes
  EXPECT_TRUE(cache.is_valid());

  bus.notify(MeshEvent::GeometryChanged);
  EXPECT_FALSE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 1);

  cache.reset();
  bus.notify(MeshEvent::TopologyChanged);
  EXPECT_FALSE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 1);

  // Should not respond to other events
  cache.reset();
  bus.notify(MeshEvent::LabelsChanged);
  EXPECT_TRUE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 0);
}

TEST_F(ObserverRegistryTest, AttachLabelCacheInvalidator) {
  auto observer = ObserverRegistry::attach_label_cache_invalidator(bus, cache);
  ASSERT_NE(observer, nullptr);
  EXPECT_EQ(bus.num_observers(), 1);

  // Label cache invalidator only responds to label changes
  EXPECT_TRUE(cache.is_valid());

  bus.notify(MeshEvent::LabelsChanged);
  EXPECT_FALSE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 1);

  // Should not respond to other events
  cache.reset();
  bus.notify(MeshEvent::TopologyChanged);
  EXPECT_TRUE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 0);

  bus.notify(MeshEvent::GeometryChanged);
  EXPECT_TRUE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 0);
}

TEST_F(ObserverRegistryTest, AttachFieldCacheInvalidator) {
  auto observer = ObserverRegistry::attach_field_cache_invalidator(bus, cache);
  ASSERT_NE(observer, nullptr);
  EXPECT_EQ(bus.num_observers(), 1);

  // Field cache invalidator only responds to field changes
  EXPECT_TRUE(cache.is_valid());

  bus.notify(MeshEvent::FieldsChanged);
  EXPECT_FALSE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 1);

  // Should not respond to other events
  cache.reset();
  bus.notify(MeshEvent::TopologyChanged);
  EXPECT_TRUE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 0);
}

TEST_F(ObserverRegistryTest, AttachEventLogger) {
  // Capture output
  std::stringstream buffer;
  std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

  auto logger = ObserverRegistry::attach_event_logger(bus, "TestLogger", true);
  ASSERT_NE(logger, nullptr);
  EXPECT_EQ(bus.num_observers(), 1);

  bus.notify(MeshEvent::TopologyChanged);

  std::string output = buffer.str();
  EXPECT_TRUE(output.find("TestLogger") != std::string::npos);
  EXPECT_TRUE(output.find("TopologyChanged") != std::string::npos);

  // Test disabled logger
  buffer.str("");
  auto disabled_logger = ObserverRegistry::attach_event_logger(bus, "DisabledLogger", false);
  bus.notify(MeshEvent::GeometryChanged);

  output = buffer.str();
  EXPECT_TRUE(output.find("TestLogger") != std::string::npos);  // First logger still logs
  EXPECT_FALSE(output.find("DisabledLogger") != std::string::npos);  // Second logger doesn't

  // Restore cout
  std::cout.rdbuf(old);
}

TEST_F(ObserverRegistryTest, AttachEventCounter) {
  auto counter = ObserverRegistry::attach_event_counter(bus);
  ASSERT_NE(counter, nullptr);
  EXPECT_EQ(bus.num_observers(), 1);

  bus.notify(MeshEvent::TopologyChanged);
  bus.notify(MeshEvent::TopologyChanged);
  bus.notify(MeshEvent::GeometryChanged);

  EXPECT_EQ(counter->count(MeshEvent::TopologyChanged), 2);
  EXPECT_EQ(counter->count(MeshEvent::GeometryChanged), 1);
  EXPECT_EQ(counter->count(MeshEvent::LabelsChanged), 0);
}

TEST_F(ObserverRegistryTest, CreateMultiEventInvalidator) {
  std::vector<MeshEvent> events = {
    MeshEvent::LabelsChanged,
    MeshEvent::FieldsChanged,
    MeshEvent::AdaptivityApplied
  };

  auto observer = ObserverRegistry::create_multi_event_invalidator(cache, events);
  ASSERT_NE(observer, nullptr);

  bus.subscribe(observer);

  // Should invalidate on specified events
  bus.notify(MeshEvent::LabelsChanged);
  EXPECT_FALSE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 1);

  cache.reset();
  bus.notify(MeshEvent::FieldsChanged);
  EXPECT_FALSE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 1);

  cache.reset();
  bus.notify(MeshEvent::AdaptivityApplied);
  EXPECT_FALSE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 1);

  // Should not invalidate on other events
  cache.reset();
  bus.notify(MeshEvent::TopologyChanged);
  EXPECT_TRUE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 0);
}

TEST_F(ObserverRegistryTest, GlobalRegistryTracking) {
  auto& registry = ObserverRegistry::instance();

  // Register some observers
  auto obs1 = std::make_shared<EventCounter>();
  registry.register_observer("mesh1", "Counter1", "Diagnostics", obs1);

  auto obs2 = std::make_shared<EventCounter>();
  registry.register_observer("mesh1", "Counter2", "Diagnostics", obs2);

  auto obs3 = std::make_shared<EventCounter>();
  registry.register_observer("mesh2", "Counter3", "Diagnostics", obs3);

  // Get observers for mesh1
  auto mesh1_observers = registry.get_observers("mesh1");
  EXPECT_EQ(mesh1_observers.size(), 2);

  // Get observers for mesh2
  auto mesh2_observers = registry.get_observers("mesh2");
  EXPECT_EQ(mesh2_observers.size(), 1);

  // Get all observers
  auto all_observers = registry.get_all_observers();
  EXPECT_GE(all_observers.size(), 3);  // May have more from other tests
}

TEST_F(ObserverRegistryTest, CleanupExpired) {
  auto& registry = ObserverRegistry::instance();

  // Register observer that will expire
  {
    auto temp_obs = std::make_shared<EventCounter>();
    registry.register_observer("temp_mesh", "TempCounter", "Diagnostics",
                              std::weak_ptr<MeshObserver>(temp_obs));

    auto observers = registry.get_observers("temp_mesh");
    EXPECT_GE(observers.size(), 1);
  }
  // temp_obs is now destroyed, weak_ptr is expired

  registry.cleanup_expired();

  auto observers = registry.get_observers("temp_mesh");
  EXPECT_EQ(observers.size(), 0);
}

TEST_F(ObserverRegistryTest, DiagnosticReport) {
  auto& registry = ObserverRegistry::instance();

  // Clean registry first
  registry.cleanup_expired();

  // Register some observers
  auto obs1 = std::make_shared<EventCounter>();
  registry.register_observer("test_mesh", "TestCounter", "Diagnostics", obs1);

  auto report = registry.diagnostic_report();

  EXPECT_TRUE(report.find("Observer Registry Diagnostic Report") != std::string::npos);
  EXPECT_TRUE(report.find("test_mesh") != std::string::npos);
  EXPECT_TRUE(report.find("TestCounter") != std::string::npos);
  EXPECT_TRUE(report.find("ACTIVE") != std::string::npos);
}

// ====================
// MultiEventInvalidator Tests
// ====================

TEST(MultiEventInvalidatorTest, SingleEventTrigger) {
  TestCache cache;
  std::vector<MeshEvent> events = { MeshEvent::TopologyChanged };

  MultiEventInvalidator<TestCache> invalidator(cache, events);

  invalidator.on_mesh_event(MeshEvent::TopologyChanged);
  EXPECT_FALSE(cache.is_valid());

  cache.reset();
  invalidator.on_mesh_event(MeshEvent::GeometryChanged);
  EXPECT_TRUE(cache.is_valid());
}

TEST(MultiEventInvalidatorTest, MultipleEventTriggers) {
  TestCache cache;
  std::vector<MeshEvent> events = {
    MeshEvent::TopologyChanged,
    MeshEvent::GeometryChanged,
    MeshEvent::PartitionChanged
  };

  MultiEventInvalidator<TestCache> invalidator(cache, events);

  // Test each trigger event
  invalidator.on_mesh_event(MeshEvent::TopologyChanged);
  EXPECT_FALSE(cache.is_valid());

  cache.reset();
  invalidator.on_mesh_event(MeshEvent::GeometryChanged);
  EXPECT_FALSE(cache.is_valid());

  cache.reset();
  invalidator.on_mesh_event(MeshEvent::PartitionChanged);
  EXPECT_FALSE(cache.is_valid());

  // Test non-trigger event
  cache.reset();
  invalidator.on_mesh_event(MeshEvent::LabelsChanged);
  EXPECT_TRUE(cache.is_valid());
}

TEST(MultiEventInvalidatorTest, EmptyEventList) {
  TestCache cache;
  std::vector<MeshEvent> events;  // Empty

  MultiEventInvalidator<TestCache> invalidator(cache, events);

  // Should not invalidate on any event
  invalidator.on_mesh_event(MeshEvent::TopologyChanged);
  EXPECT_TRUE(cache.is_valid());

  invalidator.on_mesh_event(MeshEvent::GeometryChanged);
  EXPECT_TRUE(cache.is_valid());
}

// ====================
// SearchInvalidator Tests
// ====================

TEST(SearchInvalidatorTest, InvalidatesOnCorrectEvents) {
  TestSearchAccel accel;
  SearchInvalidator<TestSearchAccel> invalidator(accel);

  // Should invalidate on topology change
  invalidator.on_mesh_event(MeshEvent::TopologyChanged);
  EXPECT_FALSE(accel.is_valid());

  // Should invalidate on geometry change
  accel.reset();
  invalidator.on_mesh_event(MeshEvent::GeometryChanged);
  EXPECT_FALSE(accel.is_valid());

  // Should invalidate on partition change
  accel.reset();
  invalidator.on_mesh_event(MeshEvent::PartitionChanged);
  EXPECT_FALSE(accel.is_valid());

  // Should not invalidate on other events
  accel.reset();
  invalidator.on_mesh_event(MeshEvent::LabelsChanged);
  EXPECT_TRUE(accel.is_valid());

  invalidator.on_mesh_event(MeshEvent::FieldsChanged);
  EXPECT_TRUE(accel.is_valid());

  invalidator.on_mesh_event(MeshEvent::AdaptivityApplied);
  EXPECT_TRUE(accel.is_valid());
}

TEST(SearchInvalidatorTest, ObserverName) {
  TestSearchAccel accel;
  SearchInvalidator<TestSearchAccel> invalidator(accel);
  EXPECT_STREQ(invalidator.observer_name(), "SearchInvalidator");
}

// ====================
// Integration Tests
// ====================

TEST(ObserverRegistryIntegrationTest, MultipleCachesAndObservers) {
  MeshEventBus bus;
  TestCache geometry_cache;
  TestCache label_cache;
  TestSearchAccel search_accel;

  // Attach various invalidators
  auto geom_obs = ObserverRegistry::attach_geometry_cache_invalidator(bus, geometry_cache);
  auto label_obs = ObserverRegistry::attach_label_cache_invalidator(bus, label_cache);
  auto search_obs = ObserverRegistry::attach_search_invalidator(bus, search_accel);
  auto counter = ObserverRegistry::attach_event_counter(bus);

  EXPECT_EQ(bus.num_observers(), 4);

  // Test topology change - should invalidate geometry and search
  bus.notify(MeshEvent::TopologyChanged);
  EXPECT_FALSE(geometry_cache.is_valid());
  EXPECT_TRUE(label_cache.is_valid());
  EXPECT_FALSE(search_accel.is_valid());
  EXPECT_EQ(counter->count(MeshEvent::TopologyChanged), 1);

  // Reset and test label change
  geometry_cache.reset();
  search_accel.reset();

  bus.notify(MeshEvent::LabelsChanged);
  EXPECT_TRUE(geometry_cache.is_valid());
  EXPECT_FALSE(label_cache.is_valid());
  EXPECT_TRUE(search_accel.is_valid());
  EXPECT_EQ(counter->count(MeshEvent::LabelsChanged), 1);
}

TEST(ObserverRegistryIntegrationTest, ComplexEventFlow) {
  MeshEventBus bus;
  auto& registry = ObserverRegistry::instance();

  // Create multiple caches
  TestCache cache1, cache2, cache3;

  // Create custom multi-event invalidators
  auto inv1 = ObserverRegistry::create_multi_event_invalidator(cache1,
    {MeshEvent::TopologyChanged, MeshEvent::GeometryChanged});

  auto inv2 = ObserverRegistry::create_multi_event_invalidator(cache2,
    {MeshEvent::LabelsChanged, MeshEvent::FieldsChanged});

  auto inv3 = ObserverRegistry::create_multi_event_invalidator(cache3,
    {MeshEvent::PartitionChanged, MeshEvent::AdaptivityApplied});

  bus.subscribe(inv1);
  bus.subscribe(inv2);
  bus.subscribe(inv3);

  // Add counter for tracking
  auto counter = ObserverRegistry::attach_event_counter(bus);

  // Simulate a sequence of events
  std::vector<MeshEvent> event_sequence = {
    MeshEvent::TopologyChanged,   // Invalidates cache1
    MeshEvent::LabelsChanged,      // Invalidates cache2
    MeshEvent::PartitionChanged,   // Invalidates cache3
    MeshEvent::GeometryChanged,    // Invalidates cache1 again
    MeshEvent::FieldsChanged,      // Invalidates cache2 again
  };

  for (auto event : event_sequence) {
    bus.notify(event);
  }

  // Verify final state
  EXPECT_FALSE(cache1.is_valid());
  EXPECT_EQ(cache1.invalidation_count, 2);

  EXPECT_FALSE(cache2.is_valid());
  EXPECT_EQ(cache2.invalidation_count, 2);

  EXPECT_FALSE(cache3.is_valid());
  EXPECT_EQ(cache3.invalidation_count, 1);

  // Verify counts
  EXPECT_EQ(counter->count(MeshEvent::TopologyChanged), 1);
  EXPECT_EQ(counter->count(MeshEvent::LabelsChanged), 1);
  EXPECT_EQ(counter->count(MeshEvent::PartitionChanged), 1);
  EXPECT_EQ(counter->count(MeshEvent::GeometryChanged), 1);
  EXPECT_EQ(counter->count(MeshEvent::FieldsChanged), 1);
}