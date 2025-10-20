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
#include "../../../Observer/MeshObserver.h"
#include <memory>
#include <vector>

using namespace svmp;

// Test observer that records events
class TestObserver : public MeshObserver {
public:
  std::vector<MeshEvent> events;
  const char* name;

  explicit TestObserver(const char* n = "TestObserver") : name(n) {}

  void on_mesh_event(MeshEvent event) override {
    events.push_back(event);
  }

  const char* observer_name() const override { return name; }

  void clear() { events.clear(); }
};

// Test cache class for CacheInvalidator
class TestCache {
public:
  bool valid = true;
  size_t invalidation_count = 0;

  void invalidate() {
    valid = false;
    invalidation_count++;
  }

  bool is_valid() const { return valid; }
};

// ====================
// MeshEvent Tests
// ====================

TEST(MeshEventTest, EventNameMapping) {
  EXPECT_STREQ(event_name(MeshEvent::TopologyChanged), "TopologyChanged");
  EXPECT_STREQ(event_name(MeshEvent::GeometryChanged), "GeometryChanged");
  EXPECT_STREQ(event_name(MeshEvent::PartitionChanged), "PartitionChanged");
  EXPECT_STREQ(event_name(MeshEvent::LabelsChanged), "LabelsChanged");
  EXPECT_STREQ(event_name(MeshEvent::FieldsChanged), "FieldsChanged");
  EXPECT_STREQ(event_name(MeshEvent::AdaptivityApplied), "AdaptivityApplied");
}

// ====================
// MeshEventBus Tests
// ====================

class MeshEventBusTest : public ::testing::Test {
protected:
  MeshEventBus bus;
  std::unique_ptr<TestObserver> observer1;
  std::unique_ptr<TestObserver> observer2;

  void SetUp() override {
    observer1 = std::make_unique<TestObserver>("Observer1");
    observer2 = std::make_unique<TestObserver>("Observer2");
  }
};

TEST_F(MeshEventBusTest, InitialState) {
  EXPECT_EQ(bus.num_observers(), 0);
  EXPECT_FALSE(bus.has_observers());
}

TEST_F(MeshEventBusTest, SubscribeNonOwning) {
  bus.subscribe(observer1.get());
  EXPECT_EQ(bus.num_observers(), 1);
  EXPECT_TRUE(bus.has_observers());

  bus.subscribe(observer2.get());
  EXPECT_EQ(bus.num_observers(), 2);
}

TEST_F(MeshEventBusTest, SubscribeOwning) {
  auto shared_obs = std::make_shared<TestObserver>("SharedObserver");
  bus.subscribe(shared_obs);
  EXPECT_EQ(bus.num_observers(), 1);
  EXPECT_TRUE(bus.has_observers());
}

TEST_F(MeshEventBusTest, PreventDuplicateSubscription) {
  bus.subscribe(observer1.get());
  bus.subscribe(observer1.get());  // Should not add duplicate
  EXPECT_EQ(bus.num_observers(), 1);
}

TEST_F(MeshEventBusTest, Unsubscribe) {
  bus.subscribe(observer1.get());
  bus.subscribe(observer2.get());
  EXPECT_EQ(bus.num_observers(), 2);

  bus.unsubscribe(observer1.get());
  EXPECT_EQ(bus.num_observers(), 1);
  EXPECT_TRUE(bus.has_observers());

  bus.unsubscribe(observer2.get());
  EXPECT_EQ(bus.num_observers(), 0);
  EXPECT_FALSE(bus.has_observers());
}

TEST_F(MeshEventBusTest, NotifySingleObserver) {
  bus.subscribe(observer1.get());

  bus.notify(MeshEvent::TopologyChanged);
  ASSERT_EQ(observer1->events.size(), 1);
  EXPECT_EQ(observer1->events[0], MeshEvent::TopologyChanged);

  bus.notify(MeshEvent::GeometryChanged);
  ASSERT_EQ(observer1->events.size(), 2);
  EXPECT_EQ(observer1->events[1], MeshEvent::GeometryChanged);
}

TEST_F(MeshEventBusTest, NotifyMultipleObservers) {
  bus.subscribe(observer1.get());
  bus.subscribe(observer2.get());

  bus.notify(MeshEvent::LabelsChanged);

  ASSERT_EQ(observer1->events.size(), 1);
  EXPECT_EQ(observer1->events[0], MeshEvent::LabelsChanged);

  ASSERT_EQ(observer2->events.size(), 1);
  EXPECT_EQ(observer2->events[0], MeshEvent::LabelsChanged);
}

TEST_F(MeshEventBusTest, NotifyWithNoObservers) {
  // Should not crash
  EXPECT_NO_THROW(bus.notify(MeshEvent::FieldsChanged));
}

TEST_F(MeshEventBusTest, Clear) {
  bus.subscribe(observer1.get());
  auto shared_obs = std::make_shared<TestObserver>();
  bus.subscribe(shared_obs);
  EXPECT_EQ(bus.num_observers(), 2);

  bus.clear();
  EXPECT_EQ(bus.num_observers(), 0);
  EXPECT_FALSE(bus.has_observers());
}

TEST_F(MeshEventBusTest, MixedOwnershipSubscriptions) {
  // Add non-owning
  bus.subscribe(observer1.get());

  // Add owning
  auto shared_obs = std::make_shared<TestObserver>("Shared");
  bus.subscribe(shared_obs);

  EXPECT_EQ(bus.num_observers(), 2);

  // Notify and check both receive events
  bus.notify(MeshEvent::PartitionChanged);
  EXPECT_EQ(observer1->events.size(), 1);
  EXPECT_EQ(shared_obs->events.size(), 1);
}

TEST_F(MeshEventBusTest, NullObserverHandling) {
  MeshObserver* null_obs = nullptr;
  bus.subscribe(null_obs);  // Should not add
  EXPECT_EQ(bus.num_observers(), 0);

  std::shared_ptr<MeshObserver> null_shared;
  bus.subscribe(null_shared);  // Should not add
  EXPECT_EQ(bus.num_observers(), 0);
}

// ====================
// CacheInvalidator Tests
// ====================

TEST(CacheInvalidatorTest, InvalidateOnGeometryChange) {
  TestCache cache;
  CacheInvalidator<TestCache> invalidator(cache);

  EXPECT_TRUE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 0);

  invalidator.on_mesh_event(MeshEvent::GeometryChanged);
  EXPECT_FALSE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 1);
}

TEST(CacheInvalidatorTest, InvalidateOnTopologyChange) {
  TestCache cache;
  CacheInvalidator<TestCache> invalidator(cache);

  EXPECT_TRUE(cache.is_valid());

  invalidator.on_mesh_event(MeshEvent::TopologyChanged);
  EXPECT_FALSE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 1);
}

TEST(CacheInvalidatorTest, NoInvalidationOnOtherEvents) {
  TestCache cache;
  CacheInvalidator<TestCache> invalidator(cache);

  invalidator.on_mesh_event(MeshEvent::LabelsChanged);
  EXPECT_TRUE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 0);

  invalidator.on_mesh_event(MeshEvent::FieldsChanged);
  EXPECT_TRUE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 0);

  invalidator.on_mesh_event(MeshEvent::PartitionChanged);
  EXPECT_TRUE(cache.is_valid());
  EXPECT_EQ(cache.invalidation_count, 0);
}

TEST(CacheInvalidatorTest, ObserverName) {
  TestCache cache;
  CacheInvalidator<TestCache> invalidator(cache);
  EXPECT_STREQ(invalidator.observer_name(), "CacheInvalidator");
}

// ====================
// EventLogger Tests
// ====================

TEST(EventLoggerTest, LoggingOutput) {
  // Redirect cout to capture output
  std::stringstream buffer;
  std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

  EventLogger logger("TestMesh");
  logger.on_mesh_event(MeshEvent::TopologyChanged);

  std::string output = buffer.str();
  EXPECT_TRUE(output.find("TestMesh") != std::string::npos);
  EXPECT_TRUE(output.find("TopologyChanged") != std::string::npos);

  // Restore cout
  std::cout.rdbuf(old);
}

TEST(EventLoggerTest, ObserverName) {
  EventLogger logger;
  EXPECT_STREQ(logger.observer_name(), "EventLogger");
}

TEST(EventLoggerTest, CustomPrefix) {
  std::stringstream buffer;
  std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

  EventLogger logger("CustomPrefix");
  logger.on_mesh_event(MeshEvent::GeometryChanged);

  std::string output = buffer.str();
  EXPECT_TRUE(output.find("CustomPrefix") != std::string::npos);
  EXPECT_TRUE(output.find("GeometryChanged") != std::string::npos);

  std::cout.rdbuf(old);
}

// ====================
// EventCounter Tests
// ====================

TEST(EventCounterTest, InitialState) {
  EventCounter counter;

  for (size_t i = 0; i <= static_cast<size_t>(MeshEvent::AdaptivityApplied); ++i) {
    EXPECT_EQ(counter.count(static_cast<MeshEvent>(i)), 0);
  }
}

TEST(EventCounterTest, CountSingleEvent) {
  EventCounter counter;

  counter.on_mesh_event(MeshEvent::TopologyChanged);
  EXPECT_EQ(counter.count(MeshEvent::TopologyChanged), 1);
  EXPECT_EQ(counter.count(MeshEvent::GeometryChanged), 0);
}

TEST(EventCounterTest, CountMultipleEvents) {
  EventCounter counter;

  counter.on_mesh_event(MeshEvent::TopologyChanged);
  counter.on_mesh_event(MeshEvent::TopologyChanged);
  counter.on_mesh_event(MeshEvent::GeometryChanged);
  counter.on_mesh_event(MeshEvent::TopologyChanged);

  EXPECT_EQ(counter.count(MeshEvent::TopologyChanged), 3);
  EXPECT_EQ(counter.count(MeshEvent::GeometryChanged), 1);
  EXPECT_EQ(counter.count(MeshEvent::LabelsChanged), 0);
}

TEST(EventCounterTest, CountAllEventTypes) {
  EventCounter counter;

  // Trigger each event type
  counter.on_mesh_event(MeshEvent::TopologyChanged);
  counter.on_mesh_event(MeshEvent::GeometryChanged);
  counter.on_mesh_event(MeshEvent::PartitionChanged);
  counter.on_mesh_event(MeshEvent::LabelsChanged);
  counter.on_mesh_event(MeshEvent::FieldsChanged);
  counter.on_mesh_event(MeshEvent::AdaptivityApplied);

  EXPECT_EQ(counter.count(MeshEvent::TopologyChanged), 1);
  EXPECT_EQ(counter.count(MeshEvent::GeometryChanged), 1);
  EXPECT_EQ(counter.count(MeshEvent::PartitionChanged), 1);
  EXPECT_EQ(counter.count(MeshEvent::LabelsChanged), 1);
  EXPECT_EQ(counter.count(MeshEvent::FieldsChanged), 1);
  EXPECT_EQ(counter.count(MeshEvent::AdaptivityApplied), 1);
}

TEST(EventCounterTest, Reset) {
  EventCounter counter;

  counter.on_mesh_event(MeshEvent::TopologyChanged);
  counter.on_mesh_event(MeshEvent::GeometryChanged);
  EXPECT_EQ(counter.count(MeshEvent::TopologyChanged), 1);
  EXPECT_EQ(counter.count(MeshEvent::GeometryChanged), 1);

  counter.reset();
  EXPECT_EQ(counter.count(MeshEvent::TopologyChanged), 0);
  EXPECT_EQ(counter.count(MeshEvent::GeometryChanged), 0);
}

TEST(EventCounterTest, ObserverName) {
  EventCounter counter;
  EXPECT_STREQ(counter.observer_name(), "EventCounter");
}

// ====================
// CallbackObserver Tests
// ====================

TEST(CallbackObserverTest, SimpleCallback) {
  std::vector<MeshEvent> received_events;

  CallbackObserver observer([&received_events](MeshEvent evt) {
    received_events.push_back(evt);
  });

  observer.on_mesh_event(MeshEvent::TopologyChanged);
  observer.on_mesh_event(MeshEvent::GeometryChanged);

  ASSERT_EQ(received_events.size(), 2);
  EXPECT_EQ(received_events[0], MeshEvent::TopologyChanged);
  EXPECT_EQ(received_events[1], MeshEvent::GeometryChanged);
}

TEST(CallbackObserverTest, CountingCallback) {
  size_t call_count = 0;

  CallbackObserver observer([&call_count](MeshEvent) {
    call_count++;
  });

  observer.on_mesh_event(MeshEvent::TopologyChanged);
  observer.on_mesh_event(MeshEvent::GeometryChanged);
  observer.on_mesh_event(MeshEvent::LabelsChanged);

  EXPECT_EQ(call_count, 3);
}

TEST(CallbackObserverTest, EmptyCallback) {
  CallbackObserver observer(nullptr);

  // Should not crash
  EXPECT_NO_THROW(observer.on_mesh_event(MeshEvent::TopologyChanged));
}

TEST(CallbackObserverTest, ObserverName) {
  CallbackObserver observer([](MeshEvent) {});
  EXPECT_STREQ(observer.observer_name(), "CallbackObserver");
}

TEST(CallbackObserverTest, ComplexCallback) {
  struct EventStats {
    std::map<MeshEvent, size_t> counts;
    MeshEvent last_event;
  } stats;

  CallbackObserver observer([&stats](MeshEvent evt) {
    stats.counts[evt]++;
    stats.last_event = evt;
  });

  observer.on_mesh_event(MeshEvent::TopologyChanged);
  observer.on_mesh_event(MeshEvent::TopologyChanged);
  observer.on_mesh_event(MeshEvent::GeometryChanged);

  EXPECT_EQ(stats.counts[MeshEvent::TopologyChanged], 2);
  EXPECT_EQ(stats.counts[MeshEvent::GeometryChanged], 1);
  EXPECT_EQ(stats.last_event, MeshEvent::GeometryChanged);
}

// ====================
// Integration Tests
// ====================

TEST(ObserverIntegrationTest, MultipleObserversOnBus) {
  MeshEventBus bus;

  // Create various observers
  TestObserver test_obs("TestObs");

  TestCache cache;
  CacheInvalidator<TestCache> cache_inv(cache);

  size_t callback_count = 0;
  CallbackObserver callback_obs([&callback_count](MeshEvent) {
    callback_count++;
  });

  auto counter = std::make_shared<EventCounter>();

  // Subscribe all observers
  bus.subscribe(&test_obs);
  bus.subscribe(&cache_inv);
  bus.subscribe(&callback_obs);
  bus.subscribe(counter);

  EXPECT_EQ(bus.num_observers(), 4);

  // Send events
  bus.notify(MeshEvent::TopologyChanged);
  bus.notify(MeshEvent::GeometryChanged);
  bus.notify(MeshEvent::LabelsChanged);

  // Verify all observers received events
  EXPECT_EQ(test_obs.events.size(), 3);
  EXPECT_FALSE(cache.is_valid());  // Should be invalidated by topology/geometry changes
  EXPECT_EQ(callback_count, 3);
  EXPECT_EQ(counter->count(MeshEvent::TopologyChanged), 1);
  EXPECT_EQ(counter->count(MeshEvent::GeometryChanged), 1);
  EXPECT_EQ(counter->count(MeshEvent::LabelsChanged), 1);
}

TEST(ObserverIntegrationTest, ObserverLifetime) {
  MeshEventBus bus;
  TestObserver persistent("Persistent");

  bus.subscribe(&persistent);

  {
    auto temporary = std::make_shared<TestObserver>("Temporary");
    bus.subscribe(temporary);
    EXPECT_EQ(bus.num_observers(), 2);

    bus.notify(MeshEvent::TopologyChanged);
    EXPECT_EQ(persistent.events.size(), 1);
    EXPECT_EQ(temporary->events.size(), 1);
  }
  // Temporary observer is destroyed but bus still holds shared_ptr

  EXPECT_EQ(bus.num_observers(), 2);

  bus.notify(MeshEvent::GeometryChanged);
  EXPECT_EQ(persistent.events.size(), 2);
}