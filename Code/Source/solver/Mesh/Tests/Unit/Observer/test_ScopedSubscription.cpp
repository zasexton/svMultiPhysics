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
#include "../../../Observer/ScopedSubscription.h"
#include <memory>
#include <vector>

using namespace svmp;

// Test observer that counts events
class CountingObserver : public MeshObserver {
public:
  size_t event_count = 0;
  std::vector<MeshEvent> events;
  bool* destroyed = nullptr;

  CountingObserver() = default;

  explicit CountingObserver(bool* destroyed_flag) : destroyed(destroyed_flag) {}

  ~CountingObserver() {
    if (destroyed) {
      *destroyed = true;
    }
  }

  void on_mesh_event(MeshEvent event) override {
    event_count++;
    events.push_back(event);
  }

  const char* observer_name() const override {
    return "CountingObserver";
  }

  void reset() {
    event_count = 0;
    events.clear();
  }
};

// ====================
// ScopedSubscription Tests
// ====================

class ScopedSubscriptionTest : public ::testing::Test {
protected:
  MeshEventBus bus;
  std::unique_ptr<CountingObserver> observer;

  void SetUp() override {
    observer = std::make_unique<CountingObserver>();
  }
};

TEST_F(ScopedSubscriptionTest, DefaultConstructor) {
  ScopedSubscription sub;
  EXPECT_FALSE(sub.is_active());
  EXPECT_EQ(sub.observer(), nullptr);
  EXPECT_EQ(sub.bus(), nullptr);
  EXPECT_FALSE(sub.is_owned());
}

TEST_F(ScopedSubscriptionTest, NonOwningSubscription) {
  {
    ScopedSubscription sub(&bus, observer.get());
    EXPECT_TRUE(sub.is_active());
    EXPECT_EQ(sub.observer(), observer.get());
    EXPECT_EQ(sub.bus(), &bus);
    EXPECT_FALSE(sub.is_owned());

    // Observer should be subscribed
    EXPECT_EQ(bus.num_observers(), 1);

    // Test that events are received
    bus.notify(MeshEvent::TopologyChanged);
    EXPECT_EQ(observer->event_count, 1);
  }

  // After scope, observer should be unsubscribed
  EXPECT_EQ(bus.num_observers(), 0);

  // Observer itself should still exist (non-owning)
  EXPECT_NE(observer.get(), nullptr);
}

TEST_F(ScopedSubscriptionTest, OwningSubscription) {
  bool observer_destroyed = false;

  {
    auto shared_obs = std::make_shared<CountingObserver>(&observer_destroyed);
    ScopedSubscription sub(&bus, shared_obs);

    EXPECT_TRUE(sub.is_active());
    EXPECT_EQ(sub.observer(), shared_obs.get());
    EXPECT_EQ(sub.bus(), &bus);
    EXPECT_TRUE(sub.is_owned());

    EXPECT_EQ(bus.num_observers(), 1);

    bus.notify(MeshEvent::GeometryChanged);
    EXPECT_EQ(shared_obs->event_count, 1);

    // Shared observer should not be destroyed yet (held by sub)
    EXPECT_FALSE(observer_destroyed);
  }

  // After scope, observer should be unsubscribed
  EXPECT_EQ(bus.num_observers(), 0);

  // Observer should be destroyed once the subscription and the last shared_ptr go out of scope.
  EXPECT_TRUE(observer_destroyed);
}

TEST_F(ScopedSubscriptionTest, ManualUnsubscribe) {
  ScopedSubscription sub(&bus, observer.get());
  EXPECT_EQ(bus.num_observers(), 1);
  EXPECT_TRUE(sub.is_active());

  sub.unsubscribe();
  EXPECT_EQ(bus.num_observers(), 0);
  EXPECT_FALSE(sub.is_active());
  EXPECT_EQ(sub.observer(), nullptr);
  EXPECT_EQ(sub.bus(), nullptr);

  // Should be safe to call unsubscribe again
  EXPECT_NO_THROW(sub.unsubscribe());
}

TEST_F(ScopedSubscriptionTest, Reset) {
  ScopedSubscription sub(&bus, observer.get());
  EXPECT_TRUE(sub.is_active());
  EXPECT_EQ(bus.num_observers(), 1);

  sub.reset();
  EXPECT_FALSE(sub.is_active());
  EXPECT_EQ(sub.observer(), nullptr);
  EXPECT_EQ(sub.bus(), nullptr);
  // Note: reset() doesn't unsubscribe, just clears the subscription object
}

TEST_F(ScopedSubscriptionTest, Release) {
  ScopedSubscription sub(&bus, observer.get());
  EXPECT_EQ(bus.num_observers(), 1);

  auto released = sub.release();
  EXPECT_EQ(released, observer.get());
  EXPECT_FALSE(sub.is_active());

  // Observer should still be subscribed after release
  EXPECT_EQ(bus.num_observers(), 1);

  // Manually unsubscribe since we released ownership
  bus.unsubscribe(observer.get());
}

TEST_F(ScopedSubscriptionTest, ReleaseOwned) {
  auto shared_obs = std::make_shared<CountingObserver>();
  ScopedSubscription sub(&bus, shared_obs);

  auto released = sub.release();
  EXPECT_EQ(released, nullptr);  // Returns nullptr for owned observers
  EXPECT_FALSE(sub.is_active());
}

TEST_F(ScopedSubscriptionTest, MoveConstructor) {
  ScopedSubscription sub1(&bus, observer.get());
  EXPECT_TRUE(sub1.is_active());
  EXPECT_EQ(bus.num_observers(), 1);

  ScopedSubscription sub2(std::move(sub1));
  EXPECT_FALSE(sub1.is_active());  // sub1 should be moved from
  EXPECT_TRUE(sub2.is_active());   // sub2 should be active
  EXPECT_EQ(sub2.observer(), observer.get());
  EXPECT_EQ(sub2.bus(), &bus);
  EXPECT_EQ(bus.num_observers(), 1);

  // Events should still work through sub2
  bus.notify(MeshEvent::LabelsChanged);
  EXPECT_EQ(observer->event_count, 1);
}

TEST_F(ScopedSubscriptionTest, MoveAssignment) {
  ScopedSubscription sub1(&bus, observer.get());
  ScopedSubscription sub2;

  EXPECT_TRUE(sub1.is_active());
  EXPECT_FALSE(sub2.is_active());
  EXPECT_EQ(bus.num_observers(), 1);

  sub2 = std::move(sub1);
  EXPECT_FALSE(sub1.is_active());  // sub1 should be moved from
  EXPECT_TRUE(sub2.is_active());   // sub2 should be active
  EXPECT_EQ(sub2.observer(), observer.get());
  EXPECT_EQ(bus.num_observers(), 1);
}

TEST_F(ScopedSubscriptionTest, MoveAssignmentWithExisting) {
  CountingObserver observer2;
  ScopedSubscription sub1(&bus, observer.get());
  ScopedSubscription sub2(&bus, &observer2);

  EXPECT_EQ(bus.num_observers(), 2);

  sub2 = std::move(sub1);

  // observer2 should be unsubscribed, only observer1 remains
  EXPECT_EQ(bus.num_observers(), 1);
  EXPECT_TRUE(sub2.is_active());
  EXPECT_EQ(sub2.observer(), observer.get());

  bus.notify(MeshEvent::FieldsChanged);
  EXPECT_EQ(observer->event_count, 1);
  EXPECT_EQ(observer2.event_count, 0);  // No longer subscribed
}

TEST_F(ScopedSubscriptionTest, SelfMoveAssignment) {
  ScopedSubscription sub(&bus, observer.get());
  EXPECT_EQ(bus.num_observers(), 1);

  // Self-assignment should be safe
  sub = std::move(sub);
  EXPECT_TRUE(sub.is_active());
  EXPECT_EQ(bus.num_observers(), 1);
}

TEST_F(ScopedSubscriptionTest, NullObserverHandling) {
  ScopedSubscription sub(&bus, nullptr);
  EXPECT_FALSE(sub.is_active());
  EXPECT_EQ(bus.num_observers(), 0);
}

TEST_F(ScopedSubscriptionTest, NullBusHandling) {
  ScopedSubscription sub(nullptr, observer.get());
  EXPECT_FALSE(sub.is_active());
  // Can't check bus observer count since bus is null
}

TEST_F(ScopedSubscriptionTest, MakeScopedSubscription) {
  auto sub = make_scoped_subscription(&bus, observer.get());
  EXPECT_TRUE(sub.is_active());
  EXPECT_EQ(bus.num_observers(), 1);
}

// ====================
// ScopedSubscriptionGroup Tests
// ====================

class ScopedSubscriptionGroupTest : public ::testing::Test {
protected:
  MeshEventBus bus1;
  MeshEventBus bus2;
  std::unique_ptr<CountingObserver> observer1;
  std::unique_ptr<CountingObserver> observer2;
  std::unique_ptr<CountingObserver> observer3;

  void SetUp() override {
    observer1 = std::make_unique<CountingObserver>();
    observer2 = std::make_unique<CountingObserver>();
    observer3 = std::make_unique<CountingObserver>();
  }
};

TEST_F(ScopedSubscriptionGroupTest, DefaultConstructor) {
  ScopedSubscriptionGroup group;
  EXPECT_EQ(group.size(), 0);
  EXPECT_TRUE(group.empty());
}

TEST_F(ScopedSubscriptionGroupTest, AddNonOwning) {
  ScopedSubscriptionGroup group;

  group.add(&bus1, observer1.get());
  group.add(&bus1, observer2.get());
  group.add(&bus2, observer3.get());

  EXPECT_EQ(group.size(), 3);
  EXPECT_FALSE(group.empty());
  EXPECT_EQ(bus1.num_observers(), 2);
  EXPECT_EQ(bus2.num_observers(), 1);

  // Test that all observers receive events
  bus1.notify(MeshEvent::TopologyChanged);
  EXPECT_EQ(observer1->event_count, 1);
  EXPECT_EQ(observer2->event_count, 1);
  EXPECT_EQ(observer3->event_count, 0);

  bus2.notify(MeshEvent::GeometryChanged);
  EXPECT_EQ(observer1->event_count, 1);
  EXPECT_EQ(observer2->event_count, 1);
  EXPECT_EQ(observer3->event_count, 1);
}

TEST_F(ScopedSubscriptionGroupTest, AddOwning) {
  ScopedSubscriptionGroup group;

  auto shared_obs1 = std::make_shared<CountingObserver>();
  auto shared_obs2 = std::make_shared<CountingObserver>();

  group.add(&bus1, shared_obs1);
  group.add(&bus2, shared_obs2);

  EXPECT_EQ(group.size(), 2);
  EXPECT_EQ(bus1.num_observers(), 1);
  EXPECT_EQ(bus2.num_observers(), 1);
}

TEST_F(ScopedSubscriptionGroupTest, AddExistingSubscription) {
  ScopedSubscriptionGroup group;

  ScopedSubscription sub(&bus1, observer1.get());
  EXPECT_EQ(bus1.num_observers(), 1);

  group.add(std::move(sub));
  EXPECT_FALSE(sub.is_active());  // Moved from
  EXPECT_EQ(group.size(), 1);
  EXPECT_EQ(bus1.num_observers(), 1);  // Still subscribed through group
}

TEST_F(ScopedSubscriptionGroupTest, UnsubscribeAll) {
  ScopedSubscriptionGroup group;

  group.add(&bus1, observer1.get());
  group.add(&bus1, observer2.get());
  group.add(&bus2, observer3.get());

  EXPECT_EQ(bus1.num_observers(), 2);
  EXPECT_EQ(bus2.num_observers(), 1);

  group.unsubscribe_all();

  EXPECT_EQ(bus1.num_observers(), 0);
  EXPECT_EQ(bus2.num_observers(), 0);
  EXPECT_EQ(group.size(), 0);  // All inactive
}

TEST_F(ScopedSubscriptionGroupTest, Clear) {
  ScopedSubscriptionGroup group;

  group.add(&bus1, observer1.get());
  group.add(&bus2, observer2.get());

  EXPECT_EQ(group.size(), 2);

  group.clear();

  EXPECT_TRUE(group.empty());
  // Note: clear() removes subscriptions from group but doesn't unsubscribe
  // The destructor of the subscriptions should handle unsubscribing
}

TEST_F(ScopedSubscriptionGroupTest, AutoUnsubscribeOnDestruction) {
  {
    ScopedSubscriptionGroup group;
    group.add(&bus1, observer1.get());
    group.add(&bus2, observer2.get());

    EXPECT_EQ(bus1.num_observers(), 1);
    EXPECT_EQ(bus2.num_observers(), 1);
  }

  // After group destruction, all should be unsubscribed
  EXPECT_EQ(bus1.num_observers(), 0);
  EXPECT_EQ(bus2.num_observers(), 0);
}

TEST_F(ScopedSubscriptionGroupTest, MoveConstructor) {
  ScopedSubscriptionGroup group1;
  group1.add(&bus1, observer1.get());
  group1.add(&bus2, observer2.get());

  EXPECT_EQ(group1.size(), 2);

  ScopedSubscriptionGroup group2(std::move(group1));

  EXPECT_EQ(group2.size(), 2);
  // group1 size is undefined after move

  EXPECT_EQ(bus1.num_observers(), 1);
  EXPECT_EQ(bus2.num_observers(), 1);
}

TEST_F(ScopedSubscriptionGroupTest, MoveAssignment) {
  ScopedSubscriptionGroup group1;
  group1.add(&bus1, observer1.get());

  ScopedSubscriptionGroup group2;
  group2.add(&bus2, observer2.get());

  EXPECT_EQ(bus1.num_observers(), 1);
  EXPECT_EQ(bus2.num_observers(), 1);

  group2 = std::move(group1);

  // group2's original subscription should be cleaned up
  EXPECT_EQ(bus2.num_observers(), 0);
  EXPECT_EQ(bus1.num_observers(), 1);  // Still has group1's subscription
}

TEST_F(ScopedSubscriptionGroupTest, MixedSubscriptionTypes) {
  ScopedSubscriptionGroup group;

  // Add non-owning
  group.add(&bus1, observer1.get());

  // Add owning
  auto shared_obs = std::make_shared<CountingObserver>();
  group.add(&bus1, shared_obs);

  // Add existing subscription
  ScopedSubscription sub(&bus2, observer2.get());
  group.add(std::move(sub));

  EXPECT_EQ(group.size(), 3);
  EXPECT_EQ(bus1.num_observers(), 2);
  EXPECT_EQ(bus2.num_observers(), 1);
}

// ====================
// RAII Behavior Tests
// ====================

TEST(ScopedSubscriptionRAIITest, ExceptionSafety) {
  MeshEventBus bus;
  CountingObserver observer;

  try {
    ScopedSubscription sub(&bus, &observer);
    EXPECT_EQ(bus.num_observers(), 1);

    // Simulate exception
    throw std::runtime_error("Test exception");
  } catch (...) {
    // After exception, subscription should be cleaned up
    EXPECT_EQ(bus.num_observers(), 0);
  }
}

TEST(ScopedSubscriptionRAIITest, NestedScopes) {
  MeshEventBus bus;
  CountingObserver obs1, obs2, obs3;

  {
    ScopedSubscription sub1(&bus, &obs1);
    EXPECT_EQ(bus.num_observers(), 1);

    {
      ScopedSubscription sub2(&bus, &obs2);
      EXPECT_EQ(bus.num_observers(), 2);

      {
        ScopedSubscription sub3(&bus, &obs3);
        EXPECT_EQ(bus.num_observers(), 3);
      }
      EXPECT_EQ(bus.num_observers(), 2);  // sub3 unsubscribed
    }
    EXPECT_EQ(bus.num_observers(), 1);  // sub2 unsubscribed
  }
  EXPECT_EQ(bus.num_observers(), 0);  // sub1 unsubscribed
}

TEST(ScopedSubscriptionRAIITest, ConditionalSubscription) {
  MeshEventBus bus;
  CountingObserver observer;

  auto create_conditional_subscription = [&](bool should_subscribe) {
    if (should_subscribe) {
      return ScopedSubscription(&bus, &observer);
    }
    return ScopedSubscription();
  };

  {
    auto sub1 = create_conditional_subscription(true);
    EXPECT_TRUE(sub1.is_active());
    EXPECT_EQ(bus.num_observers(), 1);
  }
  EXPECT_EQ(bus.num_observers(), 0);

  {
    auto sub2 = create_conditional_subscription(false);
    EXPECT_FALSE(sub2.is_active());
    EXPECT_EQ(bus.num_observers(), 0);
  }
}
