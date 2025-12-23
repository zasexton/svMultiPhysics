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

#ifndef SVMP_MESH_OBSERVER_H
#define SVMP_MESH_OBSERVER_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

namespace svmp {

// ====================
// P0 #5: Event/Observer Bus (Cache Invalidation + Coupling Hooks)
// ====================
// Mesh emits events; subsystems subscribe.
// Keeps caches coherent without tight coupling.
//
// Events: topology_change, geometry_change, partition_change, labels_change, fields_change, adaptivity_applied
// Subscribers: GeometryCache, BVH, InterfaceMesh, physics modules, etc.

// ====================
// Mesh Events
// ====================
enum class MeshEvent {
  TopologyChanged,   // cells/faces/edges added/removed/reordered
  GeometryChanged,   // coordinates modified (X_ref or X_cur)
  PartitionChanged,  // ownership/ghost layer changed (MPI redistribution)
  LabelsChanged,     // region/boundary labels modified
  FieldsChanged,     // field attached/removed/modified
  AdaptivityApplied  // mesh refinement/coarsening applied
};

// Human-readable event names
inline const char* event_name(MeshEvent evt) {
  switch (evt) {
    case MeshEvent::TopologyChanged:   return "TopologyChanged";
    case MeshEvent::GeometryChanged:   return "GeometryChanged";
    case MeshEvent::PartitionChanged:  return "PartitionChanged";
    case MeshEvent::LabelsChanged:     return "LabelsChanged";
    case MeshEvent::FieldsChanged:     return "FieldsChanged";
    case MeshEvent::AdaptivityApplied: return "AdaptivityApplied";
  }
  return "Unknown";
}

// ====================
// Observer Interface
// ====================
class MeshObserver {
public:
  virtual ~MeshObserver() = default;

  // Called when a mesh event occurs
  virtual void on_mesh_event(MeshEvent event) = 0;

  // Optional: get observer name for debugging
  virtual const char* observer_name() const { return "MeshObserver"; }
};

// ====================
// Event Bus
// ====================
// Central dispatcher for mesh events.
// Meshes hold a MeshEventBus and call notify() when they change.

class MeshEventBus {
public:
  struct State {
    mutable std::mutex mutex;
    std::vector<MeshObserver*> observers;                        // non-owning pointers
    std::vector<std::shared_ptr<MeshObserver>> owned_observers;  // owned observers
  };

  MeshEventBus() : state_(std::make_shared<State>()) {}
  MeshEventBus(const MeshEventBus&) = delete;
  MeshEventBus& operator=(const MeshEventBus&) = delete;
  MeshEventBus(MeshEventBus&&) noexcept = default;
  MeshEventBus& operator=(MeshEventBus&&) noexcept = default;

  std::weak_ptr<State> weak_state() const noexcept { return state_; }

  // Register an observer (lifetime must exceed this bus)
  void subscribe(MeshObserver* observer) {
    if (!observer) {
      return;
    }

    auto state = state_;
    if (!state) {
      return;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    if (std::find(state->observers.begin(), state->observers.end(), observer) == state->observers.end()) {
      state->observers.push_back(observer);
    }
  }

  // Register an observer with shared ownership
  void subscribe(std::shared_ptr<MeshObserver> observer) {
    if (!observer) {
      return;
    }

    auto state = state_;
    if (!state) {
      return;
    }

    std::lock_guard<std::mutex> lock(state->mutex);

    // Avoid retaining duplicate shared_ptr references for the same observer instance.
    auto* raw = observer.get();
    const bool already_owned = std::any_of(
        state->owned_observers.begin(), state->owned_observers.end(),
        [raw](const std::shared_ptr<MeshObserver>& existing) { return existing.get() == raw; });
    if (!already_owned) {
      state->owned_observers.push_back(std::move(observer));
    }

    // Ensure the raw pointer is subscribed (deduplicated by subscribe(MeshObserver*)).
    if (std::find(state->observers.begin(), state->observers.end(), raw) == state->observers.end()) {
      state->observers.push_back(raw);
    }
  }

  // Unregister an observer
  void unsubscribe(MeshObserver* observer) {
    auto state = state_;
    if (!state) {
      return;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    state->observers.erase(std::remove(state->observers.begin(), state->observers.end(), observer), state->observers.end());
    state->owned_observers.erase(
        std::remove_if(state->owned_observers.begin(), state->owned_observers.end(),
                       [observer](const std::shared_ptr<MeshObserver>& owned) {
                         return owned.get() == observer;
                       }),
        state->owned_observers.end());
  }

  // Notify all observers of an event
  void notify(MeshEvent event) {
    auto state = state_;
    if (!state) {
      return;
    }

    // Snapshot iteration to tolerate subscribe/unsubscribe and avoid holding
    // locks during callbacks. Subscribers added/removed during notification
    // will take effect on the next notify() call.
    std::vector<MeshObserver*> snapshot;
    std::vector<std::shared_ptr<MeshObserver>> owned_snapshot;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      snapshot = state->observers;
      owned_snapshot = state->owned_observers; // keep owned observers alive for the duration
    }

    (void)owned_snapshot;
    for (auto* obs : snapshot) {
      if (obs) {
        obs->on_mesh_event(event);
      }
    }
  }

  // Query
  size_t num_observers() const {
    auto state = state_;
    if (!state) {
      return 0;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return state->observers.size();
  }
  bool has_observers() const {
    auto state = state_;
    if (!state) {
      return false;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return !state->observers.empty();
  }

  bool is_subscribed(const MeshObserver* observer) const {
    auto state = state_;
    if (!state || !observer) {
      return false;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return std::find(state->observers.begin(), state->observers.end(), observer) != state->observers.end();
  }

  // Clear all observers
  void clear() {
    auto state = state_;
    if (!state) {
      return;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    state->observers.clear();
    state->owned_observers.clear();
  }

private:
  std::shared_ptr<State> state_;
};

// ====================
// Common Observer Implementations
// ====================

// Observer that invalidates a cache on geometry changes
template <typename Cache>
class CacheInvalidator : public MeshObserver {
public:
  explicit CacheInvalidator(Cache& cache) : cache_(cache) {}

  void on_mesh_event(MeshEvent event) override {
    if (event == MeshEvent::GeometryChanged || event == MeshEvent::TopologyChanged) {
      cache_.invalidate();
    }
  }

  const char* observer_name() const override { return "CacheInvalidator"; }

private:
  Cache& cache_;
};

// Observer that logs events to stdout (useful for debugging)
class EventLogger : public MeshObserver {
public:
  explicit EventLogger(const char* prefix = "Mesh") : prefix_(prefix) {}

  void on_mesh_event(MeshEvent event) override {
    std::cout << "[" << prefix_ << "] Event: " << event_name(event) << std::endl;
  }

  const char* observer_name() const override { return "EventLogger"; }

private:
  const char* prefix_;
};

// Observer that counts events by type
class EventCounter : public MeshObserver {
public:
  EventCounter() {
    counts_.resize(static_cast<size_t>(MeshEvent::AdaptivityApplied) + 1, 0);
  }

  void on_mesh_event(MeshEvent event) override {
    counts_[static_cast<size_t>(event)]++;
  }

  size_t count(MeshEvent event) const {
    return counts_[static_cast<size_t>(event)];
  }

  void reset() {
    std::fill(counts_.begin(), counts_.end(), 0);
  }

  const char* observer_name() const override { return "EventCounter"; }

private:
  std::vector<size_t> counts_;
};

// Observer that triggers a callback
class CallbackObserver : public MeshObserver {
public:
  using Callback = std::function<void(MeshEvent)>;

  explicit CallbackObserver(Callback cb) : callback_(std::move(cb)) {}

  void on_mesh_event(MeshEvent event) override {
    if (callback_) {
      callback_(event);
    }
  }

  const char* observer_name() const override { return "CallbackObserver"; }

private:
  Callback callback_;
};

} // namespace svmp

#endif // SVMP_MESH_OBSERVER_H
