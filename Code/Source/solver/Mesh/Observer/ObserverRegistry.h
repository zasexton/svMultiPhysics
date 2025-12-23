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

#ifndef SVMP_OBSERVER_REGISTRY_H
#define SVMP_OBSERVER_REGISTRY_H

#include "MeshObserver.h"
#include "../Core/MeshBase.h"
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace svmp {

// Forward declarations
class MeshBase;

/**
 * @brief Central registry for managing common mesh observers
 *
 * Provides convenience functions for attaching/detaching standard observers
 * and utilities for querying active observers for diagnostics.
 */
class ObserverRegistry {
public:
  /**
   * @brief Attach a search structure invalidator to a mesh
   *
   * Automatically invalidates the search accelerator when the mesh topology
   * or geometry changes, and on partition changes for distributed meshes.
   *
   * @param mesh The mesh to observe
   * @param accel The search accelerator to invalidate
   * @return Shared pointer to the created observer
   */
  template<typename AccelType>
  static std::shared_ptr<MeshObserver> attach_search_invalidator(
      MeshEventBus& bus, AccelType& accel);

  template<typename AccelType>
  static std::shared_ptr<MeshObserver> attach_search_invalidator(
      MeshBase& mesh, AccelType& accel);

  /**
   * @brief Attach a geometry cache invalidator to a mesh
   *
   * Invalidates the geometry cache when mesh geometry or topology changes.
   *
   * @param bus The mesh event bus to subscribe to
   * @param cache The cache to invalidate
   * @return Shared pointer to the created observer
   */
  template<typename CacheType>
  static std::shared_ptr<MeshObserver> attach_geometry_cache_invalidator(
      MeshEventBus& bus, CacheType& cache);

  template<typename CacheType>
  static std::shared_ptr<MeshObserver> attach_geometry_cache_invalidator(
      MeshBase& mesh, CacheType& cache);

  /**
   * @brief Attach a label-dependent cache invalidator
   *
   * Invalidates caches that depend on region/boundary labels.
   *
   * @param bus The mesh event bus to subscribe to
   * @param cache The cache to invalidate
   * @return Shared pointer to the created observer
   */
  template<typename CacheType>
  static std::shared_ptr<MeshObserver> attach_label_cache_invalidator(
      MeshEventBus& bus, CacheType& cache);

  template<typename CacheType>
  static std::shared_ptr<MeshObserver> attach_label_cache_invalidator(
      MeshBase& mesh, CacheType& cache);

  /**
   * @brief Attach a field-dependent cache invalidator
   *
   * Invalidates caches that depend on attached fields.
   *
   * @param bus The mesh event bus to subscribe to
   * @param cache The cache to invalidate
   * @return Shared pointer to the created observer
   */
  template<typename CacheType>
  static std::shared_ptr<MeshObserver> attach_field_cache_invalidator(
      MeshEventBus& bus, CacheType& cache);

  template<typename CacheType>
  static std::shared_ptr<MeshObserver> attach_field_cache_invalidator(
      MeshBase& mesh, CacheType& cache);

  /**
   * @brief Attach an event logger for debugging
   *
   * @param bus The mesh event bus to subscribe to
   * @param prefix Prefix for log messages (e.g., "MeshBase", "DistributedMesh")
   * @param enabled Whether the logger is initially enabled
   * @return Shared pointer to the created observer
   */
  static std::shared_ptr<MeshObserver> attach_event_logger(
      MeshEventBus& bus,
      const std::string& prefix = "Mesh",
      bool enabled = true);

  static std::shared_ptr<MeshObserver> attach_event_logger(
      MeshBase& mesh,
      const std::string& prefix = "Mesh",
      bool enabled = true);

  /**
   * @brief Attach an event counter for diagnostics
   *
   * @param bus The mesh event bus to subscribe to
   * @return Shared pointer to the created event counter
   */
  static std::shared_ptr<EventCounter> attach_event_counter(MeshEventBus& bus);
  static std::shared_ptr<EventCounter> attach_event_counter(MeshBase& mesh);

  /**
   * @brief Create a multi-event cache invalidator
   *
   * Invalidates a cache on multiple specified events.
   *
   * @param cache The cache to invalidate
   * @param events Events that should trigger invalidation
   * @return Shared pointer to the created observer
   */
  template<typename CacheType>
  static std::shared_ptr<MeshObserver> create_multi_event_invalidator(
      CacheType& cache,
      const std::vector<MeshEvent>& events);

  /**
   * @brief Registry entry for tracking attached observers
   */
  struct ObserverEntry {
    std::string name;
    std::string type;
    std::weak_ptr<MeshObserver> observer;
    std::string mesh_id;
    std::weak_ptr<MeshEventBus::State> bus_state;
  };

  /**
   * @brief Global registry instance
   */
  static ObserverRegistry& instance();

  /**
   * @brief Register an observer with the global registry
   */
  void register_observer(const std::string& mesh_id,
                         const std::string& name,
                         const std::string& type,
                         std::weak_ptr<MeshObserver> observer,
                         std::weak_ptr<MeshEventBus::State> bus_state = {});

  /**
   * @brief Get all registered observers for a mesh
   */
  std::vector<ObserverEntry> get_observers(const std::string& mesh_id) const;

  /**
   * @brief Get all registered observers
   */
  std::vector<ObserverEntry> get_all_observers() const;

  /**
   * @brief Clean up expired weak pointers
   */
  void cleanup_expired();

  /**
   * @brief Generate a diagnostic report
   */
  std::string diagnostic_report() const;

private:
  ObserverRegistry() = default;

  mutable std::mutex mutex_;
  mutable std::map<std::string, std::vector<ObserverEntry>> registry_;

  static std::string bus_id(const MeshEventBus& bus) {
    return "MeshEventBus@" + std::to_string(reinterpret_cast<std::uintptr_t>(&bus));
  }

  static std::string mesh_id(const MeshBase& mesh) {
    return mesh.mesh_id();
  }
};

// ====================
// Template Implementations
// ====================

// Multi-event invalidator observer
template<typename CacheType>
class MultiEventInvalidator : public MeshObserver {
public:
  MultiEventInvalidator(CacheType& cache, const std::vector<MeshEvent>& events)
      : cache_(cache), trigger_events_(events) {}

  void on_mesh_event(MeshEvent event) override {
    for (auto trigger : trigger_events_) {
      if (event == trigger) {
        cache_.invalidate();
        break;
      }
    }
  }

  const char* observer_name() const override {
    return "MultiEventInvalidator";
  }

private:
  CacheType& cache_;
  std::vector<MeshEvent> trigger_events_;
};

// Search invalidator - responds to topology/geometry/partition/adaptivity changes
template<typename AccelType>
  class SearchInvalidator : public MeshObserver {
  public:
	  explicit SearchInvalidator(AccelType& accel) : accel_(accel) {}

	  void on_mesh_event(MeshEvent event) override {
	    if (event == MeshEvent::TopologyChanged ||
	        event == MeshEvent::GeometryChanged ||
	        event == MeshEvent::PartitionChanged ||
	        event == MeshEvent::AdaptivityApplied) {
	      accel_.invalidate();
	    }
	  }

  const char* observer_name() const override {
    return "SearchInvalidator";
  }

private:
  AccelType& accel_;
};

template<typename AccelType>
std::shared_ptr<MeshObserver> ObserverRegistry::attach_search_invalidator(
    MeshEventBus& bus, AccelType& accel) {
  const auto id = bus_id(bus);
  auto observer = std::make_shared<SearchInvalidator<AccelType>>(accel);
  bus.subscribe(observer);

  instance().register_observer(
      id,
      "SearchInvalidator",
      "SearchAccel",
      observer,
      bus.weak_state()
  );

  return observer;
}

template<typename AccelType>
std::shared_ptr<MeshObserver> ObserverRegistry::attach_search_invalidator(
    MeshBase& mesh, AccelType& accel) {
  const auto id = mesh_id(mesh);
  auto observer = std::make_shared<SearchInvalidator<AccelType>>(accel);
  mesh.event_bus().subscribe(observer);

  // Register with global registry for diagnostics
  instance().register_observer(
      id,
      "SearchInvalidator",
      "SearchAccel",
      observer,
      mesh.event_bus().weak_state()
  );

  return observer;
}

template<typename CacheType>
std::shared_ptr<MeshObserver> ObserverRegistry::attach_geometry_cache_invalidator(
    MeshEventBus& bus, CacheType& cache) {
  const auto id = bus_id(bus);
  auto observer = std::make_shared<CacheInvalidator<CacheType>>(cache);
  bus.subscribe(observer);

  instance().register_observer(
      id,
      "GeometryCacheInvalidator",
      "GeometryCache",
      observer,
      bus.weak_state()
  );

  return observer;
}

template<typename CacheType>
std::shared_ptr<MeshObserver> ObserverRegistry::attach_geometry_cache_invalidator(
    MeshBase& mesh, CacheType& cache) {
  const auto id = mesh_id(mesh);
  auto observer = std::make_shared<CacheInvalidator<CacheType>>(cache);
  mesh.event_bus().subscribe(observer);

  instance().register_observer(
      id,
      "GeometryCacheInvalidator",
      "GeometryCache",
      observer,
      mesh.event_bus().weak_state()
  );

  return observer;
}

template<typename CacheType>
std::shared_ptr<MeshObserver> ObserverRegistry::attach_label_cache_invalidator(
    MeshEventBus& bus, CacheType& cache) {
  const auto id = bus_id(bus);
  auto observer = create_multi_event_invalidator(cache, {MeshEvent::LabelsChanged});
  bus.subscribe(observer);

  instance().register_observer(
      id,
      "LabelCacheInvalidator",
      "LabelCache",
      observer,
      bus.weak_state()
  );

  return observer;
}

template<typename CacheType>
std::shared_ptr<MeshObserver> ObserverRegistry::attach_label_cache_invalidator(
    MeshBase& mesh, CacheType& cache) {
  const auto id = mesh_id(mesh);
  auto observer = create_multi_event_invalidator(cache, {MeshEvent::LabelsChanged});
  mesh.event_bus().subscribe(observer);

  instance().register_observer(
      id,
      "LabelCacheInvalidator",
      "LabelCache",
      observer,
      mesh.event_bus().weak_state()
  );

  return observer;
}

template<typename CacheType>
std::shared_ptr<MeshObserver> ObserverRegistry::attach_field_cache_invalidator(
    MeshEventBus& bus, CacheType& cache) {
  const auto id = bus_id(bus);
  auto observer = create_multi_event_invalidator(cache, {MeshEvent::FieldsChanged});
  bus.subscribe(observer);

  instance().register_observer(
      id,
      "FieldCacheInvalidator",
      "FieldCache",
      observer,
      bus.weak_state()
  );

  return observer;
}

template<typename CacheType>
std::shared_ptr<MeshObserver> ObserverRegistry::attach_field_cache_invalidator(
    MeshBase& mesh, CacheType& cache) {
  const auto id = mesh_id(mesh);
  auto observer = create_multi_event_invalidator(cache, {MeshEvent::FieldsChanged});
  mesh.event_bus().subscribe(observer);

  instance().register_observer(
      id,
      "FieldCacheInvalidator",
      "FieldCache",
      observer,
      mesh.event_bus().weak_state()
  );

  return observer;
}

template<typename CacheType>
std::shared_ptr<MeshObserver> ObserverRegistry::create_multi_event_invalidator(
    CacheType& cache, const std::vector<MeshEvent>& events) {
  return std::make_shared<MultiEventInvalidator<CacheType>>(cache, events);
}

} // namespace svmp

#endif // SVMP_OBSERVER_REGISTRY_H
