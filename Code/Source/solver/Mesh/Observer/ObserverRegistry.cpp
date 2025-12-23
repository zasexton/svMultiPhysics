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

#include "ObserverRegistry.h"
#include <sstream>
#include <iomanip>

namespace svmp {

// ====================
// Configurable Event Logger
// ====================
class ConfigurableEventLogger : public MeshObserver {
public:
  ConfigurableEventLogger(const std::string& prefix, bool enabled)
      : prefix_(prefix), enabled_(enabled) {}

  void on_mesh_event(MeshEvent event) override {
    if (enabled_) {
      std::cout << "[" << prefix_ << "] Event: " << event_name(event) << std::endl;
    }
  }

  void set_enabled(bool enabled) { enabled_ = enabled; }
  bool is_enabled() const { return enabled_; }

  const char* observer_name() const override { return "ConfigurableEventLogger"; }

private:
  std::string prefix_;
  bool enabled_;
};

// ====================
// ObserverRegistry Implementation
// ====================

ObserverRegistry& ObserverRegistry::instance() {
  static ObserverRegistry instance;
  return instance;
}

void ObserverRegistry::register_observer(const std::string& mesh_id,
                                        const std::string& name,
                                        const std::string& type,
                                        std::weak_ptr<MeshObserver> observer,
                                        std::weak_ptr<MeshEventBus::State> bus_state) {
  std::lock_guard<std::mutex> lock(mutex_);
  ObserverEntry entry{name, type, observer, mesh_id, std::move(bus_state)};
  registry_[mesh_id].push_back(entry);
}

std::vector<ObserverRegistry::ObserverEntry>
ObserverRegistry::get_observers(const std::string& mesh_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = registry_.find(mesh_id);
  if (it != registry_.end()) {
    return it->second;
  }
  return {};
}

std::vector<ObserverRegistry::ObserverEntry>
ObserverRegistry::get_all_observers() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ObserverEntry> all_entries;
  for (const auto& [mesh_id, entries] : registry_) {
    all_entries.insert(all_entries.end(), entries.begin(), entries.end());
  }
  return all_entries;
}

void ObserverRegistry::cleanup_expired() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [mesh_id, entries] : registry_) {
    entries.erase(
      std::remove_if(entries.begin(), entries.end(),
                     [](const ObserverEntry& entry) {
                       return entry.observer.expired();
                     }),
      entries.end()
    );
  }

  // Remove empty mesh entries
  std::vector<std::string> empty_meshes;
  for (const auto& [mesh_id, entries] : registry_) {
    if (entries.empty()) {
      empty_meshes.push_back(mesh_id);
    }
  }
  for (const auto& mesh_id : empty_meshes) {
    registry_.erase(mesh_id);
  }
}

std::string ObserverRegistry::diagnostic_report() const {
  std::map<std::string, std::vector<ObserverEntry>> registry_snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    registry_snapshot = registry_;
  }
  std::stringstream report;
  report << "=== Observer Registry Diagnostic Report ===" << std::endl;
  report << "Total meshes monitored: " << registry_snapshot.size() << std::endl;

  size_t total_observers = 0;
  size_t active_observers = 0;
  size_t subscribed_observers = 0;
  size_t detached_observers = 0;

  for (const auto& [mesh_id, entries] : registry_snapshot) {
    report << std::endl << "Mesh ID: " << mesh_id << std::endl;
    report << "  Observers: " << entries.size() << std::endl;

    for (const auto& entry : entries) {
      total_observers++;
      const auto obs = entry.observer.lock();
      bool is_active = static_cast<bool>(obs);
      if (is_active) {
        active_observers++;
      }

      std::string status;
      if (!is_active) {
        status = "EXPIRED";
      } else if (const auto bus_state = entry.bus_state.lock()) {
        bool subscribed = false;
        {
          std::lock_guard<std::mutex> lock(bus_state->mutex);
          subscribed = std::find(bus_state->observers.begin(),
                                 bus_state->observers.end(),
                                 obs.get()) != bus_state->observers.end();
        }
        if (subscribed) {
          status = "ACTIVE/SUBSCRIBED";
          subscribed_observers++;
        } else {
          status = "ACTIVE/DETACHED";
          detached_observers++;
        }
      } else {
        status = "ACTIVE/UNKNOWN";
      }

      report << "    - " << std::left << std::setw(25) << entry.name
             << " [" << std::setw(15) << entry.type << "] "
             << status << std::endl;
    }
  }

  report << std::endl << "Summary:" << std::endl;
  report << "  Total observers: " << total_observers << std::endl;
  report << "  Active observers: " << active_observers << std::endl;
  report << "  Subscribed observers: " << subscribed_observers << std::endl;
  report << "  Detached observers: " << detached_observers << std::endl;
  report << "  Expired observers: " << (total_observers - active_observers) << std::endl;

  return report.str();
}

std::shared_ptr<MeshObserver> ObserverRegistry::attach_event_logger(
    MeshEventBus& bus,
    const std::string& prefix,
    bool enabled) {
  auto logger = std::make_shared<ConfigurableEventLogger>(prefix, enabled);
  bus.subscribe(logger);

  instance().register_observer(
      bus_id(bus),
      "EventLogger",
      "Logging",
      logger,
      bus.weak_state()
  );

  return logger;
}

std::shared_ptr<MeshObserver> ObserverRegistry::attach_event_logger(
    MeshBase& mesh,
    const std::string& prefix,
    bool enabled) {
  auto logger = std::make_shared<ConfigurableEventLogger>(prefix, enabled);
  mesh.event_bus().subscribe(logger);

  instance().register_observer(
      mesh_id(mesh),
      "EventLogger",
      "Logging",
      logger,
      mesh.event_bus().weak_state()
  );

  return logger;
}

std::shared_ptr<EventCounter> ObserverRegistry::attach_event_counter(
    MeshEventBus& bus) {
  auto counter = std::make_shared<EventCounter>();
  bus.subscribe(counter);

  instance().register_observer(
      bus_id(bus),
      "EventCounter",
      "Diagnostics",
      counter,
      bus.weak_state()
  );

  return counter;
}

std::shared_ptr<EventCounter> ObserverRegistry::attach_event_counter(
    MeshBase& mesh) {
  auto counter = std::make_shared<EventCounter>();
  mesh.event_bus().subscribe(counter);

  instance().register_observer(
      mesh_id(mesh),
      "EventCounter",
      "Diagnostics",
      counter,
      mesh.event_bus().weak_state()
  );

  return counter;
}

} // namespace svmp
