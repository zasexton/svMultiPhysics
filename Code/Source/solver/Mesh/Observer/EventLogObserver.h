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

#ifndef SVMP_EVENT_LOG_OBSERVER_H
#define SVMP_EVENT_LOG_OBSERVER_H

#include "MeshObserver.h"
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sstream>

namespace svmp {

/**
 * @brief Rich event logger with throttling, filtering, and sink injection
 *
 * Provides advanced logging capabilities for mesh events:
 * - Filter events by type
 * - Throttle repeated events
 * - Direct output to stdout, file, or custom sink
 * - Include timestamps and event counts
 * - Format output with various detail levels
 */
class EventLogObserver : public MeshObserver {
public:
  /**
   * @brief Log detail level
   */
  enum class LogLevel {
    MINIMAL,   // Event type only
    NORMAL,    // Event type + count
    DETAILED,  // Event type + count + time since last
    VERBOSE    // All details + cumulative stats
  };

  /**
   * @brief Output sink type
   */
  using LogSink = std::function<void(const std::string&)>;

  /**
   * @brief Configuration for the event logger
   */
  struct Config {
    std::string prefix = "Mesh";
    LogLevel level = LogLevel::NORMAL;
    bool include_timestamp = true;
    bool throttle_enabled = false;
    std::chrono::milliseconds throttle_interval{1000};  // Default 1 second
    std::set<MeshEvent> filter_events;  // Empty = log all events
    bool filter_mode_include = true;    // true = include only filtered, false = exclude filtered
  };

  /**
   * @brief Construct with default configuration (stdout output)
   */
  explicit EventLogObserver(const std::string& prefix = "Mesh");

  /**
   * @brief Construct with custom configuration (stdout output)
   */
  explicit EventLogObserver(const Config& config);

  /**
   * @brief Construct with custom sink
   */
  EventLogObserver(const Config& config, LogSink sink);

  /**
   * @brief Handle mesh event
   */
  void on_mesh_event(MeshEvent event) override;

  /**
   * @brief Get observer name
   */
  const char* observer_name() const override { return "EventLogObserver"; }

  // Configuration methods

  /**
   * @brief Set log detail level
   */
  void set_log_level(LogLevel level) { config_.level = level; }

  /**
   * @brief Enable/disable event throttling
   */
  void set_throttling(bool enabled, std::chrono::milliseconds interval = std::chrono::milliseconds{1000});

  /**
   * @brief Set event filter
   *
   * @param events Events to include/exclude
   * @param include_mode If true, only log events in the set. If false, exclude events in the set.
   */
  void set_filter(const std::set<MeshEvent>& events, bool include_mode = true);

  /**
   * @brief Clear event filter (log all events)
   */
  void clear_filter();

  /**
   * @brief Enable/disable timestamps
   */
  void set_timestamps(bool enabled) { config_.include_timestamp = enabled; }

  /**
   * @brief Set output sink
   */
  void set_sink(LogSink sink) { sink_ = sink; }

  /**
   * @brief Set output to file
   */
  void set_file_output(const std::string& filename);

  /**
   * @brief Set output to stdout
   */
  void set_stdout_output();

  /**
   * @brief Set output to string stream (useful for testing)
   */
  void set_string_output(std::shared_ptr<std::stringstream> stream);

  // Query methods

  /**
   * @brief Get total event count
   */
  size_t total_events() const { return total_events_; }

  /**
   * @brief Get count for specific event type
   */
  size_t event_count(MeshEvent event) const;

  /**
   * @brief Get throttled (dropped) event count
   */
  size_t throttled_events() const { return throttled_events_; }

  /**
   * @brief Reset all counters
   */
  void reset_counters();

  /**
   * @brief Generate summary report
   */
  std::string summary_report() const;

private:
  /**
   * @brief Check if event should be logged based on filter
   */
  bool should_log(MeshEvent event) const;

  /**
   * @brief Check if event is throttled
   */
  bool is_throttled(MeshEvent event);

  /**
   * @brief Format log message
   */
  std::string format_message(MeshEvent event) const;

  /**
   * @brief Get current timestamp string
   */
  std::string get_timestamp() const;

  Config config_;
  LogSink sink_;

  // Statistics
  size_t total_events_ = 0;
  size_t throttled_events_ = 0;
  std::map<MeshEvent, size_t> event_counts_;

  // Throttling state
  std::map<MeshEvent, std::chrono::steady_clock::time_point> last_log_time_;

  // Timing
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_event_time_;

  // File output (if used)
  std::unique_ptr<std::ofstream> file_stream_;

  // String output (if used)
  std::shared_ptr<std::stringstream> string_stream_;
};

/**
 * @brief Builder pattern for creating EventLogObserver with fluent interface
 */
class EventLogBuilder {
public:
  EventLogBuilder() = default;

  EventLogBuilder& with_prefix(const std::string& prefix) {
    config_.prefix = prefix;
    return *this;
  }

  EventLogBuilder& with_level(EventLogObserver::LogLevel level) {
    config_.level = level;
    return *this;
  }

  EventLogBuilder& with_timestamps(bool enabled = true) {
    config_.include_timestamp = enabled;
    return *this;
  }

  EventLogBuilder& with_throttling(std::chrono::milliseconds interval) {
    config_.throttle_enabled = true;
    config_.throttle_interval = interval;
    return *this;
  }

  EventLogBuilder& with_filter(const std::set<MeshEvent>& events, bool include_mode = true) {
    config_.filter_events = events;
    config_.filter_mode_include = include_mode;
    return *this;
  }

  EventLogBuilder& to_file(const std::string& filename) {
    filename_ = filename;
    use_file_ = true;
    return *this;
  }

  EventLogBuilder& to_stdout() {
    use_file_ = false;
    return *this;
  }

  std::shared_ptr<EventLogObserver> build() const;

private:
  EventLogObserver::Config config_;
  std::string filename_;
  bool use_file_ = false;
};

} // namespace svmp

#endif // SVMP_EVENT_LOG_OBSERVER_H