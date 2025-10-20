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

#include "EventLogObserver.h"
#include <ctime>
#include <iomanip>
#include <iostream>

namespace svmp {

// ====================
// EventLogObserver Implementation
// ====================

EventLogObserver::EventLogObserver(const std::string& prefix)
    : EventLogObserver(Config{prefix}) {
}

EventLogObserver::EventLogObserver(const Config& config)
    : EventLogObserver(config, [](const std::string& msg) { std::cout << msg << std::endl; }) {
}

EventLogObserver::EventLogObserver(const Config& config, LogSink sink)
    : config_(config),
      sink_(sink),
      start_time_(std::chrono::steady_clock::now()),
      last_event_time_(start_time_) {
  if (!sink_) {
    sink_ = [](const std::string& msg) { std::cout << msg << std::endl; };
  }
}

void EventLogObserver::on_mesh_event(MeshEvent event) {
  // Update statistics
  total_events_++;
  event_counts_[event]++;

  // Check filter
  if (!should_log(event)) {
    return;
  }

  // Check throttling
  if (config_.throttle_enabled && is_throttled(event)) {
    throttled_events_++;
    return;
  }

  // Format and output message
  std::string message = format_message(event);
  sink_(message);

  // Update timing
  last_event_time_ = std::chrono::steady_clock::now();
  if (config_.throttle_enabled) {
    last_log_time_[event] = last_event_time_;
  }
}

void EventLogObserver::set_throttling(bool enabled, std::chrono::milliseconds interval) {
  config_.throttle_enabled = enabled;
  config_.throttle_interval = interval;
  if (!enabled) {
    last_log_time_.clear();
  }
}

void EventLogObserver::set_filter(const std::set<MeshEvent>& events, bool include_mode) {
  config_.filter_events = events;
  config_.filter_mode_include = include_mode;
}

void EventLogObserver::clear_filter() {
  config_.filter_events.clear();
}

void EventLogObserver::set_file_output(const std::string& filename) {
  file_stream_ = std::make_unique<std::ofstream>(filename, std::ios::app);
  if (!file_stream_->is_open()) {
    std::cerr << "EventLogObserver: Failed to open file: " << filename << std::endl;
    file_stream_.reset();
    set_stdout_output();
  } else {
    sink_ = [this](const std::string& msg) {
      if (file_stream_ && file_stream_->is_open()) {
        *file_stream_ << msg << std::endl;
        file_stream_->flush();
      }
    };
  }
}

void EventLogObserver::set_stdout_output() {
  file_stream_.reset();
  string_stream_.reset();
  sink_ = [](const std::string& msg) { std::cout << msg << std::endl; };
}

void EventLogObserver::set_string_output(std::shared_ptr<std::stringstream> stream) {
  file_stream_.reset();
  string_stream_ = stream;
  if (string_stream_) {
    sink_ = [this](const std::string& msg) {
      if (string_stream_) {
        *string_stream_ << msg << std::endl;
      }
    };
  }
}

size_t EventLogObserver::event_count(MeshEvent event) const {
  auto it = event_counts_.find(event);
  return (it != event_counts_.end()) ? it->second : 0;
}

void EventLogObserver::reset_counters() {
  total_events_ = 0;
  throttled_events_ = 0;
  event_counts_.clear();
  last_log_time_.clear();
  start_time_ = std::chrono::steady_clock::now();
  last_event_time_ = start_time_;
}

std::string EventLogObserver::summary_report() const {
  std::stringstream report;
  auto now = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();

  report << "=== EventLogObserver Summary ===" << std::endl;
  report << "Prefix: " << config_.prefix << std::endl;
  report << "Duration: " << duration << " seconds" << std::endl;
  report << "Total events: " << total_events_ << std::endl;
  report << "Throttled events: " << throttled_events_ << std::endl;
  report << std::endl << "Event breakdown:" << std::endl;

  for (const auto& [event, count] : event_counts_) {
    double percentage = (total_events_ > 0) ? (100.0 * count / total_events_) : 0.0;
    report << "  " << std::left << std::setw(20) << event_name(event)
           << std::right << std::setw(8) << count
           << " (" << std::fixed << std::setprecision(1) << percentage << "%)"
           << std::endl;
  }

  if (total_events_ > 0 && duration > 0) {
    report << std::endl << "Event rate: "
           << std::fixed << std::setprecision(2)
           << (static_cast<double>(total_events_) / duration)
           << " events/second" << std::endl;
  }

  return report.str();
}

bool EventLogObserver::should_log(MeshEvent event) const {
  if (config_.filter_events.empty()) {
    return true;  // No filter, log everything
  }

  bool in_filter = config_.filter_events.count(event) > 0;
  return config_.filter_mode_include ? in_filter : !in_filter;
}

bool EventLogObserver::is_throttled(MeshEvent event) {
  if (!config_.throttle_enabled) {
    return false;
  }

  auto it = last_log_time_.find(event);
  if (it == last_log_time_.end()) {
    // First time seeing this event
    return false;
  }

  auto now = std::chrono::steady_clock::now();
  auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second);

  return time_since_last < config_.throttle_interval;
}

std::string EventLogObserver::format_message(MeshEvent event) const {
  std::stringstream msg;

  // Timestamp
  if (config_.include_timestamp) {
    msg << "[" << get_timestamp() << "] ";
  }

  // Prefix
  msg << "[" << config_.prefix << "] ";

  // Event name
  msg << "Event: " << event_name(event);

  // Additional details based on log level
  switch (config_.level) {
    case LogLevel::MINIMAL:
      break;

    case LogLevel::NORMAL:
      msg << " (#" << event_counts_.at(event) << ")";
      break;

    case LogLevel::DETAILED: {
      msg << " (#" << event_counts_.at(event) << ")";
      auto now = std::chrono::steady_clock::now();
      auto ms_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
          now - last_event_time_).count();
      msg << " [+" << ms_since_last << "ms]";
      break;
    }

    case LogLevel::VERBOSE: {
      msg << " (#" << event_counts_.at(event)
          << "/" << total_events_ << ")";

      auto now = std::chrono::steady_clock::now();
      auto ms_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
          now - last_event_time_).count();
      auto ms_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(
          now - start_time_).count();

      msg << " [+" << ms_since_last << "ms"
          << ", total: " << ms_since_start << "ms]";

      if (throttled_events_ > 0) {
        msg << " (throttled: " << throttled_events_ << ")";
      }
      break;
    }
  }

  return msg.str();
}

std::string EventLogObserver::get_timestamp() const {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()).count() % 1000;

  std::stringstream timestamp;
  timestamp << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
  timestamp << "." << std::setfill('0') << std::setw(3) << ms;

  return timestamp.str();
}

// ====================
// EventLogBuilder Implementation
// ====================

std::shared_ptr<EventLogObserver> EventLogBuilder::build() const {
  if (use_file_ && !filename_.empty()) {
    auto observer = std::make_shared<EventLogObserver>(config_);
    observer->set_file_output(filename_);
    return observer;
  } else {
    return std::make_shared<EventLogObserver>(config_);
  }
}

} // namespace svmp