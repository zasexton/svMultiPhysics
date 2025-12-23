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
    : EventLogObserver(config, nullptr) {
}

EventLogObserver::EventLogObserver(const Config& config, LogSink sink)
    : config_(config),
      sink_(sink),
      start_time_(std::chrono::steady_clock::now()),
      last_event_time_(start_time_) {
  if (!sink_) {
    sink_ = [this](const std::string& msg) {
      std::lock_guard<std::mutex> lock(output_mutex_);
      std::cout << msg << std::endl;
    };
  }
}

void EventLogObserver::on_mesh_event(MeshEvent event) {
  LogSink sink;
  std::string message;
  {
    std::lock_guard<std::mutex> lock(mutex_);

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

    message = format_message(event);

    // Update timing
    last_event_time_ = std::chrono::steady_clock::now();
    if (config_.throttle_enabled) {
      last_log_time_[event] = last_event_time_;
    }

    sink = sink_;
  }

  if (sink) {
    sink(message);
  }
}

void EventLogObserver::set_throttling(bool enabled, std::chrono::milliseconds interval) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.throttle_enabled = enabled;
  config_.throttle_interval = interval;
  if (!enabled) {
    last_log_time_.clear();
  }
}

void EventLogObserver::set_filter(const std::set<MeshEvent>& events, bool include_mode) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.filter_events = events;
  config_.filter_mode_include = include_mode;
}

void EventLogObserver::clear_filter() {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.filter_events.clear();
}

void EventLogObserver::set_file_output(const std::string& filename) {
  auto stream = std::make_unique<std::ofstream>(filename, std::ios::app);
  if (!stream->is_open()) {
    std::cerr << "EventLogObserver: Failed to open file: " << filename << std::endl;
    set_stdout_output();
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  {
    std::lock_guard<std::mutex> out_lock(output_mutex_);
    file_stream_ = std::move(stream);
    string_stream_.reset();
  }

  sink_ = [this](const std::string& msg) {
    std::lock_guard<std::mutex> out_lock(output_mutex_);
    if (file_stream_ && file_stream_->is_open()) {
      *file_stream_ << msg << std::endl;
      file_stream_->flush();
    }
  };
}

void EventLogObserver::set_stdout_output() {
  std::lock_guard<std::mutex> lock(mutex_);
  {
    std::lock_guard<std::mutex> out_lock(output_mutex_);
    file_stream_.reset();
    string_stream_.reset();
  }
  sink_ = [this](const std::string& msg) {
    std::lock_guard<std::mutex> out_lock(output_mutex_);
    std::cout << msg << std::endl;
  };
}

void EventLogObserver::set_string_output(std::shared_ptr<std::stringstream> stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  {
    std::lock_guard<std::mutex> out_lock(output_mutex_);
    file_stream_.reset();
    string_stream_ = std::move(stream);
  }
  sink_ = [this](const std::string& msg) {
    std::lock_guard<std::mutex> out_lock(output_mutex_);
    if (string_stream_) {
      *string_stream_ << msg << std::endl;
    }
  };
}

size_t EventLogObserver::event_count(MeshEvent event) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = event_counts_.find(event);
  return (it != event_counts_.end()) ? it->second : 0;
}

void EventLogObserver::reset_counters() {
  std::lock_guard<std::mutex> lock(mutex_);
  total_events_ = 0;
  throttled_events_ = 0;
  event_counts_.clear();
  last_log_time_.clear();
  start_time_ = std::chrono::steady_clock::now();
  last_event_time_ = start_time_;
}

std::string EventLogObserver::summary_report() const {
  auto now = std::chrono::steady_clock::now();
  Config cfg;
  size_t total_events = 0;
  size_t throttled_events = 0;
  std::map<MeshEvent, size_t> counts;
  std::chrono::steady_clock::time_point start_time;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    cfg = config_;
    total_events = total_events_;
    throttled_events = throttled_events_;
    counts = event_counts_;
    start_time = start_time_;
  }

  std::stringstream report;
  const double duration_s = std::chrono::duration<double>(now - start_time).count();

  report << "=== EventLogObserver Summary ===" << std::endl;
  report << "Prefix: " << cfg.prefix << std::endl;
  report << "Duration: " << std::fixed << std::setprecision(3) << duration_s << " seconds" << std::endl;
  report << "Total events: " << total_events << std::endl;
  report << "Throttled events: " << throttled_events << std::endl;
  report << std::endl << "Event breakdown:" << std::endl;

  for (const auto& [event, count] : counts) {
    double percentage = (total_events > 0) ? (100.0 * count / total_events) : 0.0;
    report << "  " << std::left << std::setw(20) << event_name(event)
           << std::right << std::setw(8) << count
           << " (" << std::fixed << std::setprecision(1) << percentage << "%)"
           << std::endl;
  }

  if (total_events > 0 && duration_s > 0.0) {
    report << std::endl << "Event rate: "
           << std::fixed << std::setprecision(2)
           << (static_cast<double>(total_events) / duration_s)
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

  std::tm tm_buf{};
#if defined(_WIN32)
  localtime_s(&tm_buf, &time_t);
#else
  localtime_r(&time_t, &tm_buf);
#endif

  std::stringstream timestamp;
  timestamp << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
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
