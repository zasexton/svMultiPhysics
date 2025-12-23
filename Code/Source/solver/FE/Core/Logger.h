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

#ifndef SVMP_FE_LOGGER_H
#define SVMP_FE_LOGGER_H

/**
 * @file Logger.h
 * @brief Logging infrastructure for the FE library
 *
 * Provides thread-safe, MPI-aware logging with multiple severity levels,
 * performance timing integration, and flexible output options.
 */

#include "Types.h"
#include "FEConfig.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mutex>
#include <memory>
#include <chrono>
#include <iomanip>
#include <vector>
#include <functional>
#include <thread>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {

// ============================================================================
// Log Levels
// ============================================================================

/**
 * @brief Logging severity levels
 */
enum class LogLevel : int {
    DEBUG    = 0,  // Detailed debug information
    INFO     = 1,  // Informational messages
    WARNING  = 2,  // Warning messages
    ERROR    = 3,  // Error messages
    CRITICAL = 4,  // Critical errors
    OFF      = 5   // Logging disabled
};

/**
 * @brief Convert log level to string
 */
inline const char* log_level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:    return "DEBUG";
        case LogLevel::INFO:     return "INFO";
        case LogLevel::WARNING:  return "WARN";
        case LogLevel::ERROR:    return "ERROR";
        case LogLevel::CRITICAL: return "CRIT";
        default:                 return "UNKNOWN";
    }
}

// ============================================================================
// Timer Class for Performance Logging
// ============================================================================

/**
 * @brief Simple timer for performance measurements
 */
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    /**
     * @brief Start or restart the timer
     */
    void start() {
        start_time_ = Clock::now();
        is_running_ = true;
    }

    /**
     * @brief Stop the timer
     */
    void stop() {
        if (is_running_) {
            end_time_ = Clock::now();
            is_running_ = false;
        }
    }

    /**
     * @brief Get elapsed time in seconds
     */
    double elapsed() const {
        TimePoint end = is_running_ ? Clock::now() : end_time_;
        Duration diff = end - start_time_;
        return diff.count();
    }

    /**
     * @brief Reset the timer
     */
    void reset() {
        start_time_ = Clock::now();
        end_time_ = start_time_;
        is_running_ = false;
    }

private:
    TimePoint start_time_;
    TimePoint end_time_;
    bool is_running_ = false;
};

// ============================================================================
// Log Message Structure
// ============================================================================

/**
 * @brief Structure representing a log message
 */
struct LogMessage {
    LogLevel level;
    std::string message;
    std::string file;
    int line;
    std::string function;
    std::chrono::system_clock::time_point timestamp;
    int mpi_rank;
    std::thread::id thread_id;
};

// ============================================================================
// Logger Class
// ============================================================================

/**
 * @brief Thread-safe, MPI-aware logger for the FE library
 *
 * Features:
 * - Multiple output destinations (console, file)
 * - Thread-safe logging with minimal contention
 * - MPI rank prefixing in parallel runs
 * - Performance timing integration
 * - Compile-time log level filtering
 * - Formatted output with timestamps
 */
class Logger {
public:
    /**
     * @brief Get singleton logger instance
     */
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    // Delete copy and move operations
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    /**
     * @brief Set minimum log level
     */
    void set_level(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        min_level_ = level;
    }

    /**
     * @brief Get current log level
     */
    LogLevel get_level() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return min_level_;
    }

    /**
     * @brief Enable/disable console output
     */
    void set_console_output(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        console_output_ = enabled;
    }

    /**
     * @brief Set file output
     *
     * @param filename Output file name (will append rank in MPI mode)
     */
    void set_file_output(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Close existing file if open
        if (file_stream_.is_open()) {
            file_stream_.close();
        }

        if (!filename.empty()) {
            std::string actual_filename = filename;

            // Add MPI rank to filename
            #if FE_HAS_MPI
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (initialized) {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                size_t dot_pos = filename.rfind('.');
                if (dot_pos != std::string::npos) {
                    actual_filename = filename.substr(0, dot_pos) +
                                     "_rank" + std::to_string(rank) +
                                     filename.substr(dot_pos);
                } else {
                    actual_filename += "_rank" + std::to_string(rank);
                }
            }
            #endif

            file_stream_.open(actual_filename, std::ios::app);
            if (!file_stream_.is_open()) {
                std::cerr << "Failed to open log file: " << actual_filename << std::endl;
            }
        }
    }

    /**
     * @brief Set whether to show MPI rank in output
     */
    void set_show_rank(bool show) {
        std::lock_guard<std::mutex> lock(mutex_);
        show_rank_ = show;
    }

    /**
     * @brief Set whether to show timestamp in output
     */
    void set_show_timestamp(bool show) {
        std::lock_guard<std::mutex> lock(mutex_);
        show_timestamp_ = show;
    }

    /**
     * @brief Log a message
     */
    void log(LogLevel level,
             const std::string& message,
             const char* file = "",
             int line = 0,
             const char* function = "") {

        // Early return for compile-time filtering
        #if !FE_DEBUG_MODE
        if (level == LogLevel::DEBUG) return;
        #endif

        // Runtime level check
        if (level < min_level_) return;

        // Create log message
        LogMessage msg;
        msg.level = level;
        msg.message = message;
        msg.file = file;
        msg.line = line;
        msg.function = function;
        msg.timestamp = std::chrono::system_clock::now();
        msg.thread_id = std::this_thread::get_id();
        msg.mpi_rank = -1;

        #if FE_HAS_MPI
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Comm_rank(MPI_COMM_WORLD, &msg.mpi_rank);
        }
        #endif

        // Format and output
        std::string formatted = format_message(msg);

        std::lock_guard<std::mutex> lock(mutex_);

        if (console_output_) {
            // Use cerr for errors and warnings
            if (level >= LogLevel::WARNING) {
                std::cerr << formatted << std::flush;
            } else {
                std::cout << formatted << std::flush;
            }
        }

        if (file_stream_.is_open()) {
            file_stream_ << formatted << std::flush;
        }

        // Call custom handlers
        for (const auto& handler : handlers_) {
            handler(msg);
        }
    }

    /**
     * @brief Log with timing information
     */
    void log_timed(LogLevel level,
                   const std::string& message,
                   double elapsed_seconds) {
        std::ostringstream oss;
        oss << message << " (elapsed: " << std::fixed << std::setprecision(3)
            << elapsed_seconds << "s)";
        log(level, oss.str());
    }

    /**
     * @brief Add custom log handler
     */
    using LogHandler = std::function<void(const LogMessage&)>;
    void add_handler(LogHandler handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        handlers_.push_back(std::move(handler));
    }

    /**
     * @brief Flush all outputs
     */
    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout.flush();
        std::cerr.flush();
        if (file_stream_.is_open()) {
            file_stream_.flush();
        }
    }

private:
    Logger() : min_level_(LogLevel::INFO),
               console_output_(true),
               show_rank_(true),
               show_timestamp_(true) {}

    ~Logger() {
        if (file_stream_.is_open()) {
            file_stream_.close();
        }
    }

    /**
     * @brief Format log message
     */
    std::string format_message(const LogMessage& msg) {
        std::ostringstream oss;

        // Timestamp
        if (show_timestamp_) {
            auto time_t = std::chrono::system_clock::to_time_t(msg.timestamp);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                msg.timestamp.time_since_epoch()) % 1000;

            oss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S")
                << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
        }

        // MPI rank
        if (show_rank_ && msg.mpi_rank >= 0) {
            oss << "[R" << msg.mpi_rank << "] ";
        }

        // Log level
        oss << "[" << log_level_to_string(msg.level) << "] ";

        // Message
        oss << msg.message;

        // Source location in debug mode
        #if FE_DEBUG_MODE
        if (msg.level >= LogLevel::WARNING && !msg.file.empty()) {
            oss << " (" << msg.file << ":" << msg.line;
            if (!msg.function.empty()) {
                oss << " in " << msg.function << "()";
            }
            oss << ")";
        }
        #endif

        oss << "\n";
        return oss.str();
    }

    mutable std::mutex mutex_;
    LogLevel min_level_;
    bool console_output_;
    bool show_rank_;
    bool show_timestamp_;
    std::ofstream file_stream_;
    std::vector<LogHandler> handlers_;
};

// ============================================================================
// Scoped Timer for RAII-based Timing
// ============================================================================

/**
 * @brief RAII timer that logs elapsed time on destruction
 */
class ScopedTimer {
public:
    ScopedTimer(const std::string& name,
                LogLevel level = LogLevel::INFO)
        : name_(name), level_(level) {
        timer_.start();
        Logger::instance().log(level_, "Starting: " + name_);
    }

    ~ScopedTimer() {
        timer_.stop();
        Logger::instance().log_timed(level_, "Completed: " + name_, timer_.elapsed());
    }

private:
    std::string name_;
    LogLevel level_;
    Timer timer_;
};

// ============================================================================
// Logging Macros
// ============================================================================

/**
 * @brief Main logging macro
 */
#define FE_LOG(level, message) \
    svmp::FE::Logger::instance().log(level, message, __FILE__, __LINE__, __FUNCTION__)

/**
 * @brief Debug logging (compiled out in release mode)
 */
#if FE_DEBUG_MODE
    #define FE_LOG_DEBUG(message) FE_LOG(svmp::FE::LogLevel::DEBUG, message)
#else
    #define FE_LOG_DEBUG(message) ((void)0)
#endif

/**
 * @brief Info logging
 */
#define FE_LOG_INFO(message) FE_LOG(svmp::FE::LogLevel::INFO, message)

/**
 * @brief Warning logging
 */
#define FE_LOG_WARNING(message) FE_LOG(svmp::FE::LogLevel::WARNING, message)

/**
 * @brief Error logging
 */
#define FE_LOG_ERROR(message) FE_LOG(svmp::FE::LogLevel::ERROR, message)

/**
 * @brief Critical logging
 */
#define FE_LOG_CRITICAL(message) FE_LOG(svmp::FE::LogLevel::CRITICAL, message)

/**
 * @brief Timed scope macro
 */
#define FE_TIMED_SCOPE(name) \
    svmp::FE::ScopedTimer _scoped_timer_##__LINE__(name)

/**
 * @brief Timed scope with custom log level
 */
#define FE_TIMED_SCOPE_LEVEL(name, level) \
    svmp::FE::ScopedTimer _scoped_timer_##__LINE__(name, level)

// ============================================================================
// Stream-based Logging Interface
// ============================================================================

/**
 * @brief Stream wrapper for convenient logging
 */
class LogStream {
public:
    LogStream(LogLevel level) : level_(level) {}

    ~LogStream() {
        Logger::instance().log(level_, stream_.str());
    }

    template<typename T>
    LogStream& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

private:
    LogLevel level_;
    std::ostringstream stream_;
};

/**
 * @brief Create log stream
 */
inline LogStream log_stream(LogLevel level) {
    return LogStream(level);
}

// Stream-based macros
#define FE_DEBUG()   svmp::FE::log_stream(svmp::FE::LogLevel::DEBUG)
#define FE_INFO()    svmp::FE::log_stream(svmp::FE::LogLevel::INFO)
#define FE_WARNING() svmp::FE::log_stream(svmp::FE::LogLevel::WARNING)
#define FE_ERROR()   svmp::FE::log_stream(svmp::FE::LogLevel::ERROR)

} // namespace FE
} // namespace svmp

#endif // SVMP_FE_LOGGER_H