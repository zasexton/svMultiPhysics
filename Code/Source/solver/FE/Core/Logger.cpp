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

/**
 * @file Logger.cpp
 * @brief Implementation of logging utilities
 *
 * This file contains implementation details for the Logger class that
 * are not inlined in the header. Most Logger functionality is in the
 * header for performance reasons, but some utility functions and
 * initialization code is here.
 */

#include "Logger.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace svmp {
namespace FE {

// ============================================================================
// Logger Initialization
// ============================================================================

namespace {

/**
 * @brief Initialize logger from environment variables
 */
class LoggerInitializer {
public:
    LoggerInitializer() {
        auto& logger = Logger::instance();

        // Check FE_LOG_LEVEL environment variable
        if (const char* env_level = std::getenv("FE_LOG_LEVEL")) {
            std::string level_str(env_level);
            std::transform(level_str.begin(), level_str.end(), level_str.begin(), ::toupper);

            if (level_str == "DEBUG") {
                logger.set_level(LogLevel::DEBUG);
            } else if (level_str == "INFO") {
                logger.set_level(LogLevel::INFO);
            } else if (level_str == "WARNING" || level_str == "WARN") {
                logger.set_level(LogLevel::WARNING);
            } else if (level_str == "ERROR") {
                logger.set_level(LogLevel::ERROR);
            } else if (level_str == "CRITICAL" || level_str == "CRIT") {
                logger.set_level(LogLevel::CRITICAL);
            } else if (level_str == "OFF") {
                logger.set_level(LogLevel::OFF);
            }
        }

        // Check FE_LOG_FILE environment variable
        if (const char* env_file = std::getenv("FE_LOG_FILE")) {
            logger.set_file_output(env_file);
        }

        // Check FE_LOG_CONSOLE environment variable
        if (const char* env_console = std::getenv("FE_LOG_CONSOLE")) {
            std::string console_str(env_console);
            std::transform(console_str.begin(), console_str.end(), console_str.begin(), ::tolower);
            logger.set_console_output(console_str != "false" && console_str != "0");
        }

        // Check FE_LOG_SHOW_RANK environment variable
        if (const char* env_rank = std::getenv("FE_LOG_SHOW_RANK")) {
            std::string rank_str(env_rank);
            std::transform(rank_str.begin(), rank_str.end(), rank_str.begin(), ::tolower);
            logger.set_show_rank(rank_str != "false" && rank_str != "0");
        }

        // Check FE_LOG_SHOW_TIME environment variable
        if (const char* env_time = std::getenv("FE_LOG_SHOW_TIME")) {
            std::string time_str(env_time);
            std::transform(time_str.begin(), time_str.end(), time_str.begin(), ::tolower);
            logger.set_show_timestamp(time_str != "false" && time_str != "0");
        }
    }
};

// Static initialization
static LoggerInitializer logger_init;

} // anonymous namespace

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Format memory size for human-readable output
 */
std::string format_memory_size(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    const size_t n_units = sizeof(units) / sizeof(units[0]);

    size_t unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < n_units - 1) {
        size /= 1024.0;
        ++unit_idx;
    }

    std::ostringstream oss;
    if (unit_idx == 0) {
        oss << bytes << " " << units[0];
    } else {
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    }
    return oss.str();
}

/**
 * @brief Format floating-point operations count
 */
std::string format_flops(double flops) {
    const char* units[] = {"FLOPS", "KFLOPS", "MFLOPS", "GFLOPS", "TFLOPS"};
    const size_t n_units = sizeof(units) / sizeof(units[0]);

    size_t unit_idx = 0;
    double value = flops;

    while (value >= 1000.0 && unit_idx < n_units - 1) {
        value /= 1000.0;
        ++unit_idx;
    }

    std::ostringstream oss;
    if (unit_idx == 0) {
        oss << static_cast<long long>(flops) << " " << units[0];
    } else {
        oss << std::fixed << std::setprecision(2) << value << " " << units[unit_idx];
    }
    return oss.str();
}

/**
 * @brief Get current thread name (platform-specific)
 */
std::string get_thread_name() {
    #ifdef __linux__
    char name[16] = {0};
    pthread_getname_np(pthread_self(), name, sizeof(name));
    return std::string(name);
    #else
    // Fallback: use thread ID
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
    #endif
}

// ============================================================================
// Performance Logging Utilities
// ============================================================================

/**
 * @brief Log memory usage statistics
 */
void log_memory_stats(const std::string& context) {
    #ifdef __linux__
    // Read from /proc/self/status for accurate memory info
    std::ifstream status("/proc/self/status");
    if (status.is_open()) {
        std::string line;
        size_t vm_size = 0, vm_rss = 0, vm_peak = 0;

        while (std::getline(status, line)) {
            if (line.find("VmSize:") == 0) {
                std::istringstream iss(line.substr(7));
                iss >> vm_size;
                vm_size *= 1024; // Convert KB to bytes
            } else if (line.find("VmRSS:") == 0) {
                std::istringstream iss(line.substr(6));
                iss >> vm_rss;
                vm_rss *= 1024;
            } else if (line.find("VmPeak:") == 0) {
                std::istringstream iss(line.substr(7));
                iss >> vm_peak;
                vm_peak *= 1024;
            }
        }

        std::ostringstream msg;
        msg << "Memory stats (" << context << "): "
            << "Virtual=" << format_memory_size(vm_size) << ", "
            << "Resident=" << format_memory_size(vm_rss) << ", "
            << "Peak=" << format_memory_size(vm_peak);
        FE_LOG_DEBUG(msg.str());
    }
    #endif
}

/**
 * @brief Log parallel statistics in MPI mode
 */
void log_mpi_stats(const std::string& context, double local_value) {
    #if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) return;

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Gather statistics
    double min_val, max_val, sum_val;
    MPI_Reduce(&local_value, &min_val, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_value, &max_val, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_value, &sum_val, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_val = sum_val / size;
        std::ostringstream msg;
        msg << "MPI stats (" << context << "): "
            << "Min=" << min_val << ", "
            << "Max=" << max_val << ", "
            << "Avg=" << avg_val << " "
            << "(imbalance=" << std::fixed << std::setprecision(1)
            << ((max_val - avg_val) / avg_val * 100) << "%)";
        FE_LOG_INFO(msg.str());
    }
    #else
    // Suppress unused parameter warnings when MPI is disabled
    (void)context;
    (void)local_value;
    #endif
}

// ============================================================================
// Thread-Local Buffer for High-Performance Logging
// ============================================================================

namespace {

/**
 * @brief Thread-local log buffer to reduce contention
 */
thread_local struct ThreadLogBuffer {
    static constexpr size_t BUFFER_SIZE = 4096;
    char buffer[BUFFER_SIZE];
    size_t position = 0;

    void reset() { position = 0; }

    bool append(const char* data, size_t len) {
        if (position + len >= BUFFER_SIZE) {
            return false;
        }
        std::memcpy(buffer + position, data, len);
        position += len;
        return true;
    }

    void flush() {
        if (position > 0) {
            // Write buffer content
            std::cout.write(buffer, static_cast<std::streamsize>(position));
            reset();
        }
    }
} thread_log_buffer;

} // anonymous namespace

/**
 * @brief Buffered logging for high-frequency output
 */
void log_buffered(LogLevel level, const std::string& message) {
    (void)level;  // Suppress unused parameter warning
    if (!thread_log_buffer.append(message.c_str(), message.size())) {
        thread_log_buffer.flush();
        thread_log_buffer.append(message.c_str(), message.size());
    }
}

/**
 * @brief Flush thread-local log buffers
 */
void flush_thread_buffers() {
    thread_log_buffer.flush();
}

// ============================================================================
// Specialized Loggers
// ============================================================================

/**
 * @brief Logger for convergence history
 */
class ConvergenceLogger {
public:
    void log_iteration(int iter, double residual, double tolerance) {
        std::ostringstream msg;
        msg << "Iteration " << std::setw(4) << iter << ": "
            << "residual = " << std::scientific << std::setprecision(6) << residual
            << " (tol = " << tolerance << ")";

        if (residual < tolerance) {
            msg << " [CONVERGED]";
            FE_LOG_INFO(msg.str());
        } else {
            FE_LOG_DEBUG(msg.str());
        }

        history_.push_back({iter, residual});
    }

    void log_summary() {
        if (history_.empty()) return;

        std::ostringstream msg;
        msg << "Convergence summary: " << history_.size() << " iterations, "
            << "initial residual = " << std::scientific << history_.front().residual << ", "
            << "final residual = " << history_.back().residual;
        FE_LOG_INFO(msg.str());
    }

private:
    struct Entry {
        int iteration;
        double residual;
    };
    std::vector<Entry> history_;
};

/**
 * @brief Global convergence logger instance
 */
ConvergenceLogger& convergence_logger() {
    static ConvergenceLogger logger;
    return logger;
}

// ============================================================================
// Public API Functions
// ============================================================================

void log_convergence(int iter, double residual, double tolerance) {
    convergence_logger().log_iteration(iter, residual, tolerance);
}

void log_convergence_summary() {
    convergence_logger().log_summary();
}

} // namespace FE
} // namespace svmp