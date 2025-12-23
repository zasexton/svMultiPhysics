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

#ifndef SVMP_FE_ASSEMBLY_ASSEMBLY_STATISTICS_H
#define SVMP_FE_ASSEMBLY_ASSEMBLY_STATISTICS_H

/**
 * @file AssemblyStatistics.h
 * @brief Performance monitoring and profiling for finite element assembly
 *
 * AssemblyStatistics provides comprehensive instrumentation for FE assembly:
 *
 * 1. TIMING BREAKDOWN:
 *    - Total assembly time
 *    - Kernel computation time (local matrices/vectors)
 *    - Scatter time (global insertion)
 *    - Communication time (MPI operations)
 *    - Constraint application time
 *
 * 2. FLOP COUNTING:
 *    - Estimated FLOPs for assembly operations
 *    - Achieved FLOP/s rate
 *    - Computational intensity (FLOPs/byte)
 *
 * 3. CACHE ANALYSIS:
 *    - Hardware counter integration hooks (PAPI, perf)
 *    - Cache miss estimation
 *    - Memory bandwidth utilization
 *
 * 4. LOAD BALANCE METRICS:
 *    - Per-thread timing
 *    - Element distribution statistics
 *    - Imbalance factor
 *
 * 5. OPTIMIZATION SUGGESTIONS:
 *    - Recommend assembly strategy based on workload
 *    - Identify bottlenecks
 *    - Suggest tuning parameters
 *
 * 6. MEMORY TRACKING:
 *    - Peak memory usage
 *    - Allocation patterns
 *    - Scratch buffer statistics
 *
 * Integration:
 *    Statistics can be collected via RAII timers or explicit start/stop calls.
 *    All timing uses high-resolution clocks for accuracy.
 *
 * @see Assembler for the main assembly interface
 * @see AssemblyLoop for loop-level timing hooks
 */

#include "Core/Types.h"

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <atomic>
#include <functional>
#include <optional>
#include <map>
#include <iosfwd>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// Timing Categories
// ============================================================================

/**
 * @brief Categories for timing breakdown
 */
enum class TimingCategory : std::uint8_t {
    Total,               ///< Total assembly time
    KernelCompute,       ///< Element kernel computation
    GlobalScatter,       ///< Insertion into global system
    Communication,       ///< MPI communication
    ConstraintApply,     ///< Constraint application
    SparsityBuild,       ///< Sparsity pattern construction
    Scheduling,          ///< Element scheduling/ordering
    Coloring,            ///< Graph coloring
    Initialization,      ///< Setup and initialization
    Finalization,        ///< Cleanup and finalization
    ContextSetup,        ///< Per-element context preparation
    DataPack,            ///< Ghost data packing
    DataUnpack,          ///< Ghost data unpacking
    Synchronization,     ///< Thread synchronization
    Other                ///< Miscellaneous
};

/**
 * @brief Convert timing category to string
 */
const char* timingCategoryName(TimingCategory category) noexcept;

// ============================================================================
// Timer Utilities
// ============================================================================

/**
 * @brief High-resolution timing data
 */
struct TimingData {
    double total_seconds{0.0};        ///< Total accumulated time
    double min_seconds{0.0};          ///< Minimum single measurement
    double max_seconds{0.0};          ///< Maximum single measurement
    std::uint64_t call_count{0};      ///< Number of measurements

    /**
     * @brief Get average time per call
     */
    [[nodiscard]] double average() const noexcept {
        return call_count > 0 ? total_seconds / static_cast<double>(call_count) : 0.0;
    }

    /**
     * @brief Accumulate another timing measurement
     */
    void accumulate(double seconds) {
        total_seconds += seconds;
        if (call_count == 0) {
            min_seconds = seconds;
            max_seconds = seconds;
        } else {
            min_seconds = std::min(min_seconds, seconds);
            max_seconds = std::max(max_seconds, seconds);
        }
        ++call_count;
    }

    /**
     * @brief Reset timing data
     */
    void reset() {
        total_seconds = 0.0;
        min_seconds = 0.0;
        max_seconds = 0.0;
        call_count = 0;
    }

    /**
     * @brief Merge timing data from another source
     */
    void merge(const TimingData& other) {
        total_seconds += other.total_seconds;
        call_count += other.call_count;
        if (other.call_count > 0) {
            if (call_count == other.call_count) {
                min_seconds = other.min_seconds;
                max_seconds = other.max_seconds;
            } else {
                min_seconds = std::min(min_seconds, other.min_seconds);
                max_seconds = std::max(max_seconds, other.max_seconds);
            }
        }
    }
};

/**
 * @brief RAII timer for automatic timing
 */
class ScopedTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    /**
     * @brief Construct and start timer
     */
    explicit ScopedTimer(TimingData& target);

    /**
     * @brief Construct with callback
     */
    explicit ScopedTimer(std::function<void(double)> callback);

    /**
     * @brief Destructor - records elapsed time
     */
    ~ScopedTimer();

    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

    /**
     * @brief Get elapsed time without stopping
     */
    [[nodiscard]] double elapsed() const;

    /**
     * @brief Stop timer early (skips destructor recording)
     */
    double stop();

private:
    TimePoint start_time_;
    TimingData* target_{nullptr};
    std::function<void(double)> callback_;
    bool stopped_{false};
};

// ============================================================================
// FLOP Counting
// ============================================================================

/**
 * @brief FLOP counting data
 */
struct FLOPData {
    std::uint64_t total_flops{0};      ///< Total floating-point operations
    std::uint64_t add_ops{0};          ///< Addition operations
    std::uint64_t mul_ops{0};          ///< Multiplication operations
    std::uint64_t div_ops{0};          ///< Division operations
    std::uint64_t fma_ops{0};          ///< Fused multiply-add operations

    /**
     * @brief Compute total (counting FMA as 2 ops)
     */
    [[nodiscard]] std::uint64_t computeTotal() const noexcept {
        return add_ops + mul_ops + div_ops + 2 * fma_ops;
    }

    /**
     * @brief Compute FLOP rate given elapsed time
     */
    [[nodiscard]] double flopRate(double seconds) const noexcept {
        if (seconds <= 0.0) {
            return 0.0;
        }
        std::uint64_t total = total_flops != 0 ? total_flops : computeTotal();
        return static_cast<double>(total) / seconds;
    }

    /**
     * @brief Reset counters
     */
    void reset() {
        total_flops = 0;
        add_ops = 0;
        mul_ops = 0;
        div_ops = 0;
        fma_ops = 0;
    }

    /**
     * @brief Merge FLOP data
     */
    void merge(const FLOPData& other) {
        total_flops += other.total_flops;
        add_ops += other.add_ops;
        mul_ops += other.mul_ops;
        div_ops += other.div_ops;
        fma_ops += other.fma_ops;
    }
};

/**
 * @brief FLOP estimator for assembly operations
 */
class FLOPEstimator {
public:
    /**
     * @brief Estimate FLOPs for stiffness matrix assembly
     *
     * @param num_dofs Number of DOFs per element
     * @param num_qpts Number of quadrature points
     * @param dim Spatial dimension
     * @return Estimated FLOPs
     */
    [[nodiscard]] static std::uint64_t stiffnessMatrix(
        LocalIndex num_dofs, LocalIndex num_qpts, int dim);

    /**
     * @brief Estimate FLOPs for mass matrix assembly
     */
    [[nodiscard]] static std::uint64_t massMatrix(
        LocalIndex num_dofs, LocalIndex num_qpts);

    /**
     * @brief Estimate FLOPs for load vector assembly
     */
    [[nodiscard]] static std::uint64_t loadVector(
        LocalIndex num_dofs, LocalIndex num_qpts);

    /**
     * @brief Estimate FLOPs for Jacobian evaluation
     */
    [[nodiscard]] static std::uint64_t jacobianEval(
        LocalIndex num_nodes, int dim);

    /**
     * @brief Estimate FLOPs for basis function evaluation
     */
    [[nodiscard]] static std::uint64_t basisEval(
        LocalIndex num_dofs, LocalIndex num_qpts, int dim, int derivative_order);
};

// ============================================================================
// Memory Statistics
// ============================================================================

/**
 * @brief Memory usage statistics
 */
struct MemoryStats {
    std::size_t peak_bytes{0};           ///< Peak memory usage
    std::size_t current_bytes{0};        ///< Current memory usage
    std::size_t total_allocated{0};      ///< Total bytes allocated
    std::size_t total_deallocated{0};    ///< Total bytes deallocated
    std::uint64_t allocation_count{0};   ///< Number of allocations
    std::uint64_t deallocation_count{0}; ///< Number of deallocations

    /**
     * @brief Track allocation
     */
    void trackAllocation(std::size_t bytes) {
        current_bytes += bytes;
        total_allocated += bytes;
        ++allocation_count;
        peak_bytes = std::max(peak_bytes, current_bytes);
    }

    /**
     * @brief Track deallocation
     */
    void trackDeallocation(std::size_t bytes) {
        current_bytes = bytes > current_bytes ? 0 : current_bytes - bytes;
        total_deallocated += bytes;
        ++deallocation_count;
    }

    /**
     * @brief Reset statistics
     */
    void reset() {
        peak_bytes = 0;
        current_bytes = 0;
        total_allocated = 0;
        total_deallocated = 0;
        allocation_count = 0;
        deallocation_count = 0;
    }
};

// ============================================================================
// Load Balance Statistics
// ============================================================================

/**
 * @brief Per-thread statistics
 */
struct ThreadStats {
    int thread_id{0};
    GlobalIndex elements_processed{0};
    GlobalIndex faces_processed{0};
    double compute_seconds{0.0};
    double scatter_seconds{0.0};
    double wait_seconds{0.0};
    std::uint64_t flops{0};

    /**
     * @brief Get total time
     */
    [[nodiscard]] double totalTime() const noexcept {
        return compute_seconds + scatter_seconds + wait_seconds;
    }
};

/**
 * @brief Load balance analysis
 */
struct LoadBalanceStats {
    int num_threads{0};
    std::vector<ThreadStats> thread_stats;

    double total_compute_seconds{0.0};
    double total_scatter_seconds{0.0};
    double total_wait_seconds{0.0};

    double imbalance_factor{1.0};  ///< max_thread_time / avg_thread_time
    double efficiency{1.0};        ///< useful_work / total_thread_time

    /**
     * @brief Compute imbalance from thread stats
     */
    void computeImbalance() {
        if (thread_stats.empty()) {
            imbalance_factor = 1.0;
            efficiency = 1.0;
            return;
        }

        double total_time = 0.0;
        double max_time = 0.0;
        double total_work_time = 0.0;

        for (const auto& ts : thread_stats) {
            double thread_time = ts.totalTime();
            total_time += thread_time;
            max_time = std::max(max_time, thread_time);
            total_work_time += ts.compute_seconds + ts.scatter_seconds;
        }

        double avg_time = total_time / static_cast<double>(thread_stats.size());
        imbalance_factor = avg_time > 0.0 ? max_time / avg_time : 1.0;
        efficiency = total_time > 0.0 ? total_work_time / total_time : 1.0;
    }
};

// ============================================================================
// Cache Statistics
// ============================================================================

/**
 * @brief Cache-related statistics
 */
struct CacheStats {
    std::uint64_t l1_hits{0};
    std::uint64_t l1_misses{0};
    std::uint64_t l2_hits{0};
    std::uint64_t l2_misses{0};
    std::uint64_t l3_hits{0};
    std::uint64_t l3_misses{0};
    std::uint64_t tlb_misses{0};

    /**
     * @brief Compute L1 cache hit rate
     */
    [[nodiscard]] double l1HitRate() const noexcept {
        std::uint64_t total = l1_hits + l1_misses;
        return total > 0 ? static_cast<double>(l1_hits) / static_cast<double>(total) : 0.0;
    }

    /**
     * @brief Compute L2 cache hit rate
     */
    [[nodiscard]] double l2HitRate() const noexcept {
        std::uint64_t total = l2_hits + l2_misses;
        return total > 0 ? static_cast<double>(l2_hits) / static_cast<double>(total) : 0.0;
    }

    /**
     * @brief Compute L3 cache hit rate
     */
    [[nodiscard]] double l3HitRate() const noexcept {
        std::uint64_t total = l3_hits + l3_misses;
        return total > 0 ? static_cast<double>(l3_hits) / static_cast<double>(total) : 0.0;
    }

    /**
     * @brief Check if hardware counters are available
     */
    [[nodiscard]] bool hasHardwareCounters() const noexcept {
        return (l1_hits + l1_misses + l2_hits + l2_misses) > 0;
    }

    /**
     * @brief Reset statistics
     */
    void reset() {
        l1_hits = l1_misses = 0;
        l2_hits = l2_misses = 0;
        l3_hits = l3_misses = 0;
        tlb_misses = 0;
    }
};

// ============================================================================
// Optimization Suggestion
// ============================================================================

/**
 * @brief Suggested optimization action
 */
enum class OptimizationAction : std::uint8_t {
    None,                    ///< No suggestion
    UseColoring,             ///< Switch to colored assembly
    UseWorkStream,           ///< Switch to work-stream pattern
    UseMatrixFree,           ///< Switch to matrix-free
    IncreaseThreads,         ///< Add more threads
    DecreaseThreads,         ///< Reduce threads (overhead)
    UseCacheBlocking,        ///< Enable cache-blocked ordering
    UseHilbertOrdering,      ///< Switch to Hilbert ordering
    IncreaseBatchSize,       ///< Larger element batches
    DecreaseBatchSize,       ///< Smaller element batches
    EnablePrefetch,          ///< Enable prefetching
    ReduceCommunication,     ///< Communication is bottleneck
    RebalanceWork,           ///< Work imbalance detected
    Custom                   ///< User-defined suggestion
};

/**
 * @brief Optimization suggestion with rationale
 */
struct OptimizationSuggestion {
    OptimizationAction action{OptimizationAction::None};
    std::string rationale;
    double expected_improvement{0.0};  ///< Estimated speedup factor
    int priority{0};                   ///< Higher = more important

    bool operator<(const OptimizationSuggestion& other) const {
        return priority > other.priority;  // Higher priority first
    }
};

// ============================================================================
// Assembly Statistics
// ============================================================================

/**
 * @brief Comprehensive assembly statistics collector
 *
 * AssemblyStatistics provides a centralized facility for collecting and
 * analyzing performance data during finite element assembly.
 *
 * Usage:
 * @code
 *   AssemblyStatistics stats;
 *   stats.startCollection();
 *
 *   // Assembly loop
 *   {
 *       auto timer = stats.scopedTimer(TimingCategory::KernelCompute);
 *       // ... kernel computation ...
 *   }
 *   {
 *       auto timer = stats.scopedTimer(TimingCategory::GlobalScatter);
 *       // ... global insertion ...
 *   }
 *
 *   stats.endCollection();
 *
 *   // Analyze results
 *   stats.printSummary();
 *   auto suggestions = stats.getOptimizationSuggestions();
 * @endcode
 */
class AssemblyStatistics {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    AssemblyStatistics();

    /**
     * @brief Destructor
     */
    ~AssemblyStatistics();

    /**
     * @brief Move constructor
     */
    AssemblyStatistics(AssemblyStatistics&& other) noexcept;

    /**
     * @brief Move assignment
     */
    AssemblyStatistics& operator=(AssemblyStatistics&& other) noexcept;

    // Non-copyable
    AssemblyStatistics(const AssemblyStatistics&) = delete;
    AssemblyStatistics& operator=(const AssemblyStatistics&) = delete;

    // =========================================================================
    // Collection Control
    // =========================================================================

    /**
     * @brief Start statistics collection
     */
    void startCollection();

    /**
     * @brief End statistics collection
     */
    void endCollection();

    /**
     * @brief Check if collection is active
     */
    [[nodiscard]] bool isCollecting() const noexcept;

    /**
     * @brief Reset all statistics
     */
    void reset();

    /**
     * @brief Enable/disable specific categories
     */
    void setEnabled(TimingCategory category, bool enabled);

    /**
     * @brief Check if category is enabled
     */
    [[nodiscard]] bool isEnabled(TimingCategory category) const noexcept;

    // =========================================================================
    // Timing
    // =========================================================================

    /**
     * @brief Start timing a category
     */
    void startTiming(TimingCategory category);

    /**
     * @brief Stop timing a category
     */
    void stopTiming(TimingCategory category);

    /**
     * @brief Create RAII timer for category
     */
    [[nodiscard]] ScopedTimer scopedTimer(TimingCategory category);

    /**
     * @brief Record timing directly
     */
    void recordTiming(TimingCategory category, double seconds);

    /**
     * @brief Get timing data for category
     */
    [[nodiscard]] const TimingData& getTiming(TimingCategory category) const;

    /**
     * @brief Get all timing data
     */
    [[nodiscard]] const std::map<TimingCategory, TimingData>& getAllTimings() const;

    // =========================================================================
    // FLOP Counting
    // =========================================================================

    /**
     * @brief Record FLOPs
     */
    void recordFLOPs(std::uint64_t flops);

    /**
     * @brief Record FLOP breakdown
     */
    void recordFLOPs(const FLOPData& data);

    /**
     * @brief Get FLOP data
     */
    [[nodiscard]] const FLOPData& getFLOPData() const;

    /**
     * @brief Compute achieved FLOP/s
     */
    [[nodiscard]] double getFLOPRate() const;

    // =========================================================================
    // Memory Tracking
    // =========================================================================

    /**
     * @brief Enable memory tracking
     */
    void enableMemoryTracking(bool enable);

    /**
     * @brief Record allocation
     */
    void trackAllocation(std::size_t bytes);

    /**
     * @brief Record deallocation
     */
    void trackDeallocation(std::size_t bytes);

    /**
     * @brief Get memory statistics
     */
    [[nodiscard]] const MemoryStats& getMemoryStats() const;

    // =========================================================================
    // Load Balance
    // =========================================================================

    /**
     * @brief Set number of threads for load balance tracking
     */
    void setNumThreads(int num_threads);

    /**
     * @brief Record per-thread statistics
     */
    void recordThreadStats(int thread_id, const ThreadStats& stats);

    /**
     * @brief Record element completion for thread
     */
    void recordElementCompletion(int thread_id, double compute_time, double scatter_time);

    /**
     * @brief Get load balance statistics
     */
    [[nodiscard]] const LoadBalanceStats& getLoadBalanceStats() const;

    /**
     * @brief Compute load balance metrics
     */
    void computeLoadBalance();

    // =========================================================================
    // Cache Statistics (with hardware counter integration)
    // =========================================================================

    /**
     * @brief Enable cache statistics collection
     *
     * @return true if hardware counters are available
     */
    bool enableCacheStats();

    /**
     * @brief Disable cache statistics
     */
    void disableCacheStats();

    /**
     * @brief Get cache statistics
     */
    [[nodiscard]] const CacheStats& getCacheStats() const;

    /**
     * @brief Update cache stats from hardware counters
     */
    void updateCacheStats();

    // =========================================================================
    // Element/Assembly Counts
    // =========================================================================

    /**
     * @brief Record element assembly
     */
    void recordElement(GlobalIndex cell_id, ElementType type);

    /**
     * @brief Record matrix entries inserted
     */
    void recordMatrixEntries(GlobalIndex count);

    /**
     * @brief Record vector entries inserted
     */
    void recordVectorEntries(GlobalIndex count);

    /**
     * @brief Get total elements assembled
     */
    [[nodiscard]] GlobalIndex totalElements() const noexcept;

    /**
     * @brief Get total matrix entries
     */
    [[nodiscard]] GlobalIndex totalMatrixEntries() const noexcept;

    /**
     * @brief Get total vector entries
     */
    [[nodiscard]] GlobalIndex totalVectorEntries() const noexcept;

    // =========================================================================
    // Analysis and Reporting
    // =========================================================================

    /**
     * @brief Get optimization suggestions
     */
    [[nodiscard]] std::vector<OptimizationSuggestion> getOptimizationSuggestions() const;

    /**
     * @brief Print summary to stream
     */
    void printSummary(std::ostream& os) const;

    /**
     * @brief Print summary to stdout
     */
    void printSummary() const;

    /**
     * @brief Get summary as string
     */
    [[nodiscard]] std::string getSummary() const;

    /**
     * @brief Print detailed breakdown
     */
    void printDetailed(std::ostream& os) const;

    /**
     * @brief Export statistics to JSON
     */
    [[nodiscard]] std::string exportJSON() const;

    /**
     * @brief Export statistics to CSV
     */
    [[nodiscard]] std::string exportCSV() const;

    // =========================================================================
    // Comparison
    // =========================================================================

    /**
     * @brief Compare with baseline statistics
     */
    [[nodiscard]] std::string compareWith(const AssemblyStatistics& baseline) const;

    /**
     * @brief Merge statistics from another collector
     */
    void merge(const AssemblyStatistics& other);

    // =========================================================================
    // Thread-Safe Recording (for parallel assembly)
    // =========================================================================

    /**
     * @brief Thread-safe timing recording
     */
    void recordTimingThreadSafe(TimingCategory category, double seconds);

    /**
     * @brief Thread-safe element recording
     */
    void recordElementThreadSafe(GlobalIndex cell_id, ElementType type);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Global Statistics Access
// ============================================================================

/**
 * @brief Get global statistics collector
 *
 * Provides access to a process-wide statistics collector for convenience.
 */
AssemblyStatistics& globalAssemblyStatistics();

/**
 * @brief Reset global statistics
 */
void resetGlobalStatistics();

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Format bytes as human-readable string
 */
std::string formatBytes(std::size_t bytes);

/**
 * @brief Format time as human-readable string
 */
std::string formatTime(double seconds);

/**
 * @brief Format FLOP rate as human-readable string
 */
std::string formatFLOPRate(double flops_per_second);

/**
 * @brief Format percentage
 */
std::string formatPercent(double fraction);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_ASSEMBLY_STATISTICS_H
