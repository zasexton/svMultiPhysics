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

#include "AssemblyStatistics.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <mutex>
#include <cmath>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// Timing Category Names
// ============================================================================

const char* timingCategoryName(TimingCategory category) noexcept
{
    switch (category) {
        case TimingCategory::Total:          return "Total";
        case TimingCategory::KernelCompute:  return "KernelCompute";
        case TimingCategory::GlobalScatter:  return "GlobalScatter";
        case TimingCategory::Communication:  return "Communication";
        case TimingCategory::ConstraintApply: return "ConstraintApply";
        case TimingCategory::SparsityBuild:  return "SparsityBuild";
        case TimingCategory::Scheduling:     return "Scheduling";
        case TimingCategory::Coloring:       return "Coloring";
        case TimingCategory::Initialization: return "Initialization";
        case TimingCategory::Finalization:   return "Finalization";
        case TimingCategory::ContextSetup:   return "ContextSetup";
        case TimingCategory::DataPack:       return "DataPack";
        case TimingCategory::DataUnpack:     return "DataUnpack";
        case TimingCategory::Synchronization: return "Synchronization";
        case TimingCategory::Other:          return "Other";
        default:                             return "Unknown";
    }
}

// ============================================================================
// ScopedTimer Implementation
// ============================================================================

ScopedTimer::ScopedTimer(TimingData& target)
    : start_time_(Clock::now())
    , target_(&target)
{
}

ScopedTimer::ScopedTimer(std::function<void(double)> callback)
    : start_time_(Clock::now())
    , callback_(std::move(callback))
{
}

ScopedTimer::~ScopedTimer()
{
    if (!stopped_) {
        double elapsed_seconds = elapsed();
        if (target_) {
            target_->accumulate(elapsed_seconds);
        }
        if (callback_) {
            callback_(elapsed_seconds);
        }
    }
}

double ScopedTimer::elapsed() const
{
    auto now = Clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

double ScopedTimer::stop()
{
    double elapsed_seconds = elapsed();
    if (!stopped_) {
        if (target_) {
            target_->accumulate(elapsed_seconds);
        }
        if (callback_) {
            callback_(elapsed_seconds);
        }
        stopped_ = true;
    }
    return elapsed_seconds;
}

// ============================================================================
// FLOPEstimator Implementation
// ============================================================================

std::uint64_t FLOPEstimator::stiffnessMatrix(
    LocalIndex num_dofs, LocalIndex num_qpts, int dim)
{
    // For each DOF pair (i,j): for each quadrature point:
    //   - dim multiplications for gradient components
    //   - dim-1 additions for dot product
    //   - 1 FMA for weight multiplication and accumulation
    // Plus symmetry means we compute n(n+1)/2 pairs

    std::uint64_t n = static_cast<std::uint64_t>(num_dofs);
    std::uint64_t q = static_cast<std::uint64_t>(num_qpts);
    std::uint64_t d = static_cast<std::uint64_t>(dim);

    std::uint64_t pairs = n * (n + 1) / 2;
    std::uint64_t ops_per_pair_per_qpt = 2 * d + 1;  // grad_i*grad_j + accumulate

    return pairs * q * ops_per_pair_per_qpt;
}

std::uint64_t FLOPEstimator::massMatrix(
    LocalIndex num_dofs, LocalIndex num_qpts)
{
    // For each DOF pair: for each quadrature point:
    //   - 1 multiplication (phi_i * phi_j)
    //   - 1 FMA (weight * product + accumulate)

    std::uint64_t n = static_cast<std::uint64_t>(num_dofs);
    std::uint64_t q = static_cast<std::uint64_t>(num_qpts);
    std::uint64_t pairs = n * (n + 1) / 2;

    return pairs * q * 2;
}

std::uint64_t FLOPEstimator::loadVector(
    LocalIndex num_dofs, LocalIndex num_qpts)
{
    // For each DOF: for each quadrature point:
    //   - 1 FMA (weight * f * phi + accumulate)

    std::uint64_t n = static_cast<std::uint64_t>(num_dofs);
    std::uint64_t q = static_cast<std::uint64_t>(num_qpts);

    return n * q * 2;
}

std::uint64_t FLOPEstimator::jacobianEval(LocalIndex num_nodes, int dim)
{
    // Jacobian is dim x dim matrix
    // Each entry requires num_nodes multiplications + (num_nodes-1) additions

    std::uint64_t n = static_cast<std::uint64_t>(num_nodes);
    std::uint64_t d = static_cast<std::uint64_t>(dim);

    std::uint64_t ops_per_entry = 2 * n - 1;
    return d * d * ops_per_entry;
}

std::uint64_t FLOPEstimator::basisEval(
    LocalIndex num_dofs, LocalIndex num_qpts, int dim, int derivative_order)
{
    std::uint64_t n = static_cast<std::uint64_t>(num_dofs);
    std::uint64_t q = static_cast<std::uint64_t>(num_qpts);
    std::uint64_t d = static_cast<std::uint64_t>(dim);

    // Values only
    std::uint64_t flops = n * q * 5;  // Polynomial evaluation

    // Gradients
    if (derivative_order >= 1) {
        flops += n * q * d * 8;  // Derivative evaluation
    }

    // Hessians
    if (derivative_order >= 2) {
        flops += n * q * d * d * 10;
    }

    return flops;
}

// ============================================================================
// AssemblyStatistics Implementation
// ============================================================================

class AssemblyStatistics::Impl {
public:
    Impl() {
        // Enable all categories by default
        for (int i = 0; i <= static_cast<int>(TimingCategory::Other); ++i) {
            enabled_categories_[static_cast<TimingCategory>(i)] = true;
        }
    }

    std::mutex mutex_;  // For thread-safe operations

    bool collecting_{false};
    std::chrono::high_resolution_clock::time_point collection_start_;

    // Timing data per category
    std::map<TimingCategory, TimingData> timings_;
    std::map<TimingCategory, bool> enabled_categories_;
    std::map<TimingCategory, std::chrono::high_resolution_clock::time_point> timing_starts_;

    // FLOP data
    FLOPData flop_data_;

    // Memory stats
    MemoryStats memory_stats_;
    bool memory_tracking_{false};

    // Load balance
    LoadBalanceStats load_balance_;

    // Cache stats
    CacheStats cache_stats_;
    bool cache_stats_enabled_{false};

    // Counts
    std::atomic<GlobalIndex> total_elements_{0};
    std::atomic<GlobalIndex> total_matrix_entries_{0};
    std::atomic<GlobalIndex> total_vector_entries_{0};

    // Element type counts
    std::map<ElementType, GlobalIndex> element_type_counts_;
};

AssemblyStatistics::AssemblyStatistics()
    : impl_(std::make_unique<Impl>())
{
}

AssemblyStatistics::~AssemblyStatistics() = default;

AssemblyStatistics::AssemblyStatistics(AssemblyStatistics&& other) noexcept = default;
AssemblyStatistics& AssemblyStatistics::operator=(AssemblyStatistics&& other) noexcept = default;

void AssemblyStatistics::startCollection()
{
    impl_->collecting_ = true;
    impl_->collection_start_ = std::chrono::high_resolution_clock::now();
}

void AssemblyStatistics::endCollection()
{
    if (impl_->collecting_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_seconds = std::chrono::duration<double>(
            end_time - impl_->collection_start_).count();
        impl_->timings_[TimingCategory::Total].accumulate(total_seconds);
        impl_->collecting_ = false;
    }
    computeLoadBalance();
}

bool AssemblyStatistics::isCollecting() const noexcept
{
    return impl_->collecting_;
}

void AssemblyStatistics::reset()
{
    std::lock_guard<std::mutex> lock(impl_->mutex_);

    impl_->collecting_ = false;
    impl_->timings_.clear();
    impl_->timing_starts_.clear();
    impl_->flop_data_.reset();
    impl_->memory_stats_.reset();
    impl_->load_balance_ = LoadBalanceStats{};
    impl_->cache_stats_.reset();
    impl_->total_elements_ = 0;
    impl_->total_matrix_entries_ = 0;
    impl_->total_vector_entries_ = 0;
    impl_->element_type_counts_.clear();
}

void AssemblyStatistics::setEnabled(TimingCategory category, bool enabled)
{
    impl_->enabled_categories_[category] = enabled;
}

bool AssemblyStatistics::isEnabled(TimingCategory category) const noexcept
{
    auto it = impl_->enabled_categories_.find(category);
    return it != impl_->enabled_categories_.end() ? it->second : true;
}

void AssemblyStatistics::startTiming(TimingCategory category)
{
    if (isEnabled(category)) {
        impl_->timing_starts_[category] = std::chrono::high_resolution_clock::now();
    }
}

void AssemblyStatistics::stopTiming(TimingCategory category)
{
    if (!isEnabled(category)) return;

    auto it = impl_->timing_starts_.find(category);
    if (it != impl_->timing_starts_.end()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end_time - it->second).count();
        impl_->timings_[category].accumulate(seconds);
        impl_->timing_starts_.erase(it);
    }
}

ScopedTimer AssemblyStatistics::scopedTimer(TimingCategory category)
{
    return ScopedTimer(impl_->timings_[category]);
}

void AssemblyStatistics::recordTiming(TimingCategory category, double seconds)
{
    if (isEnabled(category)) {
        impl_->timings_[category].accumulate(seconds);
    }
}

const TimingData& AssemblyStatistics::getTiming(TimingCategory category) const
{
    static const TimingData empty;
    auto it = impl_->timings_.find(category);
    return it != impl_->timings_.end() ? it->second : empty;
}

const std::map<TimingCategory, TimingData>& AssemblyStatistics::getAllTimings() const
{
    return impl_->timings_;
}

void AssemblyStatistics::recordFLOPs(std::uint64_t flops)
{
    impl_->flop_data_.total_flops += flops;
}

void AssemblyStatistics::recordFLOPs(const FLOPData& data)
{
    impl_->flop_data_.merge(data);
}

const FLOPData& AssemblyStatistics::getFLOPData() const
{
    return impl_->flop_data_;
}

double AssemblyStatistics::getFLOPRate() const
{
    const auto& total = getTiming(TimingCategory::Total);
    return impl_->flop_data_.flopRate(total.total_seconds);
}

void AssemblyStatistics::enableMemoryTracking(bool enable)
{
    impl_->memory_tracking_ = enable;
}

void AssemblyStatistics::trackAllocation(std::size_t bytes)
{
    if (impl_->memory_tracking_) {
        impl_->memory_stats_.trackAllocation(bytes);
    }
}

void AssemblyStatistics::trackDeallocation(std::size_t bytes)
{
    if (impl_->memory_tracking_) {
        impl_->memory_stats_.trackDeallocation(bytes);
    }
}

const MemoryStats& AssemblyStatistics::getMemoryStats() const
{
    return impl_->memory_stats_;
}

void AssemblyStatistics::setNumThreads(int num_threads)
{
    impl_->load_balance_.num_threads = num_threads;
    impl_->load_balance_.thread_stats.resize(static_cast<std::size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        impl_->load_balance_.thread_stats[static_cast<std::size_t>(t)].thread_id = t;
    }
}

void AssemblyStatistics::recordThreadStats(int thread_id, const ThreadStats& stats)
{
    if (thread_id >= 0 &&
        thread_id < static_cast<int>(impl_->load_balance_.thread_stats.size())) {
        impl_->load_balance_.thread_stats[static_cast<std::size_t>(thread_id)] = stats;
    }
}

void AssemblyStatistics::recordElementCompletion(
    int thread_id, double compute_time, double scatter_time)
{
    if (thread_id >= 0 &&
        thread_id < static_cast<int>(impl_->load_balance_.thread_stats.size())) {
        auto& ts = impl_->load_balance_.thread_stats[static_cast<std::size_t>(thread_id)];
        ts.elements_processed++;
        ts.compute_seconds += compute_time;
        ts.scatter_seconds += scatter_time;
    }
}

const LoadBalanceStats& AssemblyStatistics::getLoadBalanceStats() const
{
    return impl_->load_balance_;
}

void AssemblyStatistics::computeLoadBalance()
{
    impl_->load_balance_.computeImbalance();
}

bool AssemblyStatistics::enableCacheStats()
{
    // Hardware counter integration would go here
    // Using PAPI or perf_event on Linux
    impl_->cache_stats_enabled_ = false;  // Not available without PAPI
    return impl_->cache_stats_enabled_;
}

void AssemblyStatistics::disableCacheStats()
{
    impl_->cache_stats_enabled_ = false;
}

const CacheStats& AssemblyStatistics::getCacheStats() const
{
    return impl_->cache_stats_;
}

void AssemblyStatistics::updateCacheStats()
{
    // Would read hardware counters here
}

void AssemblyStatistics::recordElement(GlobalIndex /*cell_id*/, ElementType type)
{
    ++impl_->total_elements_;
    impl_->element_type_counts_[type]++;
}

void AssemblyStatistics::recordMatrixEntries(GlobalIndex count)
{
    impl_->total_matrix_entries_ += count;
}

void AssemblyStatistics::recordVectorEntries(GlobalIndex count)
{
    impl_->total_vector_entries_ += count;
}

GlobalIndex AssemblyStatistics::totalElements() const noexcept
{
    return impl_->total_elements_.load();
}

GlobalIndex AssemblyStatistics::totalMatrixEntries() const noexcept
{
    return impl_->total_matrix_entries_.load();
}

GlobalIndex AssemblyStatistics::totalVectorEntries() const noexcept
{
    return impl_->total_vector_entries_.load();
}

std::vector<OptimizationSuggestion> AssemblyStatistics::getOptimizationSuggestions() const
{
    std::vector<OptimizationSuggestion> suggestions;

    const auto& total = getTiming(TimingCategory::Total);
    const auto& compute = getTiming(TimingCategory::KernelCompute);
    const auto& scatter = getTiming(TimingCategory::GlobalScatter);
    const auto& comm = getTiming(TimingCategory::Communication);
    const auto& sync = getTiming(TimingCategory::Synchronization);

    double total_time = total.total_seconds;
    if (total_time <= 0.0) return suggestions;

    // Check for communication bottleneck
    double comm_fraction = (comm.total_seconds + sync.total_seconds) / total_time;
    if (comm_fraction > 0.3) {
        OptimizationSuggestion s;
        s.action = OptimizationAction::ReduceCommunication;
        s.rationale = "Communication accounts for " +
                      formatPercent(comm_fraction) + " of total time";
        s.expected_improvement = 1.0 / (1.0 - comm_fraction * 0.5);
        s.priority = 10;
        suggestions.push_back(s);
    }

    // Check for scatter bottleneck
    double scatter_fraction = scatter.total_seconds / total_time;
    if (scatter_fraction > 0.4) {
        OptimizationSuggestion s;
        s.action = OptimizationAction::UseColoring;
        s.rationale = "Global scatter accounts for " +
                      formatPercent(scatter_fraction) + " - colored assembly may help";
        s.expected_improvement = 1.2;
        s.priority = 8;
        suggestions.push_back(s);
    }

    // Check for load imbalance
    const auto& lb = impl_->load_balance_;
    if (lb.imbalance_factor > 1.2) {
        OptimizationSuggestion s;
        s.action = OptimizationAction::RebalanceWork;
        s.rationale = "Load imbalance factor is " +
                      std::to_string(lb.imbalance_factor) + " (ideal: 1.0)";
        s.expected_improvement = lb.imbalance_factor / 1.1;
        s.priority = 9;
        suggestions.push_back(s);
    }

    // Check FLOP rate
    double flop_rate = getFLOPRate();
    if (flop_rate > 0 && flop_rate < 1e9) {  // Less than 1 GFLOP/s
        double compute_fraction = compute.total_seconds / total_time;
        if (compute_fraction > 0.5) {
            OptimizationSuggestion s;
            s.action = OptimizationAction::UseMatrixFree;
            s.rationale = "Low FLOP rate (" + formatFLOPRate(flop_rate) +
                          ") suggests memory-bound assembly";
            s.expected_improvement = 1.5;
            s.priority = 7;
            suggestions.push_back(s);
        }
    }

    // Check for cache efficiency
    if (impl_->cache_stats_.hasHardwareCounters()) {
        double l1_hit = impl_->cache_stats_.l1HitRate();
        if (l1_hit < 0.8) {
            OptimizationSuggestion s;
            s.action = OptimizationAction::UseCacheBlocking;
            s.rationale = "L1 cache hit rate is " + formatPercent(l1_hit) +
                          " - consider cache-blocked ordering";
            s.expected_improvement = 0.9 / l1_hit;
            s.priority = 6;
            suggestions.push_back(s);
        }
    }

    // Sort by priority
    std::sort(suggestions.begin(), suggestions.end());

    return suggestions;
}

void AssemblyStatistics::printSummary(std::ostream& os) const
{
    os << "\n=== Assembly Statistics Summary ===\n";

    const auto& total = getTiming(TimingCategory::Total);
    os << "Total Time:      " << formatTime(total.total_seconds) << "\n";
    os << "Elements:        " << totalElements() << "\n";
    os << "Matrix Entries:  " << totalMatrixEntries() << "\n";
    os << "Vector Entries:  " << totalVectorEntries() << "\n";

    if (total.total_seconds > 0) {
        os << "\n--- Timing Breakdown ---\n";
        for (const auto& [category, data] : impl_->timings_) {
            if (data.total_seconds > 0 && category != TimingCategory::Total) {
                double fraction = data.total_seconds / total.total_seconds;
                os << std::setw(16) << std::left << timingCategoryName(category)
                   << ": " << formatTime(data.total_seconds)
                   << " (" << formatPercent(fraction) << ")\n";
            }
        }
    }

    double flop_rate = getFLOPRate();
    if (flop_rate > 0) {
        os << "\n--- Performance ---\n";
        os << "Total FLOPs:     " << impl_->flop_data_.computeTotal() << "\n";
        os << "FLOP Rate:       " << formatFLOPRate(flop_rate) << "\n";
    }

    const auto& lb = impl_->load_balance_;
    if (lb.num_threads > 1) {
        os << "\n--- Load Balance ---\n";
        os << "Threads:         " << lb.num_threads << "\n";
        os << "Imbalance:       " << std::fixed << std::setprecision(2)
           << lb.imbalance_factor << "x\n";
        os << "Efficiency:      " << formatPercent(lb.efficiency) << "\n";
    }

    const auto& mem = impl_->memory_stats_;
    if (mem.peak_bytes > 0) {
        os << "\n--- Memory ---\n";
        os << "Peak Usage:      " << formatBytes(mem.peak_bytes) << "\n";
        os << "Allocations:     " << mem.allocation_count << "\n";
    }

    auto suggestions = getOptimizationSuggestions();
    if (!suggestions.empty()) {
        os << "\n--- Optimization Suggestions ---\n";
        for (const auto& s : suggestions) {
            os << "* " << s.rationale << "\n";
        }
    }

    os << "\n";
}

void AssemblyStatistics::printSummary() const
{
    printSummary(std::cout);
}

std::string AssemblyStatistics::getSummary() const
{
    std::ostringstream oss;
    printSummary(oss);
    return oss.str();
}

void AssemblyStatistics::printDetailed(std::ostream& os) const
{
    printSummary(os);

    os << "=== Detailed Timing ===\n";
    for (const auto& [category, data] : impl_->timings_) {
        os << timingCategoryName(category) << ":\n"
           << "  Total:   " << formatTime(data.total_seconds) << "\n"
           << "  Calls:   " << data.call_count << "\n"
           << "  Average: " << formatTime(data.average()) << "\n"
           << "  Min:     " << formatTime(data.min_seconds) << "\n"
           << "  Max:     " << formatTime(data.max_seconds) << "\n";
    }

    const auto& lb = impl_->load_balance_;
    if (!lb.thread_stats.empty()) {
        os << "\n=== Per-Thread Statistics ===\n";
        for (const auto& ts : lb.thread_stats) {
            os << "Thread " << ts.thread_id << ":\n"
               << "  Elements: " << ts.elements_processed << "\n"
               << "  Compute:  " << formatTime(ts.compute_seconds) << "\n"
               << "  Scatter:  " << formatTime(ts.scatter_seconds) << "\n"
               << "  Wait:     " << formatTime(ts.wait_seconds) << "\n";
        }
    }
}

std::string AssemblyStatistics::exportJSON() const
{
    std::ostringstream oss;
    oss << "{\n";

    oss << "  \"total_time_seconds\": " << getTiming(TimingCategory::Total).total_seconds << ",\n";
    oss << "  \"total_elements\": " << totalElements() << ",\n";
    oss << "  \"total_matrix_entries\": " << totalMatrixEntries() << ",\n";
    oss << "  \"total_vector_entries\": " << totalVectorEntries() << ",\n";

    oss << "  \"timings\": {\n";
    bool first = true;
    for (const auto& [category, data] : impl_->timings_) {
        if (!first) oss << ",\n";
        first = false;
        oss << "    \"" << timingCategoryName(category) << "\": {"
            << "\"total\": " << data.total_seconds << ", "
            << "\"calls\": " << data.call_count << ", "
            << "\"avg\": " << data.average() << "}";
    }
    oss << "\n  },\n";

    oss << "  \"flops\": " << impl_->flop_data_.computeTotal() << ",\n";
    oss << "  \"flop_rate\": " << getFLOPRate() << ",\n";

    oss << "  \"memory\": {\n";
    oss << "    \"peak_bytes\": " << impl_->memory_stats_.peak_bytes << ",\n";
    oss << "    \"allocations\": " << impl_->memory_stats_.allocation_count << "\n";
    oss << "  },\n";

    oss << "  \"load_balance\": {\n";
    oss << "    \"threads\": " << impl_->load_balance_.num_threads << ",\n";
    oss << "    \"imbalance\": " << impl_->load_balance_.imbalance_factor << ",\n";
    oss << "    \"efficiency\": " << impl_->load_balance_.efficiency << "\n";
    oss << "  }\n";

    oss << "}\n";
    return oss.str();
}

std::string AssemblyStatistics::exportCSV() const
{
    std::ostringstream oss;

    // Header
    oss << "category,total_seconds,call_count,avg_seconds,min_seconds,max_seconds\n";

    // Timing data
    for (const auto& [category, data] : impl_->timings_) {
        oss << timingCategoryName(category) << ","
            << data.total_seconds << ","
            << data.call_count << ","
            << data.average() << ","
            << data.min_seconds << ","
            << data.max_seconds << "\n";
    }

    return oss.str();
}

std::string AssemblyStatistics::compareWith(const AssemblyStatistics& baseline) const
{
    std::ostringstream oss;
    oss << "\n=== Performance Comparison ===\n";

    const auto& base_total = baseline.getTiming(TimingCategory::Total);
    const auto& this_total = getTiming(TimingCategory::Total);

    if (base_total.total_seconds > 0 && this_total.total_seconds > 0) {
        double speedup = base_total.total_seconds / this_total.total_seconds;
        oss << "Total Time: " << formatTime(this_total.total_seconds)
            << " vs " << formatTime(base_total.total_seconds)
            << " (speedup: " << std::fixed << std::setprecision(2) << speedup << "x)\n";
    }

    // Compare categories
    for (const auto& [category, data] : impl_->timings_) {
        const auto& base_data = baseline.getTiming(category);
        if (data.total_seconds > 0 && base_data.total_seconds > 0) {
            double speedup = base_data.total_seconds / data.total_seconds;
            oss << std::setw(16) << std::left << timingCategoryName(category)
                << ": " << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
    }

    return oss.str();
}

void AssemblyStatistics::merge(const AssemblyStatistics& other)
{
    for (const auto& [category, data] : other.impl_->timings_) {
        impl_->timings_[category].merge(data);
    }

    impl_->flop_data_.merge(other.impl_->flop_data_);
    impl_->total_elements_ += other.impl_->total_elements_.load();
    impl_->total_matrix_entries_ += other.impl_->total_matrix_entries_.load();
    impl_->total_vector_entries_ += other.impl_->total_vector_entries_.load();
}

void AssemblyStatistics::recordTimingThreadSafe(TimingCategory category, double seconds)
{
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->timings_[category].accumulate(seconds);
}

void AssemblyStatistics::recordElementThreadSafe(GlobalIndex cell_id, ElementType type)
{
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    recordElement(cell_id, type);
}

// ============================================================================
// Global Statistics
// ============================================================================

AssemblyStatistics& globalAssemblyStatistics()
{
    static AssemblyStatistics global_stats;
    return global_stats;
}

void resetGlobalStatistics()
{
    globalAssemblyStatistics().reset();
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string formatBytes(std::size_t bytes)
{
    std::ostringstream oss;

    if (bytes < 1024) {
        oss << bytes << " B";
    } else if (bytes < 1024 * 1024) {
        oss << std::fixed << std::setprecision(2) << (bytes / 1024.0) << " KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0)) << " MB";
    } else {
        oss << std::fixed << std::setprecision(2)
            << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB";
    }

    return oss.str();
}

std::string formatTime(double seconds)
{
    std::ostringstream oss;

    if (seconds < 1e-6) {
        oss << std::fixed << std::setprecision(2) << (seconds * 1e9) << " ns";
    } else if (seconds < 1e-3) {
        oss << std::fixed << std::setprecision(2) << (seconds * 1e6) << " us";
    } else if (seconds < 1.0) {
        oss << std::fixed << std::setprecision(2) << (seconds * 1e3) << " ms";
    } else if (seconds < 60.0) {
        oss << std::fixed << std::setprecision(2) << seconds << " s";
    } else {
        int minutes = static_cast<int>(seconds / 60.0);
        double remaining = seconds - minutes * 60.0;
        oss << minutes << " m " << std::fixed << std::setprecision(1) << remaining << " s";
    }

    return oss.str();
}

std::string formatFLOPRate(double flops_per_second)
{
    std::ostringstream oss;

    if (flops_per_second < 1e3) {
        oss << std::fixed << std::setprecision(2) << flops_per_second << " FLOP/s";
    } else if (flops_per_second < 1e6) {
        oss << std::fixed << std::setprecision(2) << (flops_per_second / 1e3) << " KFLOP/s";
    } else if (flops_per_second < 1e9) {
        oss << std::fixed << std::setprecision(2) << (flops_per_second / 1e6) << " MFLOP/s";
    } else if (flops_per_second < 1e12) {
        oss << std::fixed << std::setprecision(2) << (flops_per_second / 1e9) << " GFLOP/s";
    } else {
        oss << std::fixed << std::setprecision(2) << (flops_per_second / 1e12) << " TFLOP/s";
    }

    return oss.str();
}

std::string formatPercent(double fraction)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << (fraction * 100.0) << "%";
    return oss.str();
}

} // namespace assembly
} // namespace FE
} // namespace svmp
