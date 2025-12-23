/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_AssemblyStatistics.cpp
 * @brief Unit tests for AssemblyStatistics
 */

#include <gtest/gtest.h>
#include "Assembly/AssemblyStatistics.h"
#include "Core/Types.h"

#include <vector>
#include <memory>
#include <cmath>
#include <thread>
#include <chrono>
#include <sstream>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

// ============================================================================
// Test Fixtures
// ============================================================================

class AssemblyStatisticsTest : public ::testing::Test {
protected:
    void SetUp() override {
        stats_ = std::make_unique<AssemblyStatistics>();
    }

    void TearDown() override {
        stats_.reset();
    }

    std::unique_ptr<AssemblyStatistics> stats_;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, DefaultConstruction) {
    AssemblyStatistics stats;
    EXPECT_FALSE(stats.isCollecting());
}

TEST_F(AssemblyStatisticsTest, MoveConstruction) {
    stats_->startCollection();
    stats_->recordElement(0, ElementType::Tetra4);

    AssemblyStatistics stats2(std::move(*stats_));

    EXPECT_TRUE(stats2.isCollecting());
}

TEST_F(AssemblyStatisticsTest, MoveAssignment) {
    stats_->startCollection();
    stats_->recordElement(0, ElementType::Tetra4);

    AssemblyStatistics stats2;
    stats2 = std::move(*stats_);

    EXPECT_TRUE(stats2.isCollecting());
}

// ============================================================================
// Collection Control Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, StartCollection) {
    EXPECT_FALSE(stats_->isCollecting());

    stats_->startCollection();

    EXPECT_TRUE(stats_->isCollecting());
}

TEST_F(AssemblyStatisticsTest, EndCollection) {
    stats_->startCollection();
    EXPECT_TRUE(stats_->isCollecting());

    stats_->endCollection();

    EXPECT_FALSE(stats_->isCollecting());
}

TEST_F(AssemblyStatisticsTest, EndCollectionRecordsTotalTime) {
    stats_->startCollection();

    // Sleep briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    stats_->endCollection();

    const auto& total = stats_->getTiming(TimingCategory::Total);
    EXPECT_GT(total.total_seconds, 0.0);
}

TEST_F(AssemblyStatisticsTest, Reset) {
    stats_->startCollection();
    stats_->recordElement(0, ElementType::Tetra4);
    stats_->recordMatrixEntries(100);
    stats_->endCollection();

    stats_->reset();

    EXPECT_EQ(stats_->totalElements(), 0);
    EXPECT_EQ(stats_->totalMatrixEntries(), 0);
    EXPECT_FALSE(stats_->isCollecting());
}

// ============================================================================
// Timing Category Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, TimingCategoryName) {
    EXPECT_STREQ(timingCategoryName(TimingCategory::Total), "Total");
    EXPECT_STREQ(timingCategoryName(TimingCategory::KernelCompute), "KernelCompute");
    EXPECT_STREQ(timingCategoryName(TimingCategory::GlobalScatter), "GlobalScatter");
    EXPECT_STREQ(timingCategoryName(TimingCategory::Communication), "Communication");
    EXPECT_STREQ(timingCategoryName(TimingCategory::ConstraintApply), "ConstraintApply");
    EXPECT_STREQ(timingCategoryName(TimingCategory::SparsityBuild), "SparsityBuild");
    EXPECT_STREQ(timingCategoryName(TimingCategory::Scheduling), "Scheduling");
    EXPECT_STREQ(timingCategoryName(TimingCategory::Coloring), "Coloring");
    EXPECT_STREQ(timingCategoryName(TimingCategory::Initialization), "Initialization");
    EXPECT_STREQ(timingCategoryName(TimingCategory::Finalization), "Finalization");
    EXPECT_STREQ(timingCategoryName(TimingCategory::Other), "Other");
}

// ============================================================================
// Timing Data Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, TimingDataDefaults) {
    TimingData data;

    EXPECT_EQ(data.total_seconds, 0.0);
    EXPECT_EQ(data.min_seconds, 0.0);
    EXPECT_EQ(data.max_seconds, 0.0);
    EXPECT_EQ(data.call_count, 0u);
    EXPECT_EQ(data.average(), 0.0);
}

TEST_F(AssemblyStatisticsTest, TimingDataAccumulate) {
    TimingData data;

    data.accumulate(1.0);
    EXPECT_EQ(data.total_seconds, 1.0);
    EXPECT_EQ(data.min_seconds, 1.0);
    EXPECT_EQ(data.max_seconds, 1.0);
    EXPECT_EQ(data.call_count, 1u);

    data.accumulate(2.0);
    EXPECT_EQ(data.total_seconds, 3.0);
    EXPECT_EQ(data.min_seconds, 1.0);
    EXPECT_EQ(data.max_seconds, 2.0);
    EXPECT_EQ(data.call_count, 2u);

    EXPECT_EQ(data.average(), 1.5);
}

TEST_F(AssemblyStatisticsTest, TimingDataReset) {
    TimingData data;
    data.accumulate(1.0);
    data.accumulate(2.0);

    data.reset();

    EXPECT_EQ(data.total_seconds, 0.0);
    EXPECT_EQ(data.call_count, 0u);
}

TEST_F(AssemblyStatisticsTest, TimingDataMerge) {
    TimingData data1;
    data1.accumulate(1.0);
    data1.accumulate(2.0);

    TimingData data2;
    data2.accumulate(0.5);
    data2.accumulate(3.0);

    data1.merge(data2);

    EXPECT_EQ(data1.total_seconds, 6.5);
    EXPECT_EQ(data1.call_count, 4u);
    EXPECT_EQ(data1.min_seconds, 0.5);
    EXPECT_EQ(data1.max_seconds, 3.0);
}

// ============================================================================
// Scoped Timer Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, ScopedTimerWithTimingData) {
    TimingData data;

    {
        ScopedTimer timer(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_GT(data.total_seconds, 0.0);
    EXPECT_EQ(data.call_count, 1u);
}

TEST_F(AssemblyStatisticsTest, ScopedTimerWithCallback) {
    double recorded_time = 0.0;

    {
        ScopedTimer timer([&recorded_time](double t) { recorded_time = t; });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_GT(recorded_time, 0.0);
}

TEST_F(AssemblyStatisticsTest, ScopedTimerElapsed) {
    TimingData data;
    ScopedTimer timer(data);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    double elapsed = timer.elapsed();
    EXPECT_GT(elapsed, 0.0);
}

TEST_F(AssemblyStatisticsTest, ScopedTimerStop) {
    TimingData data;
    ScopedTimer timer(data);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    double stopped_time = timer.stop();
    EXPECT_GT(stopped_time, 0.0);

    // Timer should only record once
    EXPECT_EQ(data.call_count, 1u);
}

// ============================================================================
// Start/Stop Timing Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, StartStopTiming) {
    stats_->startTiming(TimingCategory::KernelCompute);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stats_->stopTiming(TimingCategory::KernelCompute);

    const auto& data = stats_->getTiming(TimingCategory::KernelCompute);
    EXPECT_GT(data.total_seconds, 0.0);
    EXPECT_EQ(data.call_count, 1u);
}

TEST_F(AssemblyStatisticsTest, RecordTiming) {
    stats_->recordTiming(TimingCategory::GlobalScatter, 0.5);
    stats_->recordTiming(TimingCategory::GlobalScatter, 0.3);

    const auto& data = stats_->getTiming(TimingCategory::GlobalScatter);
    EXPECT_EQ(data.total_seconds, 0.8);
    EXPECT_EQ(data.call_count, 2u);
}

TEST_F(AssemblyStatisticsTest, EnableDisableCategory) {
    stats_->setEnabled(TimingCategory::Other, false);
    EXPECT_FALSE(stats_->isEnabled(TimingCategory::Other));

    stats_->recordTiming(TimingCategory::Other, 1.0);

    // Should not record when disabled
    const auto& data = stats_->getTiming(TimingCategory::Other);
    EXPECT_EQ(data.total_seconds, 0.0);
}

TEST_F(AssemblyStatisticsTest, GetAllTimings) {
    stats_->recordTiming(TimingCategory::KernelCompute, 0.5);
    stats_->recordTiming(TimingCategory::GlobalScatter, 0.3);

    const auto& all = stats_->getAllTimings();

    EXPECT_GE(all.size(), 2u);
    EXPECT_TRUE(all.count(TimingCategory::KernelCompute) > 0);
    EXPECT_TRUE(all.count(TimingCategory::GlobalScatter) > 0);
}

// ============================================================================
// FLOP Data Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, FLOPDataDefaults) {
    FLOPData data;

    EXPECT_EQ(data.total_flops, 0u);
    EXPECT_EQ(data.add_ops, 0u);
    EXPECT_EQ(data.mul_ops, 0u);
    EXPECT_EQ(data.div_ops, 0u);
    EXPECT_EQ(data.fma_ops, 0u);
}

TEST_F(AssemblyStatisticsTest, FLOPDataComputeTotal) {
    FLOPData data;
    data.add_ops = 100;
    data.mul_ops = 200;
    data.div_ops = 50;
    data.fma_ops = 75;

    // FMA counts as 2 ops
    EXPECT_EQ(data.computeTotal(), 100u + 200u + 50u + 150u);
}

TEST_F(AssemblyStatisticsTest, FLOPDataFlopRate) {
    FLOPData data;
    data.total_flops = 1000000;

    EXPECT_DOUBLE_EQ(data.flopRate(1.0), 1000000.0);
    EXPECT_DOUBLE_EQ(data.flopRate(0.5), 2000000.0);
    EXPECT_DOUBLE_EQ(data.flopRate(0.0), 0.0);
}

TEST_F(AssemblyStatisticsTest, RecordFLOPs) {
    stats_->recordFLOPs(1000);
    stats_->recordFLOPs(500);

    const auto& data = stats_->getFLOPData();
    EXPECT_EQ(data.total_flops, 1500u);
}

TEST_F(AssemblyStatisticsTest, RecordFLOPsWithData) {
    FLOPData flops;
    flops.add_ops = 100;
    flops.mul_ops = 200;

    stats_->recordFLOPs(flops);

    const auto& data = stats_->getFLOPData();
    EXPECT_EQ(data.add_ops, 100u);
    EXPECT_EQ(data.mul_ops, 200u);
}

// ============================================================================
// FLOP Estimator Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, FLOPEstimatorStiffnessMatrix) {
    auto flops = FLOPEstimator::stiffnessMatrix(4, 4, 3);
    EXPECT_GT(flops, 0u);
}

TEST_F(AssemblyStatisticsTest, FLOPEstimatorMassMatrix) {
    auto flops = FLOPEstimator::massMatrix(4, 4);
    EXPECT_GT(flops, 0u);
}

TEST_F(AssemblyStatisticsTest, FLOPEstimatorLoadVector) {
    auto flops = FLOPEstimator::loadVector(4, 4);
    EXPECT_GT(flops, 0u);
}

TEST_F(AssemblyStatisticsTest, FLOPEstimatorJacobianEval) {
    auto flops = FLOPEstimator::jacobianEval(4, 3);
    EXPECT_GT(flops, 0u);
}

TEST_F(AssemblyStatisticsTest, FLOPEstimatorBasisEval) {
    auto flops_values = FLOPEstimator::basisEval(4, 4, 3, 0);
    auto flops_grads = FLOPEstimator::basisEval(4, 4, 3, 1);
    auto flops_hess = FLOPEstimator::basisEval(4, 4, 3, 2);

    EXPECT_GT(flops_values, 0u);
    EXPECT_GT(flops_grads, flops_values);
    EXPECT_GT(flops_hess, flops_grads);
}

// ============================================================================
// Memory Statistics Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, MemoryStatsDefaults) {
    MemoryStats stats;

    EXPECT_EQ(stats.peak_bytes, 0u);
    EXPECT_EQ(stats.current_bytes, 0u);
    EXPECT_EQ(stats.total_allocated, 0u);
    EXPECT_EQ(stats.allocation_count, 0u);
}

TEST_F(AssemblyStatisticsTest, MemoryStatsTrackAllocation) {
    MemoryStats stats;

    stats.trackAllocation(1000);
    EXPECT_EQ(stats.current_bytes, 1000u);
    EXPECT_EQ(stats.peak_bytes, 1000u);
    EXPECT_EQ(stats.allocation_count, 1u);

    stats.trackAllocation(500);
    EXPECT_EQ(stats.current_bytes, 1500u);
    EXPECT_EQ(stats.peak_bytes, 1500u);
}

TEST_F(AssemblyStatisticsTest, MemoryStatsTrackDeallocation) {
    MemoryStats stats;

    stats.trackAllocation(1000);
    stats.trackDeallocation(400);

    EXPECT_EQ(stats.current_bytes, 600u);
    EXPECT_EQ(stats.peak_bytes, 1000u);  // Peak unchanged
    EXPECT_EQ(stats.deallocation_count, 1u);
}

TEST_F(AssemblyStatisticsTest, EnableMemoryTracking) {
    stats_->enableMemoryTracking(true);
    stats_->trackAllocation(1000);

    const auto& mem = stats_->getMemoryStats();
    EXPECT_EQ(mem.current_bytes, 1000u);
}

TEST_F(AssemblyStatisticsTest, DisableMemoryTracking) {
    stats_->enableMemoryTracking(false);
    stats_->trackAllocation(1000);

    const auto& mem = stats_->getMemoryStats();
    EXPECT_EQ(mem.current_bytes, 0u);  // Not tracked
}

// ============================================================================
// Load Balance Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, ThreadStatsDefaults) {
    ThreadStats ts;

    EXPECT_EQ(ts.thread_id, 0);
    EXPECT_EQ(ts.elements_processed, 0);
    EXPECT_EQ(ts.compute_seconds, 0.0);
    EXPECT_EQ(ts.totalTime(), 0.0);
}

TEST_F(AssemblyStatisticsTest, ThreadStatsTotalTime) {
    ThreadStats ts;
    ts.compute_seconds = 1.0;
    ts.scatter_seconds = 0.5;
    ts.wait_seconds = 0.2;

    EXPECT_DOUBLE_EQ(ts.totalTime(), 1.7);
}

TEST_F(AssemblyStatisticsTest, SetNumThreads) {
    stats_->setNumThreads(4);

    const auto& lb = stats_->getLoadBalanceStats();
    EXPECT_EQ(lb.num_threads, 4);
    EXPECT_EQ(lb.thread_stats.size(), 4u);
}

TEST_F(AssemblyStatisticsTest, RecordThreadStats) {
    stats_->setNumThreads(4);

    ThreadStats ts;
    ts.thread_id = 0;
    ts.elements_processed = 100;
    ts.compute_seconds = 0.5;

    stats_->recordThreadStats(0, ts);

    const auto& lb = stats_->getLoadBalanceStats();
    EXPECT_EQ(lb.thread_stats[0].elements_processed, 100);
}

TEST_F(AssemblyStatisticsTest, RecordElementCompletion) {
    stats_->setNumThreads(4);

    stats_->recordElementCompletion(0, 0.1, 0.05);
    stats_->recordElementCompletion(0, 0.2, 0.03);

    const auto& lb = stats_->getLoadBalanceStats();
    EXPECT_EQ(lb.thread_stats[0].elements_processed, 2);
    EXPECT_DOUBLE_EQ(lb.thread_stats[0].compute_seconds, 0.3);
    EXPECT_DOUBLE_EQ(lb.thread_stats[0].scatter_seconds, 0.08);
}

TEST_F(AssemblyStatisticsTest, LoadBalanceStatsComputeImbalance) {
    LoadBalanceStats lb;
    lb.num_threads = 4;
    lb.thread_stats.resize(4);

    lb.thread_stats[0].compute_seconds = 1.0;
    lb.thread_stats[1].compute_seconds = 1.2;
    lb.thread_stats[2].compute_seconds = 0.8;
    lb.thread_stats[3].compute_seconds = 1.5;

    lb.computeImbalance();

    // Imbalance should be > 1 since max > avg
    EXPECT_GT(lb.imbalance_factor, 1.0);
}

// ============================================================================
// Cache Statistics Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, CacheStatsDefaults) {
    CacheStats stats;

    EXPECT_EQ(stats.l1_hits, 0u);
    EXPECT_EQ(stats.l1_misses, 0u);
    EXPECT_FALSE(stats.hasHardwareCounters());
}

TEST_F(AssemblyStatisticsTest, CacheStatsHitRate) {
    CacheStats stats;
    stats.l1_hits = 900;
    stats.l1_misses = 100;

    EXPECT_DOUBLE_EQ(stats.l1HitRate(), 0.9);
}

TEST_F(AssemblyStatisticsTest, CacheStatsHitRateZero) {
    CacheStats stats;
    EXPECT_DOUBLE_EQ(stats.l1HitRate(), 0.0);
}

// ============================================================================
// Element/Entry Recording Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, RecordElement) {
    stats_->recordElement(0, ElementType::Tetra4);
    stats_->recordElement(1, ElementType::Tetra4);
    stats_->recordElement(2, ElementType::Hex8);

    EXPECT_EQ(stats_->totalElements(), 3);
}

TEST_F(AssemblyStatisticsTest, RecordMatrixEntries) {
    stats_->recordMatrixEntries(100);
    stats_->recordMatrixEntries(200);

    EXPECT_EQ(stats_->totalMatrixEntries(), 300);
}

TEST_F(AssemblyStatisticsTest, RecordVectorEntries) {
    stats_->recordVectorEntries(50);
    stats_->recordVectorEntries(30);

    EXPECT_EQ(stats_->totalVectorEntries(), 80);
}

// ============================================================================
// Optimization Suggestion Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, OptimizationSuggestionDefaults) {
    OptimizationSuggestion s;

    EXPECT_EQ(s.action, OptimizationAction::None);
    EXPECT_TRUE(s.rationale.empty());
    EXPECT_EQ(s.expected_improvement, 0.0);
    EXPECT_EQ(s.priority, 0);
}

TEST_F(AssemblyStatisticsTest, OptimizationSuggestionOrdering) {
    OptimizationSuggestion s1, s2, s3;
    s1.priority = 5;
    s2.priority = 10;
    s3.priority = 3;

    EXPECT_TRUE(s2 < s1);  // Higher priority < lower priority (sorted first)
    EXPECT_TRUE(s1 < s3);
}

TEST_F(AssemblyStatisticsTest, GetOptimizationSuggestions) {
    stats_->startCollection();

    // Record some timing data
    stats_->recordTiming(TimingCategory::Total, 1.0);
    stats_->recordTiming(TimingCategory::Communication, 0.5);

    stats_->endCollection();

    auto suggestions = stats_->getOptimizationSuggestions();

    // Should suggest reducing communication since it's 50% of total
    bool found_comm_suggestion = false;
    for (const auto& s : suggestions) {
        if (s.action == OptimizationAction::ReduceCommunication) {
            found_comm_suggestion = true;
        }
    }
    EXPECT_TRUE(found_comm_suggestion);
}

// ============================================================================
// Reporting Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, PrintSummary) {
    stats_->startCollection();
    stats_->recordElement(0, ElementType::Tetra4);
    stats_->recordMatrixEntries(100);
    stats_->recordTiming(TimingCategory::KernelCompute, 0.5);
    stats_->endCollection();

    std::ostringstream oss;
    stats_->printSummary(oss);

    std::string summary = oss.str();
    EXPECT_FALSE(summary.empty());
    EXPECT_TRUE(summary.find("Assembly Statistics") != std::string::npos);
}

TEST_F(AssemblyStatisticsTest, GetSummary) {
    stats_->startCollection();
    stats_->recordElement(0, ElementType::Tetra4);
    stats_->endCollection();

    std::string summary = stats_->getSummary();
    EXPECT_FALSE(summary.empty());
}

TEST_F(AssemblyStatisticsTest, PrintDetailed) {
    stats_->startCollection();
    stats_->recordTiming(TimingCategory::KernelCompute, 0.5);
    stats_->recordTiming(TimingCategory::GlobalScatter, 0.3);
    stats_->endCollection();

    std::ostringstream oss;
    stats_->printDetailed(oss);

    std::string detailed = oss.str();
    EXPECT_FALSE(detailed.empty());
    EXPECT_TRUE(detailed.find("Detailed Timing") != std::string::npos);
}

TEST_F(AssemblyStatisticsTest, ExportJSON) {
    stats_->startCollection();
    stats_->recordElement(0, ElementType::Tetra4);
    stats_->recordMatrixEntries(100);
    stats_->endCollection();

    std::string json = stats_->exportJSON();

    EXPECT_FALSE(json.empty());
    EXPECT_TRUE(json.find("{") != std::string::npos);
    EXPECT_TRUE(json.find("total_time_seconds") != std::string::npos);
    EXPECT_TRUE(json.find("total_elements") != std::string::npos);
}

TEST_F(AssemblyStatisticsTest, ExportCSV) {
    stats_->recordTiming(TimingCategory::KernelCompute, 0.5);
    stats_->recordTiming(TimingCategory::GlobalScatter, 0.3);

    std::string csv = stats_->exportCSV();

    EXPECT_FALSE(csv.empty());
    EXPECT_TRUE(csv.find("category") != std::string::npos);
    EXPECT_TRUE(csv.find("KernelCompute") != std::string::npos);
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, CompareWith) {
    AssemblyStatistics baseline;
    baseline.startCollection();
    baseline.recordTiming(TimingCategory::Total, 2.0);
    baseline.endCollection();

    stats_->startCollection();
    stats_->recordTiming(TimingCategory::Total, 1.0);
    stats_->endCollection();

    std::string comparison = stats_->compareWith(baseline);

    EXPECT_FALSE(comparison.empty());
    EXPECT_TRUE(comparison.find("Comparison") != std::string::npos);
    EXPECT_TRUE(comparison.find("speedup") != std::string::npos);
}

TEST_F(AssemblyStatisticsTest, Merge) {
    AssemblyStatistics other;
    other.recordTiming(TimingCategory::KernelCompute, 0.5);
    other.recordElement(0, ElementType::Tetra4);
    other.recordElement(1, ElementType::Tetra4);

    stats_->recordTiming(TimingCategory::KernelCompute, 0.3);
    stats_->recordElement(0, ElementType::Hex8);

    stats_->merge(other);

    EXPECT_EQ(stats_->totalElements(), 3);
    EXPECT_DOUBLE_EQ(stats_->getTiming(TimingCategory::KernelCompute).total_seconds, 0.8);
}

// ============================================================================
// Thread-Safe Recording Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, RecordTimingThreadSafe) {
    stats_->recordTimingThreadSafe(TimingCategory::Other, 0.5);
    stats_->recordTimingThreadSafe(TimingCategory::Other, 0.3);

    const auto& data = stats_->getTiming(TimingCategory::Other);
    EXPECT_DOUBLE_EQ(data.total_seconds, 0.8);
}

TEST_F(AssemblyStatisticsTest, RecordElementThreadSafe) {
    stats_->recordElementThreadSafe(0, ElementType::Tetra4);
    stats_->recordElementThreadSafe(1, ElementType::Hex8);

    EXPECT_EQ(stats_->totalElements(), 2);
}

// ============================================================================
// Global Statistics Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, GlobalStatistics) {
    AssemblyStatistics& global = globalAssemblyStatistics();

    global.recordElement(0, ElementType::Tetra4);
    EXPECT_EQ(global.totalElements(), 1);

    resetGlobalStatistics();
    EXPECT_EQ(global.totalElements(), 0);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, FormatBytes) {
    EXPECT_EQ(formatBytes(500), "500 B");
    EXPECT_EQ(formatBytes(1024), "1.00 KB");
    EXPECT_EQ(formatBytes(1048576), "1.00 MB");
    EXPECT_EQ(formatBytes(1073741824), "1.00 GB");
}

TEST_F(AssemblyStatisticsTest, FormatTime) {
    EXPECT_TRUE(formatTime(1e-9).find("ns") != std::string::npos);
    EXPECT_TRUE(formatTime(1e-6).find("us") != std::string::npos);
    EXPECT_TRUE(formatTime(1e-3).find("ms") != std::string::npos);
    EXPECT_TRUE(formatTime(1.0).find(" s") != std::string::npos);
    EXPECT_TRUE(formatTime(120.0).find("m") != std::string::npos);
}

TEST_F(AssemblyStatisticsTest, FormatFLOPRate) {
    EXPECT_TRUE(formatFLOPRate(500.0).find("FLOP/s") != std::string::npos);
    EXPECT_TRUE(formatFLOPRate(5000.0).find("KFLOP/s") != std::string::npos);
    EXPECT_TRUE(formatFLOPRate(5e6).find("MFLOP/s") != std::string::npos);
    EXPECT_TRUE(formatFLOPRate(5e9).find("GFLOP/s") != std::string::npos);
    EXPECT_TRUE(formatFLOPRate(5e12).find("TFLOP/s") != std::string::npos);
}

TEST_F(AssemblyStatisticsTest, FormatPercent) {
    EXPECT_EQ(formatPercent(0.5), "50.0%");
    EXPECT_EQ(formatPercent(1.0), "100.0%");
    EXPECT_EQ(formatPercent(0.0), "0.0%");
}

// ============================================================================
// FLOP Rate Computation Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, GetFLOPRate) {
    stats_->startCollection();
    stats_->recordFLOPs(1000000);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stats_->endCollection();

    double rate = stats_->getFLOPRate();
    EXPECT_GT(rate, 0.0);
}

// ============================================================================
// Cache Stats Enable/Disable Tests
// ============================================================================

TEST_F(AssemblyStatisticsTest, EnableCacheStats) {
    // Hardware counters typically not available in test environment
    bool enabled = stats_->enableCacheStats();
    // Just verify it doesn't crash
    (void)enabled;

    stats_->disableCacheStats();
}

TEST_F(AssemblyStatisticsTest, UpdateCacheStats) {
    stats_->updateCacheStats();
    // Just verify it doesn't crash
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp
