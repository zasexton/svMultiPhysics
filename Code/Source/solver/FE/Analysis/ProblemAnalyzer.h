/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_PROBLEM_ANALYZER_H
#define SVMP_FE_ANALYSIS_PROBLEM_ANALYZER_H

/**
 * @file ProblemAnalyzer.h
 * @brief Orchestrator for running analysis passes over a ProblemAnalysisContext
 *
 * ProblemAnalyzer maintains an ordered list of AnalyzerPass objects and runs
 * them sequentially over a ProblemAnalysisContext.  Each pass appends
 * PropertyClaims and AnalysisIssues to the shared ProblemAnalysisReport.
 *
 * Passes may read claims emitted by earlier passes (e.g. CompatibilityAnalyzer
 * reads KernelAnalyzer's nullspace claims).  The execution order matters.
 *
 * Usage:
 * @code
 *   ProblemAnalysisContext ctx;
 *   // ... populate ctx with formulation records, BCs, topology ...
 *
 *   auto analyzer = ProblemAnalyzer::createDefault();
 *   auto report = analyzer.analyze(ctx);
 *   report.print(std::cout);
 * @endcode
 *
 * @see AnalyzerPass for the abstract pass interface
 * @see ProblemAnalysisContext for the input metadata
 * @see ProblemAnalysisReport for the output
 */

#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/ProblemAnalysisContext.h"

#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

class AnalysisSummaryProducerRegistry;
class AssemblyAccess;
class MeshAccess;
class SolverAccess;

// ============================================================================
// AnalyzerPass — abstract base for individual analysis passes
// ============================================================================

/**
 * @brief Abstract base class for a single analysis pass
 *
 * Each pass inspects the ProblemAnalysisContext (and optionally prior claims
 * already in the report) and appends new PropertyClaims and/or AnalysisIssues.
 */
class AnalyzerPass {
public:
    virtual ~AnalyzerPass() = default;

    /// Human-readable name for diagnostics and logging
    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Execute this pass
     *
     * @param context   Read-only problem metadata
     * @param report    Mutable report — append claims and issues here.
     *                  Earlier passes' claims are already present and may be read.
     */
    virtual void run(const ProblemAnalysisContext& context,
                     ProblemAnalysisReport& report) const = 0;
};

// ============================================================================
// ProblemAnalyzer — orchestrator
// ============================================================================

/**
 * @brief Orchestrates analysis passes over problem metadata
 */
class ProblemAnalyzer {
public:
    ProblemAnalyzer() = default;

    /// Add a pass (executed in insertion order)
    void addPass(std::unique_ptr<AnalyzerPass> pass);

    /// Number of registered passes
    [[nodiscard]] std::size_t numPasses() const noexcept { return passes_.size(); }

    /// Names of registered passes in execution order
    [[nodiscard]] std::vector<std::string> passNames() const;

    /**
     * @brief Run all registered passes and produce a report
     *
     * @param context  Problem metadata populated by the FE system
     * @return         Aggregate report with all claims and issues
     */
    [[nodiscard]] ProblemAnalysisReport analyze(const ProblemAnalysisContext& context) const;

    /**
     * @brief Run all passes while fulfilling NumericSummaryPlanner requests
     *        before later summary-backed analyzers execute.
     *
     * This overload copies the input context, runs each `NumericSummaryPlanner`
     * pass, asks the producer registry to satisfy the scoped request plan, and
     * attaches any produced summaries to the working context for subsequent
     * passes. Backend hooks may decline a request; those failures are recorded
     * in request production fields, run logs, and issues without upgrading
     * mathematical claims.
     */
    [[nodiscard]] ProblemAnalysisReport analyzeWithEvidenceSynthesis(
        const ProblemAnalysisContext& context,
        const AnalysisSummaryProducerRegistry& producer_registry,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const;

    /**
     * @brief Create a ProblemAnalyzer with all built-in passes
     *
     * Passes are added in dependency order. The first `NumericSummaryPlanner`
     * runs after structural analyzers so `analyzeWithEvidenceSynthesis()` can
     * materialize scoped evidence before Fortin, inf-sup, matrix, temporal,
     * DAE, Schur, and other summary-backed analyzers run. A final planner pass
     * records follow-up requests introduced by summary-backed claims.
     */
    [[nodiscard]] static ProblemAnalyzer createDefault();

private:
    std::vector<std::unique_ptr<AnalyzerPass>> passes_;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_PROBLEM_ANALYZER_H
