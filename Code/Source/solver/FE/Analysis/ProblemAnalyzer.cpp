/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/AnalysisSummaryProducer.h"
#include "Analysis/KernelAnalyzer.h"
#include "Analysis/CouplingGraphAnalyzer.h"
#include "Analysis/ConstraintRankAnalyzer.h"
#include "Analysis/MixedOperatorAnalyzer.h"
#include "Analysis/OperatorClassAnalyzer.h"
#include "Analysis/StabilizationAnalyzer.h"
#include "Analysis/CompatibilityAnalyzer.h"
#include "Analysis/TopologyScopeAnalyzer.h"
#include "Analysis/InterfaceValidationAnalyzer.h"
#include "Analysis/InfSupAnalyzer.h"
#include "Analysis/TransportCharacterAnalyzer.h"
#include "Analysis/ConservationAnalyzer.h"
#include "Analysis/DAEStructureAnalyzer.h"
#include "Analysis/SpaceCompatibilityAnalyzer.h"
#include "Analysis/DiscreteMonotonicityAnalyzer.h"
#include "Analysis/MeshGeometryAnalyzer.h"
#include "Analysis/AdvancedStabilityAnalyzers.h"
#include "Analysis/SolverCompatibilityAnalyzer.h"
#include "Analysis/NumericSummaryPlanner.h"
#include "Analysis/FortinOperatorAutogeneration.h"

#include <chrono>
#include <exception>

namespace svmp {
namespace FE {
namespace analysis {

void ProblemAnalyzer::addPass(std::unique_ptr<AnalyzerPass> pass) {
    if (pass) {
        passes_.push_back(std::move(pass));
    }
}

std::vector<std::string> ProblemAnalyzer::passNames() const {
    std::vector<std::string> names;
    names.reserve(passes_.size());
    for (const auto& p : passes_) {
        names.push_back(p->name());
    }
    return names;
}

ProblemAnalysisReport ProblemAnalyzer::analyze(const ProblemAnalysisContext& context) const {
    ProblemAnalysisReport report;
    for (const auto& pass : passes_) {
        const auto claims_before = report.claims.size();
        const auto issues_before = report.issues.size();
        const auto requests_before = report.request_plan.summary_requests.size();
        const auto start = std::chrono::steady_clock::now();

        AnalyzerRunLogSummary log;
        log.analyzer = pass->name();
        log.pass_name = pass->name();
        log.pass_version = "1";
        log.attempted_count = 1u;

        try {
            pass->run(context, report);
            log.status = "completed";
        } catch (const std::exception& e) {
            log.status = "failed";
            log.message = e.what();
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Error;
            issue.message =
                "Analysis pass '" + pass->name() +
                "' failed: " + e.what();
            report.issues.push_back(std::move(issue));
        } catch (...) {
            log.status = "failed";
            log.message = "unknown exception";
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Error;
            issue.message =
                "Analysis pass '" + pass->name() +
                "' failed with an unknown exception";
            report.issues.push_back(std::move(issue));
        }

        const auto stop = std::chrono::steady_clock::now();
        log.runtime_microseconds =
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    stop - start).count());
        log.claims_added =
            static_cast<std::uint64_t>(report.claims.size() - claims_before);
        log.issues_added =
            static_cast<std::uint64_t>(report.issues.size() - issues_before);
        log.requests_added = static_cast<std::uint64_t>(
            report.request_plan.summary_requests.size() - requests_before);
        report.run_logs.push_back(std::move(log));
    }
    return report;
}

ProblemAnalysisReport ProblemAnalyzer::analyzeWithEvidenceSynthesis(
    const ProblemAnalysisContext& context,
    const AnalysisSummaryProducerRegistry& producer_registry,
    const AssemblyAccess& assembly,
    const MeshAccess& mesh,
    const SolverAccess& solver) const
{
    ProblemAnalysisContext working_context = context;
    AnalysisSummarySet working_summaries =
        context.analysisSummaries() ? *context.analysisSummaries()
                                    : AnalysisSummarySet{};
    ProblemAnalysisReport report;

    for (const auto& pass : passes_) {
        const auto claims_before = report.claims.size();
        const auto issues_before = report.issues.size();
        const auto requests_before = report.request_plan.summary_requests.size();
        const auto start = std::chrono::steady_clock::now();

        AnalyzerRunLogSummary log;
        log.analyzer = pass->name();
        log.pass_name = pass->name();
        log.pass_version = "1";
        log.attempted_count = 1u;

        try {
            pass->run(working_context, report);
            log.status = "completed";
        } catch (const std::exception& e) {
            log.status = "failed";
            log.message = e.what();
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Error;
            issue.message =
                "Analysis pass '" + pass->name() +
                "' failed: " + e.what();
            report.issues.push_back(std::move(issue));
        } catch (...) {
            log.status = "failed";
            log.message = "unknown exception";
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Error;
            issue.message =
                "Analysis pass '" + pass->name() +
                "' failed with an unknown exception";
            report.issues.push_back(std::move(issue));
        }

        const auto stop = std::chrono::steady_clock::now();
        log.runtime_microseconds =
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    stop - start).count());
        log.claims_added =
            static_cast<std::uint64_t>(report.claims.size() - claims_before);
        log.issues_added =
            static_cast<std::uint64_t>(report.issues.size() - issues_before);
        log.requests_added = static_cast<std::uint64_t>(
            report.request_plan.summary_requests.size() - requests_before);
        report.run_logs.push_back(std::move(log));

        if (pass->name() == "NumericSummaryPlanner") {
            auto batch = producer_registry.fulfillRequestPlan(
                report.request_plan,
                working_context,
                assembly,
                mesh,
                solver,
                working_summaries);
            mergeAnalysisSummarySets(working_summaries,
                                     batch.produced_summaries);
            working_context.setAnalysisSummaries(working_summaries);
            report.request_plan = std::move(batch.fulfilled_plan);
            for (const auto& issue : batch.issues) {
                report.issues.push_back(issue);
            }
            for (auto& run_log : batch.run_logs) {
                report.run_logs.push_back(std::move(run_log));
            }
        }
    }

    return report;
}

ProblemAnalyzer ProblemAnalyzer::createDefault() {
    ProblemAnalyzer analyzer;

    // Passes in dependency order:
    //   1. CouplingGraphAnalyzer (no dependencies)
    //   2. KernelAnalyzer (no dependencies)
    //   3. MixedOperatorAnalyzer (no dependencies)
    //   4. OperatorClassAnalyzer (no dependencies)
    //   5. StabilizationAnalyzer (no dependencies)
    //   6. ConstraintRankAnalyzer (reads KernelAnalyzer claims)
    //   7. CompatibilityAnalyzer (reads KernelAnalyzer + ConstraintRankAnalyzer claims)
    //   8. TopologyScopeAnalyzer (reads all prior claims)
    //   9. InterfaceValidationAnalyzer
    //  10. TransportCharacterAnalyzer (structural transport traits)
    //  11. ConservationAnalyzer (structural balance descriptors)
    //  12. DAEStructureAnalyzer (structural dynamic/algebraic descriptors)
    //  13. First NumericSummaryPlanner (seeds scoped evidence requests)
    //  14. FortinCertificationAnalyzer (request-plan-driven theorem matching)
    //  15. InfSupAnalyzer (reads Fortin/numeric inf-sup summaries when present)
    //  16. SpaceCompatibilityAnalyzer (reads spaces plus inf-sup/complex claims)
    //  17. NullspaceDegeneracyAnalyzer (reads reduced nullspace summaries)
    //  18. OperatorApplicabilityAnalyzer (reads applicability summaries)
    //  19. DiscreteMonotonicityAnalyzer (reads matrix/stencil summaries)
    //  20. MeshGeometryAnalyzer (reads mesh quality summaries)
    //  21. TemporalStabilityAnalyzer (reads temporal summaries)
    //  22. EnergyEntropyLawAnalyzer (reads balance-law summaries)
    //  23. CoefficientConstitutiveAnalyzer (reads coefficient summaries)
    //  24. NonlinearTangentAnalyzer (reads tangent summaries)
    //  25. LockingRiskAnalyzer (reads constraint/inf-sup/space evidence)
    //  26. SpectralSpuriousModeAnalyzer (reads spectral summaries)
    //  27. ErrorEstimatorAnalyzer (reads estimator summaries)
    //  28. QuadratureAdequacyAnalyzer (reads quadrature summaries)
    //  29. MinimumResidualStabilityAnalyzer (reads PG/DPG/min-res summaries)
    //  30. PreservationStructureAnalyzer (reads preservation/transfer summaries)
    //  31. CoupledSystemStabilityAnalyzer (reads coupling/DAE evidence)
    //  32. RobustnessTrendAnalyzer (reads cross-run metric summaries)
    //  33. SchurQualityAnalyzer (reads preconditioned Schur summaries)
    //  34. ToleranceAdequacyAnalyzer (reads numerical error budgets)
    //  35. SolverCompatibilityAnalyzer (reads claims + configured solver)
    //  36. Final NumericSummaryPlanner (sees summary-backed claims)

    analyzer.addPass(std::make_unique<CouplingGraphAnalyzer>());
    analyzer.addPass(std::make_unique<KernelAnalyzer>());
    analyzer.addPass(std::make_unique<MixedOperatorAnalyzer>());
    analyzer.addPass(std::make_unique<OperatorClassAnalyzer>());
    analyzer.addPass(std::make_unique<StabilizationAnalyzer>());
    analyzer.addPass(std::make_unique<ConstraintRankAnalyzer>());
    analyzer.addPass(std::make_unique<CompatibilityAnalyzer>());
    analyzer.addPass(std::make_unique<TopologyScopeAnalyzer>());
    analyzer.addPass(std::make_unique<InterfaceValidationAnalyzer>());
    analyzer.addPass(std::make_unique<TransportCharacterAnalyzer>());
    analyzer.addPass(std::make_unique<ConservationAnalyzer>());
    analyzer.addPass(std::make_unique<DAEStructureAnalyzer>());
    analyzer.addPass(std::make_unique<NumericSummaryPlanner>());
    analyzer.addPass(std::make_unique<FortinCertificationAnalyzer>());
    analyzer.addPass(std::make_unique<InfSupAnalyzer>());
    analyzer.addPass(std::make_unique<SpaceCompatibilityAnalyzer>());
    analyzer.addPass(std::make_unique<NullspaceDegeneracyAnalyzer>());
    analyzer.addPass(std::make_unique<OperatorApplicabilityAnalyzer>());
    analyzer.addPass(std::make_unique<DiscreteMonotonicityAnalyzer>());
    analyzer.addPass(std::make_unique<MeshGeometryAnalyzer>());
    analyzer.addPass(std::make_unique<TemporalStabilityAnalyzer>());
    analyzer.addPass(std::make_unique<EnergyEntropyLawAnalyzer>());
    analyzer.addPass(std::make_unique<CoefficientConstitutiveAnalyzer>());
    analyzer.addPass(std::make_unique<NonlinearTangentAnalyzer>());
    analyzer.addPass(std::make_unique<LockingRiskAnalyzer>());
    analyzer.addPass(std::make_unique<SpectralSpuriousModeAnalyzer>());
    analyzer.addPass(std::make_unique<ErrorEstimatorAnalyzer>());
    analyzer.addPass(std::make_unique<QuadratureAdequacyAnalyzer>());
    analyzer.addPass(std::make_unique<MinimumResidualStabilityAnalyzer>());
    analyzer.addPass(std::make_unique<PreservationStructureAnalyzer>());
    analyzer.addPass(std::make_unique<CoupledSystemStabilityAnalyzer>());
    analyzer.addPass(std::make_unique<RobustnessTrendAnalyzer>());
    analyzer.addPass(std::make_unique<SchurQualityAnalyzer>());
    analyzer.addPass(std::make_unique<ToleranceAdequacyAnalyzer>());
    analyzer.addPass(std::make_unique<SolverCompatibilityAnalyzer>());
    analyzer.addPass(std::make_unique<NumericSummaryPlanner>());

    return analyzer;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
