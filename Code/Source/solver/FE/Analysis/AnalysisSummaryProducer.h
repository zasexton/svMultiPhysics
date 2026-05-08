/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_ANALYSIS_SUMMARY_PRODUCER_H
#define SVMP_FE_ANALYSIS_ANALYSIS_SUMMARY_PRODUCER_H

/**
 * @file AnalysisSummaryProducer.h
 * @brief Request-plan driven automatic evidence production for FE/Analysis.
 */

#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

class AssemblyAccess {
public:
    virtual ~AssemblyAccess() = default;

    [[nodiscard]] virtual bool evidencePending() const noexcept;

    [[nodiscard]] virtual std::optional<NormMetadataSummary>
    normMetadata(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<DiscreteMatrixSummary>
    discreteMatrixSummary(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<ReducedMatrixSummary>
    reducedMatrixSummary(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<CoefficientPropertySummary>
    coefficientProperties(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<BoundarySymbolSummary>
    boundarySymbol(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<InfSupEstimateSummary>
    infSupEstimate(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<DAEStructureEvidenceSummary>
    daeStructureEvidence(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<QuadratureAdequacySummary>
    quadratureAdequacy(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<NullspaceDegeneracySummary>
    nullspaceDegeneracy(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<ParameterScaleSummary>
    parameterScale(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<StabilizationAdequacySummary>
    stabilizationAdequacy(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<InitialCompatibilitySummary>
    initialCompatibility(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<LocalStencilSummary>
    localStencil(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<NumericalErrorBudgetSummary>
    numericalErrorBudget(const AnalysisSummaryRequest& request) const;
};

class MeshAccess {
public:
    virtual ~MeshAccess() = default;

    [[nodiscard]] virtual bool evidencePending() const noexcept;

    [[nodiscard]] virtual std::string meshRevision() const;

    [[nodiscard]] virtual std::optional<MeshGeometryQualitySummary>
    meshGeometryQuality(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<RobustnessTrendSummary>
    refinementExperiment(const AnalysisSummaryRequest& request,
                         const AssemblyAccess& assembly) const;
};

class SolverAccess {
public:
    virtual ~SolverAccess() = default;

    [[nodiscard]] virtual bool evidencePending() const noexcept;

    [[nodiscard]] virtual std::optional<TemporalStabilitySummary>
    temporalStability(const AnalysisSummaryRequest& request) const;

    [[nodiscard]] virtual std::optional<SchurComplementSummary>
    schurComplement(const AnalysisSummaryRequest& request) const;
};

enum class SummaryProductionStatus : std::uint8_t {
    Produced,
    AlreadyAvailable,
    Pending,
    Unavailable,
    Unsupported,
    Failed,
    FailedStrictScopeCoverage,
    Skipped,
};

[[nodiscard]] const char* toString(SummaryProductionStatus status) noexcept;

struct SummaryProductionResult {
    AnalysisSummaryRequest request;
    SummaryProductionStatus status{SummaryProductionStatus::Unavailable};
    AnalysisSummarySet produced_summaries;
    std::string producer_id;
    std::string producer_version;
    std::string message;
    std::vector<std::string> missing_backend_hooks;
    std::vector<std::string> missing_scope_components;
    std::vector<std::string> diagnostics;
    std::vector<AnalysisIssue> issues;

    [[nodiscard]] std::size_t producedSummaryCount() const noexcept {
        return produced_summaries.totalSummaryCount();
    }
};

struct SummaryProductionBatch {
    AnalysisRequestPlan fulfilled_plan;
    AnalysisSummarySet produced_summaries;
    std::vector<SummaryProductionResult> results;
    std::vector<AnalyzerRunLogSummary> run_logs;
    std::vector<AnalysisIssue> issues;
    std::uint64_t produced_count{0};
    std::uint64_t already_available_count{0};
    std::uint64_t pending_count{0};
    std::uint64_t unavailable_count{0};
    std::uint64_t failed_count{0};
};

class AnalysisSummaryProducer {
public:
    virtual ~AnalysisSummaryProducer() = default;

    [[nodiscard]] virtual std::string producerId() const = 0;
    [[nodiscard]] virtual std::string producerVersion() const { return "1"; }
    [[nodiscard]] virtual std::vector<AnalysisSummaryKind> producedKinds() const = 0;

    [[nodiscard]] virtual bool canProduce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context) const = 0;

    [[nodiscard]] virtual SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const = 0;
};

class AnalysisSummaryProducerRegistry {
public:
    void registerProducer(std::unique_ptr<AnalysisSummaryProducer> producer);

    [[nodiscard]] std::vector<const AnalysisSummaryProducer*>
    producersFor(AnalysisSummaryKind kind) const;

    [[nodiscard]] SummaryProductionBatch fulfillRequestPlan(
        const AnalysisRequestPlan& request_plan,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver,
        AnalysisSummarySet existing_summaries = {}) const;

    [[nodiscard]] static AnalysisSummaryProducerRegistry createDefault();

private:
    std::unordered_map<AnalysisSummaryKind,
                       std::vector<std::unique_ptr<AnalysisSummaryProducer>>>
        producers_;
};

class NormMetadataProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class TheoremApplicabilityProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class FortinStablePairProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class InfSupNumericalTestProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class MeshFamilyProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class DiscreteMatrixSummaryProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class ReducedMatrixSummaryProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class CoefficientBoundsProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class BoundarySymbolProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class TemporalStabilityProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class DAEStructureEvidenceProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class SchurComplementProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class RefinementExperimentRunner final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class QuadratureAdequacyProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class InvariantDomainProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class NullspaceDegeneracyProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class ParameterScaleProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class StabilizationAdequacyProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class InitialCompatibilityProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class LocalStencilProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

class NumericalErrorBudgetProducer final : public AnalysisSummaryProducer {
public:
    [[nodiscard]] std::string producerId() const override;
    [[nodiscard]] std::vector<AnalysisSummaryKind> producedKinds() const override;
    [[nodiscard]] bool canProduce(const AnalysisSummaryRequest& request,
                                  const ProblemAnalysisContext& context) const override;
    [[nodiscard]] SummaryProductionResult produce(
        const AnalysisSummaryRequest& request,
        const ProblemAnalysisContext& context,
        const AssemblyAccess& assembly,
        const MeshAccess& mesh,
        const SolverAccess& solver) const override;
};

void mergeAnalysisSummarySets(AnalysisSummarySet& target,
                              const AnalysisSummarySet& source);

[[nodiscard]] SummaryEvidenceMetadata makeSummaryEvidenceMetadata(
    const AnalysisSummaryRequest& request,
    AnalysisSummaryKind kind,
    const ProblemAnalysisContext& context,
    const MeshAccess& mesh,
    const std::string& producer_id,
    const std::string& producer_version);

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_ANALYSIS_SUMMARY_PRODUCER_H
