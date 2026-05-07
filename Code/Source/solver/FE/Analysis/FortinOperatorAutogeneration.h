/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_FORTIN_OPERATOR_AUTOGENERATION_H
#define SVMP_FE_ANALYSIS_FORTIN_OPERATOR_AUTOGENERATION_H

/**
 * @file FortinOperatorAutogeneration.h
 * @brief FE-side Fortin/stable-pair theorem matching infrastructure.
 */

#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

enum class MixedCouplingFamily : std::uint8_t {
    VectorDivergenceScalarMultiplier,
    GradientDivergenceAdjoint,
    MixedHDivDivergence,
    TraceMultiplier,
    MortarConstraint,
    CurlDivDeRham,
    GenericMultiplierConstraint,
    Unknown,
};

enum class MixedCouplingEvidenceStrength : std::uint8_t {
    Strong,
    Weak,
    Ambiguous,
    None,
};

enum class FortinEvidenceKind : std::uint8_t {
    KnownStablePair,
    ExplicitFortinConstruction,
    CommutingProjection,
    NumericEstimateOnly,
    StabilizedSurrogate,
};

enum class FortinBoundAvailability : std::uint8_t {
    Unavailable,
    Symbolic,
    Scoped,
    Numeric,
};

enum class FortinPolynomialOrderRelation : std::uint8_t {
    Equal,
    PrimalOneHigher,
    MiniP1P1,
};

enum class FortinRejectionReason : std::uint8_t {
    UnsupportedPair,
    MissingMetadata,
    WrongOrderRelation,
    WrongMeshFamily,
    MissingBoundaryNullspaceAssumption,
    ContradictedAssumption,
    StabilizedSurrogate,
    AmbiguousCoupling,
};

enum class FortinCandidateStatus : std::uint8_t {
    Complete,
    Incomplete,
    Blocked,
};

enum class LocalFortinProjectionStatus : std::uint8_t {
    NotRequested,
    MetadataOnly,
    Constructed,
    LocalDiagnosticPass,
    Failed,
    Unsupported,
};

struct MixedCouplingDescriptor {
    VariableKey primal_variable;
    VariableKey multiplier_variable;
    MixedCouplingFamily coupling_family{MixedCouplingFamily::Unknown};
    MixedCouplingEvidenceStrength evidence_strength{
        MixedCouplingEvidenceStrength::None};
    DomainKind domain{DomainKind::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};
    std::string operator_tag;
    std::string pairing_group;
    std::vector<std::string> contribution_ids;
    bool evidence_from_dag{false};
    bool evidence_from_contribution_role{false};
    bool evidence_from_pairing_metadata{false};
    bool has_stabilization_surrogate{false};
    std::vector<std::string> diagnostics;
};

struct FortinSpaceRequirement {
    std::vector<SpaceFamily> space_families;
    std::vector<ElementFamily> element_families;
    SpaceContinuityClass continuity_class{SpaceContinuityClass::Unknown};
    MappingTransform mapping_transform{MappingTransform::Unknown};
    int minimum_order{0};
    int maximum_order{-1};
};

struct FortinTheoremEntry {
    std::string theorem_id;
    std::string pair_family;
    MixedCouplingFamily coupling_family{MixedCouplingFamily::Unknown};
    FortinSpaceRequirement primal_space;
    FortinSpaceRequirement multiplier_space;
    FortinPolynomialOrderRelation order_relation{
        FortinPolynomialOrderRelation::Equal};
    std::vector<int> supported_dimensions;
    std::vector<ReferenceCellFamily> supported_cell_families;
    std::string mesh_assumption;
    std::string domain_assumption;
    std::string boundary_nullspace_assumption;
    FortinEvidenceKind evidence_kind{FortinEvidenceKind::KnownStablePair};
    FortinBoundAvailability beta_bound{FortinBoundAvailability::Symbolic};
    FortinBoundAvailability fortin_norm_bound{
        FortinBoundAvailability::Unavailable};
    Real beta_lower_bound{};
    Real fortin_norm_bound_value{};
};

struct FortinRegistryMatch {
    const FortinTheoremEntry* entry{nullptr};
    bool matched{false};
    std::vector<FortinRejectionReason> rejection_reasons;
    std::vector<std::string> diagnostics;
};

struct FortinProjectionPlan {
    std::string projection_type;
    std::string interpolation_or_projection;
    std::string local_moment_constraints;
    std::string correction_space;
    std::string preserved_quantity;
    std::string target_norm;
    bool local_operator_shape_metadata_present{false};
    int local_rows{-1};
    int local_cols{-1};
};

struct LocalFortinMatrix {
    std::string matrix_id;
    int rows{0};
    int cols{0};
    std::vector<Real> row_major_values;
};

struct FortinCandidate {
    MixedCouplingDescriptor coupling;
    FortinTheoremEntry theorem;
    FortinCandidateStatus status{FortinCandidateStatus::Incomplete};
    bool stable_pair_only{false};
    bool constructive_fortin{false};
    bool commuting_projection{false};
    bool mesh_assumption_satisfied{false};
    bool domain_assumption_satisfied{false};
    bool boundary_nullspace_assumption_satisfied{false};
    std::string global_constraint_handling;
    std::optional<FortinProjectionPlan> projection_plan;
    std::vector<FortinRejectionReason> missing_or_rejected_reasons;
    std::vector<std::string> diagnostics;
};

struct FortinCandidateBuildResult {
    std::vector<FortinCandidate> candidates;
    std::vector<std::string> diagnostics;
};

struct LocalFortinProjectionOptions {
    bool build_local_projection_matrices{false};
    bool verify_preservation_identities{false};
    bool estimate_norm_bound{false};
    int quadrature_order{-1};
};

struct LocalFortinProjectionResult {
    LocalFortinProjectionStatus status{
        LocalFortinProjectionStatus::NotRequested};
    FortinProjectionPlan plan;
    std::string reference_projection_id;
    ReferenceCellFamily reference_cell_family{ReferenceCellFamily::Unknown};
    int topological_dimension{0};
    int source_polynomial_order{-1};
    int target_dof_count{0};
    int source_dof_count{0};
    int multiplier_dof_count{0};
    int independent_moment_constraint_count{0};
    int quadrature_order{-1};
    bool local_operator_shape_metadata_present{false};
    bool local_projection_matrix_present{false};
    bool divergence_preservation_metadata_present{false};
    bool trace_preservation_metadata_present{false};
    bool commuting_projection_metadata_present{false};
    bool preservation_identity_verified{false};
    Real preservation_residual_frobenius_norm{};
    Real preservation_residual_max_abs{};
    Real preservation_residual_tolerance{};
    bool norm_bound_estimate_present{false};
    Real norm_bound_estimate{};
    std::string norm_bound_estimate_kind;
    std::vector<LocalFortinMatrix> local_matrices;
    std::vector<std::string> diagnostics;
};

class MixedCouplingClassifier {
public:
    [[nodiscard]] std::vector<MixedCouplingDescriptor>
    classify(const ProblemAnalysisContext& context) const;
};

class FortinTheoremRegistry {
public:
    FortinTheoremRegistry();

    [[nodiscard]] const std::vector<FortinTheoremEntry>& entries() const noexcept {
        return entries_;
    }

    [[nodiscard]] FortinRegistryMatch
    match(const MixedCouplingDescriptor& coupling,
          const FieldDescriptor& primal,
          const FieldDescriptor& multiplier) const;

private:
    std::vector<FortinTheoremEntry> entries_;
};

class FortinCandidateBuilder {
public:
    FortinCandidateBuilder();

    [[nodiscard]] FortinCandidateBuildResult
    build(const ProblemAnalysisContext& context) const;

private:
    FortinTheoremRegistry registry_;
    MixedCouplingClassifier coupling_classifier_;
};

class LocalFortinProjectionBuilder {
public:
    [[nodiscard]] LocalFortinProjectionResult
    build(const FortinCandidate& candidate,
          const LocalFortinProjectionOptions& options) const;

    [[nodiscard]] LocalFortinProjectionResult
    build(const FortinCandidate& candidate,
          const ProblemAnalysisContext& context,
          const LocalFortinProjectionOptions& options) const;

private:
    [[nodiscard]] LocalFortinProjectionResult
    build(const FortinCandidate& candidate,
          const ProblemAnalysisContext* context,
          const LocalFortinProjectionOptions& options) const;
};

[[nodiscard]] std::optional<InfSupPairCertificationSummary>
makeInfSupPairCertificationSummary(const FortinCandidate& candidate);

[[nodiscard]] std::optional<InfSupPairCertificationSummary>
makeInfSupPairCertificationSummary(const FortinCandidate& candidate,
                                   const ProblemAnalysisContext& context);

class FortinCertificationAnalyzer : public AnalyzerPass {
public:
    FortinCertificationAnalyzer() = default;
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

[[nodiscard]] const char* toString(MixedCouplingFamily family) noexcept;
[[nodiscard]] const char* toString(MixedCouplingEvidenceStrength strength) noexcept;
[[nodiscard]] const char* toString(FortinEvidenceKind kind) noexcept;
[[nodiscard]] const char* toString(FortinBoundAvailability availability) noexcept;
[[nodiscard]] const char* toString(FortinRejectionReason reason) noexcept;
[[nodiscard]] const char* toString(FortinCandidateStatus status) noexcept;
[[nodiscard]] const char* toString(LocalFortinProjectionStatus status) noexcept;

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_FORTIN_OPERATOR_AUTOGENERATION_H
