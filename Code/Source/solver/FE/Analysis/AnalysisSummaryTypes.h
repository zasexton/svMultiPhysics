/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_ANALYSIS_SUMMARY_TYPES_H
#define SVMP_FE_ANALYSIS_ANALYSIS_SUMMARY_TYPES_H

/**
 * @file AnalysisSummaryTypes.h
 * @brief Lightweight numeric-summary contracts for optional FE analysis evidence.
 *
 * These types describe compact diagnostics produced by mesh, constraint,
 * assembly, backend, and time-integration owners. They intentionally store only
 * scalar counts, tolerances, identifiers, and bounded worst-case samples. They
 * do not own VTK objects, sparse matrices, dense matrices, or backend handles.
 */

#include "Analysis/ContributionDescriptor.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Backends/Interfaces/BackendKind.h"
#include "Core/Types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

using GlobalDofId = GlobalIndex;
using ElementId = GlobalIndex;
using MeshRevisionId = std::uint64_t;
using CoefficientId = std::string;

inline constexpr std::size_t kDefaultWorstSampleLimit = 8;

enum class AnalysisSummaryKind : std::uint8_t {
    CoefficientProperties,
    DiscreteMatrix,
    ReducedMatrix,
    LocalStencil,
    MeshGeometryQuality,
    FluxBalance,
    TemporalStability,
    BoundarySymbol,
    InfSupEstimate,
    EnergyEntropyBalance,
    InvariantDomain,
    EquilibriumPreservation,
    MovingDomain,
    TransferOperator,
    AdjointConsistency,
    ParameterScale,
    InitialCompatibility,
    CompatibleComplex,
    NonlinearTangent,
    SpectralStructure,
    ErrorEstimator,
    QuadratureAdequacy,
    CoupledSystemStability,
};

inline const char* toString(AnalysisSummaryKind kind) noexcept {
    switch (kind) {
        case AnalysisSummaryKind::CoefficientProperties: return "CoefficientProperties";
        case AnalysisSummaryKind::DiscreteMatrix: return "DiscreteMatrix";
        case AnalysisSummaryKind::ReducedMatrix: return "ReducedMatrix";
        case AnalysisSummaryKind::LocalStencil: return "LocalStencil";
        case AnalysisSummaryKind::MeshGeometryQuality: return "MeshGeometryQuality";
        case AnalysisSummaryKind::FluxBalance: return "FluxBalance";
        case AnalysisSummaryKind::TemporalStability: return "TemporalStability";
        case AnalysisSummaryKind::BoundarySymbol: return "BoundarySymbol";
        case AnalysisSummaryKind::InfSupEstimate: return "InfSupEstimate";
        case AnalysisSummaryKind::EnergyEntropyBalance: return "EnergyEntropyBalance";
        case AnalysisSummaryKind::InvariantDomain: return "InvariantDomain";
        case AnalysisSummaryKind::EquilibriumPreservation: return "EquilibriumPreservation";
        case AnalysisSummaryKind::MovingDomain: return "MovingDomain";
        case AnalysisSummaryKind::TransferOperator: return "TransferOperator";
        case AnalysisSummaryKind::AdjointConsistency: return "AdjointConsistency";
        case AnalysisSummaryKind::ParameterScale: return "ParameterScale";
        case AnalysisSummaryKind::InitialCompatibility: return "InitialCompatibility";
        case AnalysisSummaryKind::CompatibleComplex: return "CompatibleComplex";
        case AnalysisSummaryKind::NonlinearTangent: return "NonlinearTangent";
        case AnalysisSummaryKind::SpectralStructure: return "SpectralStructure";
        case AnalysisSummaryKind::ErrorEstimator: return "ErrorEstimator";
        case AnalysisSummaryKind::QuadratureAdequacy: return "QuadratureAdequacy";
        case AnalysisSummaryKind::CoupledSystemStability: return "CoupledSystemStability";
    }
    return "Unknown";
}

enum class NumericSummaryScope : std::uint8_t {
    FullMatrix,
    ConstrainedFullMatrix,
    ReducedFreeFree,
    LocalElement,
    BoundaryBlock,
    InterfaceBlock,
};

enum class TensorRank : std::uint8_t {
    Scalar,
    Vector,
    Rank2Tensor,
    Rank3Tensor,
    Rank4Tensor,
    Unknown,
};

enum class SymmetryClass : std::uint8_t {
    Symmetric,
    Skew,
    Nonsymmetric,
    NotApplicable,
    Unknown,
};

enum class PositivityClass : std::uint8_t {
    Positive,
    Nonnegative,
    Negative,
    Nonpositive,
    Indefinite,
    Unknown,
};

enum class ConstraintReductionKind : std::uint8_t {
    None,
    StrongDirichletElimination,
    AffineTransform,
    PeriodicEquivalence,
    MultiplierRetained,
    PenaltyRetained,
    Unknown,
};

enum class BalanceSignClass : std::uint8_t {
    Nonpositive,
    Nonnegative,
    Zero,
    Unconstrained,
    Unknown,
};

enum class EnergyEntropyLawKind : std::uint8_t {
    Energy,
    Entropy,
    Generic,
};

enum class TangentConsistencyClass : std::uint8_t {
    Exact,
    Approximate,
    Frozen,
    Inconsistent,
    Unknown,
};

struct OperatorBlockId {
    std::vector<VariableKey> test_variables;
    std::vector<VariableKey> trial_variables;
    DomainKind domain{DomainKind::Cell};
    ContributionRole role{ContributionRole::DiagonalBlock};
    std::string operator_tag;
    int marker{-1};
};

struct MatrixEntrySample {
    GlobalDofId row{INVALID_GLOBAL_INDEX};
    GlobalDofId col{INVALID_GLOBAL_INDEX};
    Real value{};
    int owning_rank{0};
    std::uint64_t sample_index{0};
    std::string note;
};

[[nodiscard]] inline bool moreSevereSample(const MatrixEntrySample& a,
                                           const MatrixEntrySample& b) noexcept
{
    const auto av = std::abs(a.value);
    const auto bv = std::abs(b.value);
    if (av != bv) return av > bv;
    if (a.owning_rank != b.owning_rank) return a.owning_rank < b.owning_rank;
    if (a.row != b.row) return a.row < b.row;
    if (a.col != b.col) return a.col < b.col;
    return a.sample_index < b.sample_index;
}

inline void addBoundedWorstSample(std::vector<MatrixEntrySample>& samples,
                                  MatrixEntrySample sample,
                                  std::size_t limit = kDefaultWorstSampleLimit)
{
    if (limit == 0u) return;
    samples.push_back(std::move(sample));
    std::sort(samples.begin(), samples.end(), moreSevereSample);
    if (samples.size() > limit) {
        samples.resize(limit);
    }
}

struct CoefficientPropertySummary {
    CoefficientId coefficient;
    TensorRank tensor_rank{TensorRank::Unknown};
    SymmetryClass symmetry{SymmetryClass::Unknown};
    PositivityClass positivity{PositivityClass::Unknown};
    DomainKind domain{DomainKind::Cell};
    Real positivity_tolerance{};
    Real min_eigenvalue{};
    Real max_eigenvalue{};
    Real anisotropy_ratio{};
    Real contrast_ratio{};
    bool state_dependent{false};
    bool time_dependent{false};
};

struct DiscreteMatrixSummary {
    OperatorBlockId block;
    NumericSummaryScope scope{NumericSummaryScope::FullMatrix};
    std::optional<backends::BackendKind> backend_kind;
    GlobalIndex rows{0};
    GlobalIndex cols{0};

    bool square{false};
    bool structurally_symmetric{false};
    bool numerically_symmetric{false};
    bool symmetry_evidence_complete{false};

    Real sign_tolerance{};
    Real row_sum_tolerance{};
    Real symmetry_tolerance{};
    Real max_abs_entry{};
    Real max_abs_offdiag{};
    Real max_positive_offdiag{};
    Real max_symmetry_error{};
    Real min_row_sum{};
    Real max_row_sum{};
    Real max_abs_row_sum{};
    std::optional<Real> condition_estimate;

    std::uint64_t diagonal_count{0};
    std::uint64_t nonpositive_diagonal_count{0};
    std::uint64_t negative_diagonal_count{0};
    std::uint64_t near_zero_diagonal_count{0};
    std::uint64_t offdiag_count{0};
    std::uint64_t positive_offdiag_count{0};
    std::uint64_t negative_offdiag_count{0};
    std::uint64_t near_zero_offdiag_count{0};
    std::uint64_t row_sum_violation_count{0};

    std::size_t worst_entry_sample_limit{kDefaultWorstSampleLimit};
    std::vector<MatrixEntrySample> worst_entries;

    void addWorstEntry(MatrixEntrySample sample) {
        addBoundedWorstSample(worst_entries, std::move(sample), worst_entry_sample_limit);
    }
};

struct ReducedMatrixSummary {
    DiscreteMatrixSummary free_free_matrix;
    ConstraintReductionKind reduction_kind{ConstraintReductionKind::None};
    NumericSummaryScope eliminated_scope{NumericSummaryScope::ReducedFreeFree};
    std::uint64_t free_dof_count{0};
    std::uint64_t constrained_dof_count{0};
    std::uint64_t retained_multiplier_dof_count{0};
    bool affine_terms_accounted_for{false};
    bool reduction_exact_for_analysis{false};
};

struct LocalStencilSummary {
    OperatorBlockId block;
    ElementId element{INVALID_GLOBAL_INDEX};
    NumericSummaryScope scope{NumericSummaryScope::LocalElement};
    Real sign_tolerance{};
    std::uint64_t positive_offdiag_count{0};
    std::uint64_t negative_offdiag_count{0};
    std::uint64_t near_zero_offdiag_count{0};
    std::size_t worst_entry_sample_limit{kDefaultWorstSampleLimit};
    std::vector<MatrixEntrySample> worst_local_entries;

    void addWorstLocalEntry(MatrixEntrySample sample) {
        addBoundedWorstSample(worst_local_entries, std::move(sample), worst_entry_sample_limit);
    }
};

struct MeshGeometryQualitySummary {
    MeshRevisionId mesh_revision{};
    DomainKind domain{DomainKind::Cell};
    Real min_jacobian{};
    Real max_jacobian{};
    Real min_angle{};
    Real max_angle{};
    Real min_dihedral_angle{};
    Real max_dihedral_angle{};
    Real max_aspect_ratio{};
    Real max_anisotropy_alignment_indicator{};
    Real min_cut_cell_fraction{};
    Real max_cut_cell_fraction{};
    std::uint64_t inverted_element_count{0};
    std::uint64_t poor_quality_element_count{0};
    std::uint64_t cut_cell_count{0};
    std::size_t worst_element_sample_limit{kDefaultWorstSampleLimit};
    std::vector<ElementId> worst_elements;
};

struct FluxBalanceSummary {
    OperatorBlockId block;
    Real balance_tolerance{};
    Real local_residual_norm{};
    Real global_residual_norm{};
    Real interface_pair_residual_norm{};
    std::uint64_t local_violation_count{0};
    std::uint64_t interface_pair_count{0};
};

struct TemporalStabilitySummary {
    std::string time_scheme;
    TemporalStabilityClass stability_class{TemporalStabilityClass::Unknown};
    Real cfl_estimate{};
    Real eigenvalue_scale_estimate{};
    Real amplification_radius{};
    Real high_frequency_dissipation{};
};

struct BoundarySymbolSummary {
    OperatorBlockId block;
    int principal_operator_order{0};
    int boundary_operator_order{0};
    TraceCapabilityFlags trace_coverage{TraceCapabilityFlags::None};
    std::optional<bool> complementing_condition_satisfied;
    std::string evidence_scope;
};

struct InfSupEstimateSummary {
    OperatorBlockId block;
    VariableKey primal_variable;
    VariableKey multiplier_variable;
    Real estimate_value{};
    GlobalIndex test_rows{0};
    GlobalIndex test_cols{0};
    std::string estimate_scope;
    NullspaceHandlingClass nullspace_handling{NullspaceHandlingClass::Unknown};
};

struct EnergyEntropySummary {
    std::string energy_entropy_id;
    EnergyEntropyLawKind law_kind{EnergyEntropyLawKind::Generic};
    BalanceSignClass expected_production_sign{BalanceSignClass::Unknown};
    Real balance_tolerance{};
    Real observed_discrete_balance{};
    Real observed_production{};
    std::uint64_t violation_count{0};
};

struct InvariantDomainSummary {
    std::string invariant_set_id;
    std::vector<VariableKey> variables;
    Real lower_bound{};
    Real upper_bound{};
    bool lower_bound_active{false};
    bool upper_bound_active{false};
    bool limiter_evidence_present{false};
    std::uint64_t post_step_violation_count{0};
};

struct EquilibriumPreservationSummary {
    std::string equilibrium_id;
    Real flux_source_residual{};
    Real residual_tolerance{};
    bool source_quadrature_metadata_present{false};
    bool reconstruction_metadata_present{false};
};

struct MovingDomainSummary {
    MeshRevisionId mesh_revision{};
    bool mesh_velocity_metadata_present{false};
    Real min_geometric_jacobian{};
    Real max_geometric_jacobian{};
    Real geometric_conservation_residual{};
    Real geometric_conservation_tolerance{};
};

struct TransferOperatorSummary {
    std::string interface_pair_id;
    std::string projection_space_id;
    Real conservation_residual{};
    Real constant_preservation_residual{};
    bool rank_metadata_present{false};
};

struct AdjointConsistencySummary {
    std::string contribution_id;
    AdjointConsistencyKind adjoint_consistency{AdjointConsistencyKind::Unknown};
    bool transpose_backend_support{false};
    std::string goal_functional_id;
};

struct ParameterScaleSummary {
    std::string nondimensional_parameter_id;
    Real min_scale_value{};
    Real max_scale_value{};
    Real layer_resolution_metric{};
    Real frequency_resolution_metric{};
};

struct InitialCompatibilitySummary {
    Real initial_constraint_residual{};
    Real initial_boundary_residual{};
    std::uint64_t invariant_domain_initial_violation_count{0};
    Real residual_tolerance{};
};

struct CompatibleComplexSummary {
    std::string complex_id;
    std::vector<VariableKey> variables;
    bool exact_sequence_compatible{false};
    bool commuting_projection_available{false};
    bool trace_sequence_compatible{false};
    std::uint64_t missing_space_count{0};
};

struct NonlinearTangentSummary {
    std::string residual_id;
    OperatorBlockId block;
    TangentConsistencyClass tangent_consistency{TangentConsistencyClass::Unknown};
    SymmetryClass tangent_symmetry{SymmetryClass::Unknown};
    PositivityClass tangent_positivity{PositivityClass::Unknown};
    Real finite_difference_action_error{};
    Real finite_difference_tolerance{};
    std::uint64_t newton_stagnation_count{0};
    bool jacobian_action_available{false};
};

struct SpectralStructureSummary {
    OperatorBlockId block;
    bool eigenproblem_declared{false};
    bool self_adjoint_evidence{false};
    bool compactness_evidence{false};
    std::uint64_t spurious_mode_count{0};
    Real spectral_tolerance{};
    Real rayleigh_quotient_lower_bound{};
    NullspaceHandlingClass nullspace_handling{NullspaceHandlingClass::Unknown};
};

struct ErrorEstimatorSummary {
    std::string estimator_id;
    OperatorBlockId block;
    bool residual_metadata_present{false};
    bool jump_metadata_present{false};
    bool flux_reconstruction_present{false};
    bool adjoint_weighting_available{false};
    std::uint64_t missing_required_metadata_count{0};
    Real effectivity_lower_bound{};
    Real effectivity_upper_bound{};
};

struct QuadratureAdequacySummary {
    OperatorBlockId block;
    int integrand_polynomial_degree{-1};
    int quadrature_exact_degree{-1};
    bool nonlinear_aliasing_control_present{false};
    bool reduced_integration_declared{false};
    bool hourglass_control_present{false};
    std::uint64_t underintegrated_entry_count{0};
    std::uint64_t zero_energy_mode_count{0};
    Real aliasing_indicator{};
    Real aliasing_tolerance{};
};

struct CoupledSystemStabilitySummary {
    std::string coupling_group;
    std::vector<VariableKey> variables;
    bool monolithic_coupling{false};
    bool partitioned_coupling{false};
    Real exchange_residual{};
    Real coupling_tolerance{};
    Real partition_iteration_spectral_radius{};
    Real constraint_drift_norm{};
    std::uint64_t unstable_exchange_count{0};
};

struct AnalysisSummarySet {
    std::vector<CoefficientPropertySummary> coefficient_properties;
    std::vector<DiscreteMatrixSummary> discrete_matrices;
    std::vector<ReducedMatrixSummary> reduced_matrices;
    std::vector<LocalStencilSummary> local_stencils;
    std::vector<MeshGeometryQualitySummary> mesh_geometry_quality;
    std::vector<FluxBalanceSummary> flux_balances;
    std::vector<TemporalStabilitySummary> temporal_stability;
    std::vector<BoundarySymbolSummary> boundary_symbols;
    std::vector<InfSupEstimateSummary> inf_sup_estimates;
    std::vector<EnergyEntropySummary> energy_entropy;
    std::vector<InvariantDomainSummary> invariant_domains;
    std::vector<EquilibriumPreservationSummary> equilibrium_preservation;
    std::vector<MovingDomainSummary> moving_domain;
    std::vector<TransferOperatorSummary> transfer_operators;
    std::vector<AdjointConsistencySummary> adjoint_consistency;
    std::vector<ParameterScaleSummary> parameter_scales;
    std::vector<InitialCompatibilitySummary> initial_compatibility;
    std::vector<CompatibleComplexSummary> compatible_complexes;
    std::vector<NonlinearTangentSummary> nonlinear_tangents;
    std::vector<SpectralStructureSummary> spectral_structures;
    std::vector<ErrorEstimatorSummary> error_estimators;
    std::vector<QuadratureAdequacySummary> quadrature_adequacy;
    std::vector<CoupledSystemStabilitySummary> coupled_system_stability;

    [[nodiscard]] bool empty() const noexcept {
        return totalSummaryCount() == 0u;
    }

    [[nodiscard]] std::size_t totalSummaryCount() const noexcept {
        return coefficient_properties.size()
             + discrete_matrices.size()
             + reduced_matrices.size()
             + local_stencils.size()
             + mesh_geometry_quality.size()
             + flux_balances.size()
             + temporal_stability.size()
             + boundary_symbols.size()
             + inf_sup_estimates.size()
             + energy_entropy.size()
             + invariant_domains.size()
             + equilibrium_preservation.size()
             + moving_domain.size()
             + transfer_operators.size()
             + adjoint_consistency.size()
             + parameter_scales.size()
             + initial_compatibility.size()
             + compatible_complexes.size()
             + nonlinear_tangents.size()
             + spectral_structures.size()
             + error_estimators.size()
             + quadrature_adequacy.size()
             + coupled_system_stability.size();
    }

    [[nodiscard]] bool has(AnalysisSummaryKind kind) const noexcept {
        switch (kind) {
            case AnalysisSummaryKind::CoefficientProperties: return !coefficient_properties.empty();
            case AnalysisSummaryKind::DiscreteMatrix: return !discrete_matrices.empty();
            case AnalysisSummaryKind::ReducedMatrix: return !reduced_matrices.empty();
            case AnalysisSummaryKind::LocalStencil: return !local_stencils.empty();
            case AnalysisSummaryKind::MeshGeometryQuality: return !mesh_geometry_quality.empty();
            case AnalysisSummaryKind::FluxBalance: return !flux_balances.empty();
            case AnalysisSummaryKind::TemporalStability: return !temporal_stability.empty();
            case AnalysisSummaryKind::BoundarySymbol: return !boundary_symbols.empty();
            case AnalysisSummaryKind::InfSupEstimate: return !inf_sup_estimates.empty();
            case AnalysisSummaryKind::EnergyEntropyBalance: return !energy_entropy.empty();
            case AnalysisSummaryKind::InvariantDomain: return !invariant_domains.empty();
            case AnalysisSummaryKind::EquilibriumPreservation: return !equilibrium_preservation.empty();
            case AnalysisSummaryKind::MovingDomain: return !moving_domain.empty();
            case AnalysisSummaryKind::TransferOperator: return !transfer_operators.empty();
            case AnalysisSummaryKind::AdjointConsistency: return !adjoint_consistency.empty();
            case AnalysisSummaryKind::ParameterScale: return !parameter_scales.empty();
            case AnalysisSummaryKind::InitialCompatibility: return !initial_compatibility.empty();
            case AnalysisSummaryKind::CompatibleComplex: return !compatible_complexes.empty();
            case AnalysisSummaryKind::NonlinearTangent: return !nonlinear_tangents.empty();
            case AnalysisSummaryKind::SpectralStructure: return !spectral_structures.empty();
            case AnalysisSummaryKind::ErrorEstimator: return !error_estimators.empty();
            case AnalysisSummaryKind::QuadratureAdequacy: return !quadrature_adequacy.empty();
            case AnalysisSummaryKind::CoupledSystemStability: return !coupled_system_stability.empty();
        }
        return false;
    }

    [[nodiscard]] CertificationClass certificationOrUnknown(
        AnalysisSummaryKind kind,
        CertificationClass when_present) const noexcept
    {
        return has(kind) ? when_present : CertificationClass::Unknown;
    }
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_ANALYSIS_SUMMARY_TYPES_H
