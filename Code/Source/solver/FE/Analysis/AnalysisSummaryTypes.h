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
    InfSupPairCertification,
    StabilizationAdequacy,
    DAEStructureEvidence,
    SchurComplement,
    MinimumResidualStability,
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
        case AnalysisSummaryKind::InfSupPairCertification: return "InfSupPairCertification";
        case AnalysisSummaryKind::StabilizationAdequacy: return "StabilizationAdequacy";
        case AnalysisSummaryKind::DAEStructureEvidence: return "DAEStructureEvidence";
        case AnalysisSummaryKind::SchurComplement: return "SchurComplement";
        case AnalysisSummaryKind::MinimumResidualStability: return "MinimumResidualStability";
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

enum class DAEFormClass : std::uint8_t {
    SemiExplicit,
    FullyImplicit,
    DescriptorPencil,
    Unknown,
};

enum class MinimumResidualMethodClass : std::uint8_t {
    LeastSquares,
    PetrovGalerkin,
    DPG,
    Unknown,
};

enum class ParameterScaleRole : std::uint8_t {
    PecletLike,
    CflLike,
    WeakBoundaryPenalty,
    FrequencyResolution,
    LayerResolution,
    Generic,
    Unknown,
};

struct OperatorBlockId {
    std::vector<VariableKey> test_variables;
    std::vector<VariableKey> trial_variables;
    DomainKind domain{DomainKind::Cell};
    ContributionRole role{ContributionRole::DiagonalBlock};
    std::string contribution_id;
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
    OperatorBlockId block;
    std::vector<VariableKey> variables;
    std::string contribution_id;
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
    bool robustness_certificate_present{false};
    std::string robustness_certificate_scope;
    std::string coverage_scope;
    std::string producer_certificate_id;
    bool coefficient_region_coverage_complete{false};
    bool quadrature_point_coverage_complete{false};
    bool state_sample_coverage_complete{false};
    bool lower_bound_valid_for_all_samples{false};
    bool tolerance_metadata_present{false};
};

struct DiscreteMatrixSummary {
    OperatorBlockId block;
    std::vector<std::string> contribution_ids;
    std::vector<std::string> contribution_tags;
    bool contribution_provenance_complete{false};
    NumericSummaryScope scope{NumericSummaryScope::FullMatrix};
    std::optional<backends::BackendKind> backend_kind;
    GlobalIndex rows{0};
    GlobalIndex cols{0};

    bool square{false};
    bool structurally_symmetric{false};
    bool numerically_symmetric{false};
    bool symmetry_evidence_complete{false};
    bool sign_evidence_complete{false};
    bool row_sum_evidence_complete{false};

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
    std::optional<Real> min_eigenvalue_estimate;
    std::optional<Real> coercivity_lower_bound;
    std::optional<Real> nonnormality_indicator;
    std::optional<Real> nonsymmetry_indicator;

    std::uint64_t diagonal_count{0};
    std::uint64_t nonpositive_diagonal_count{0};
    std::uint64_t negative_diagonal_count{0};
    std::uint64_t near_zero_diagonal_count{0};
    std::uint64_t offdiag_count{0};
    std::uint64_t positive_offdiag_count{0};
    std::uint64_t negative_offdiag_count{0};
    std::uint64_t near_zero_offdiag_count{0};
    std::uint64_t row_sum_violation_count{0};
    std::uint64_t scanned_row_count{0};
    std::uint64_t expected_row_count{0};
    std::uint64_t scanned_entry_count{0};

    bool cholesky_factorization_succeeded{false};
    bool ldlt_factorization_nonnegative{false};
    bool m_matrix_certification_evidence{false};
    bool dmp_applicability_evidence{false};
    bool dmp_rhs_sign_evidence{false};
    bool inverse_positivity_evidence{false};
    bool irreducible_diagonal_dominance_evidence{false};
    bool stieltjes_matrix_evidence{false};
    std::string m_matrix_theorem_id;

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

struct SchurComplementSummary {
    std::string schur_id;
    OperatorBlockId block;
    std::vector<VariableKey> variables;
    bool schur_available{false};
    bool reduction_exact_for_analysis{false};
    bool primal_block_invertible_evidence_present{false};
    bool inf_sup_evidence_present{false};
    bool nullspace_handling_evidence_present{false};
    NullspaceHandlingClass nullspace_handling{NullspaceHandlingClass::Unknown};
    bool schur_definiteness_evidence_present{false};
    PositivityClass schur_positivity{PositivityClass::Unknown};
    bool spectral_equivalence_bounds_present{false};
    Real spectral_equivalence_lower_bound{};
    Real spectral_equivalence_upper_bound{};
    bool preconditioner_equivalence_bounds_present{false};
    Real preconditioner_equivalence_lower_bound{};
    Real preconditioner_equivalence_upper_bound{};
    bool condition_estimate_present{false};
    Real condition_estimate{};
    bool inexact_solve_tolerance_present{false};
    Real inexact_solve_tolerance{};
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
    Real aspect_ratio_warning_threshold{};
    Real max_anisotropy_alignment_indicator{};
    Real shape_regular_constant{};
    Real min_cut_cell_fraction{};
    Real max_cut_cell_fraction{};
    std::uint64_t inverted_element_count{0};
    std::uint64_t poor_quality_element_count{0};
    std::uint64_t cut_cell_count{0};
    bool shape_regular_evidence_present{false};
    std::size_t worst_element_sample_limit{kDefaultWorstSampleLimit};
    std::vector<ElementId> worst_elements;
};

struct FluxBalanceSummary {
    OperatorBlockId block;
    std::string balance_group;
    Real balance_tolerance{};
    Real local_residual_norm{};
    Real global_residual_norm{};
    Real interface_pair_residual_norm{};
    std::uint64_t local_violation_count{0};
    std::uint64_t interface_pair_count{0};
    bool symbolic_balance_evidence_present{false};
    std::string symbolic_balance_group;
    std::string symbolic_balance_contribution_id;
    bool flux_variable_metadata_present{false};
    bool element_residual_evidence_present{false};
    bool face_pair_residual_evidence_present{false};
    bool source_quadrature_consistency_present{false};
    bool orientation_consistency_present{false};
    bool boundary_flux_accounted_for{false};
    bool time_update_balance_present{false};
};

struct TemporalStabilitySummary {
    std::string time_scheme;
    OperatorBlockId block;
    std::vector<VariableKey> variables;
    std::string contribution_id;
    std::string stability_theorem_id;
    TemporalStabilityClass stability_class{TemporalStabilityClass::Unknown};
    Real cfl_estimate{};
    Real cfl_margin{};
    Real eigenvalue_scale_estimate{};
    Real amplification_radius{};
    Real high_frequency_dissipation{};
    Real nonnormal_growth_bound{};
    bool cfl_estimate_present{false};
    bool cfl_derivation_metadata_present{false};
    bool cfl_margin_present{false};
    bool amplification_radius_present{false};
    bool stability_metadata_present{false};
    bool stability_region_evidence_present{false};
    bool operator_spectrum_coverage_present{false};
    bool numerical_range_coverage_present{false};
    bool scalar_modal_bound_only{false};
    bool operator_normality_evidence_present{false};
    bool contractivity_norm_metadata_present{false};
    bool energy_norm_contractivity_evidence_present{false};
    bool logarithmic_norm_bound_present{false};
    bool pseudospectral_bound_present{false};
    bool nonnormal_growth_bound_present{false};
    bool nonnormal_growth_bound_finite{false};
    bool ssp_or_tvd_evidence_present{false};
    bool invariant_domain_evidence_present{false};
    bool nonlinear_stability_evidence_present{false};
};

struct BoundarySymbolSummary {
    OperatorBlockId block;
    int principal_operator_order{0};
    int boundary_operator_order{0};
    TraceCapabilityFlags trace_coverage{TraceCapabilityFlags::None};
    std::optional<bool> complementing_condition_satisfied;
    std::string evidence_scope;
    std::string complementing_theorem_id;
    Real complementing_margin{};
    std::uint64_t boundary_condition_count{0};
    std::uint64_t required_boundary_condition_count{0};
    std::uint64_t missing_symbol_count{0};
    std::uint64_t root_subspace_mismatch_count{0};
    bool principal_symbol_rank_evidence_present{false};
    bool boundary_symbol_rank_evidence_present{false};
    bool tangential_frequency_coverage_present{false};
    bool decaying_root_count_evidence_present{false};
    bool stable_subspace_dimension_evidence_present{false};
    bool parameter_ellipticity_evidence_present{false};
    bool complementing_margin_present{false};
    bool component_coverage_complete{false};
    bool dof_coverage_complete{false};
    bool mixed_corner_edge_coverage_present{false};
};

struct InfSupEstimateSummary {
    OperatorBlockId block;
    VariableKey primal_variable;
    VariableKey multiplier_variable;
    Real estimate_value{};
    Real estimate_tolerance{};
    Real uniform_lower_bound{};
    GlobalIndex test_rows{0};
    GlobalIndex test_cols{0};
    std::string estimate_scope;
    std::string inf_sup_theorem_id;
    NullspaceHandlingClass nullspace_handling{NullspaceHandlingClass::Unknown};
    bool estimator_metadata_present{false};
    bool norm_metadata_present{false};
    bool mesh_refinement_evidence_present{false};
    std::uint64_t mesh_refinement_sample_count{0};
    bool uniform_lower_bound_evidence_present{false};
    bool uniform_lower_bound_value_present{false};
    bool mesh_family_scope_present{false};
    bool boundary_condition_scope_present{false};
};

struct InfSupPairCertificationSummary {
    OperatorBlockId block;
    VariableKey primal_variable;
    VariableKey multiplier_variable;
    std::string pair_family;
    std::string inf_sup_theorem_id;
    Real beta_lower_bound{};
    Real fortin_operator_norm_bound{};
    int primal_polynomial_order{-1};
    int multiplier_polynomial_order{-1};
    SpaceFamily primal_space_family{SpaceFamily::Unknown};
    SpaceFamily multiplier_space_family{SpaceFamily::Unknown};
    bool known_stable_pair{false};
    bool fortin_operator_evidence_present{false};
    bool mesh_assumption_evidence_present{false};
    bool domain_assumption_evidence_present{false};
    bool boundary_condition_scope_present{false};
    bool beta_lower_bound_present{false};
    bool fortin_operator_norm_bound_present{false};
};

struct EnergyEntropySummary {
    std::string energy_entropy_id;
    std::string energy_functional_id;
    std::string energy_norm_id;
    std::string energy_entropy_theorem_id;
    EnergyEntropyLawKind law_kind{EnergyEntropyLawKind::Generic};
    BalanceSignClass expected_production_sign{BalanceSignClass::Unknown};
    Real balance_tolerance{};
    Real observed_discrete_balance{};
    Real observed_production{};
    std::uint64_t violation_count{0};
    bool energy_functional_metadata_present{false};
    bool energy_norm_metadata_present{false};
    bool energy_positivity_evidence_present{false};
    bool energy_coercivity_evidence_present{false};
    bool discrete_dissipation_identity_evidence_present{false};
    bool boundary_source_energy_accounting_present{false};
    bool convex_entropy_metadata_present{false};
    bool entropy_variables_metadata_present{false};
    bool entropy_flux_metadata_present{false};
    bool entropy_dissipation_metadata_present{false};
    bool boundary_source_entropy_metadata_present{false};
};

struct InvariantDomainSummary {
    std::string invariant_set_id;
    std::vector<VariableKey> variables;
    Real lower_bound{};
    Real upper_bound{};
    bool lower_bound_active{false};
    bool upper_bound_active{false};
    bool limiter_evidence_present{false};
    bool cfl_condition_satisfied{false};
    bool ssp_time_discretization_evidence_present{false};
    bool source_admissibility_evidence_present{false};
    bool low_order_invariant_domain_evidence_present{false};
    bool convex_limiting_evidence_present{false};
    bool spatial_monotonicity_evidence_present{false};
    bool mass_positivity_evidence_present{false};
    std::uint64_t post_step_violation_count{0};
    std::string invariant_domain_theorem_id;
};

struct EquilibriumPreservationSummary {
    std::string equilibrium_id;
    Real flux_source_residual{};
    Real residual_tolerance{};
    bool source_quadrature_metadata_present{false};
    bool reconstruction_metadata_present{false};
    bool boundary_compatibility_metadata_present{false};
};

struct MovingDomainSummary {
    MeshRevisionId mesh_revision{};
    bool mesh_velocity_metadata_present{false};
    bool time_integration_metadata_present{false};
    bool remap_metadata_present{false};
    Real min_geometric_jacobian{};
    Real max_geometric_jacobian{};
    Real geometric_conservation_residual{};
    Real geometric_conservation_tolerance{};
};

struct TransferOperatorSummary {
    std::string interface_pair_id;
    std::string projection_space_id;
    Real residual_tolerance{};
    Real conservation_residual{};
    Real constant_preservation_residual{};
    bool rank_metadata_present{false};
    bool interface_scope_metadata_present{false};
    bool projection_consistency_metadata_present{false};
    bool mortar_inf_sup_or_dual_consistency_metadata_present{false};
    bool interface_mass_conditioning_metadata_present{false};
    bool action_reaction_flux_metadata_present{false};
};

struct AdjointConsistencySummary {
    std::string contribution_id;
    AdjointConsistencyKind adjoint_consistency{AdjointConsistencyKind::Unknown};
    bool transpose_backend_support{false};
    bool boundary_adjoint_metadata_present{false};
    bool stabilization_adjoint_metadata_present{false};
    bool goal_linearization_metadata_present{false};
    std::string goal_functional_id;
    bool discrete_adjoint_residual_present{false};
    Real discrete_adjoint_residual{};
    Real discrete_adjoint_tolerance{};
};

struct ParameterScaleSummary {
    std::string nondimensional_parameter_id;
    ParameterScaleRole role{ParameterScaleRole::Unknown};
    OperatorBlockId block;
    std::vector<VariableKey> variables;
    std::string contribution_id;
    Real min_scale_value{};
    Real max_scale_value{};
    Real required_lower_bound{};
    bool required_lower_bound_present{false};
    int polynomial_order{-1};
    bool trace_inverse_metadata_present{false};
    Real trace_inverse_constant{};
    Real mesh_quality_factor{1};
    Real coefficient_contrast_factor{1};
    Real layer_resolution_metric{};
    Real frequency_resolution_metric{};
};

struct StabilizationAdequacySummary {
    std::string stabilization_id;
    std::string method_family;
    OperatorBlockId block;
    std::vector<VariableKey> variables;
    bool parameter_formula_metadata_present{false};
    bool residual_consistency_evidence_present{false};
    bool regime_metadata_present{false};
    bool peclet_condition_satisfied{false};
    bool cfl_condition_satisfied{false};
    bool adjoint_consistency_evidence_present{false};
    std::uint64_t violation_count{0};
};

struct InitialCompatibilitySummary {
    Real initial_constraint_residual{};
    Real initial_boundary_residual{};
    std::uint64_t invariant_domain_initial_violation_count{0};
    Real residual_tolerance{};
};

struct DAEStructureEvidenceSummary {
    std::string system_id;
    std::string dae_index_theorem_id;
    std::vector<VariableKey> variables;
    DAEFormClass dae_form_class{DAEFormClass::Unknown};
    bool mass_matrix_rank_metadata_present{false};
    bool algebraic_jacobian_rank_metadata_present{false};
    bool algebraic_jacobian_full_rank{false};
    bool hidden_constraint_metadata_present{false};
    std::uint64_t hidden_constraint_count{0};
    bool consistent_initial_condition_evidence_present{false};
    bool descriptor_pencil_metadata_present{false};
    bool regular_descriptor_pencil_evidence_present{false};
    bool strangeness_index_metadata_present{false};
    int strangeness_index{-1};
    bool projector_index_metadata_present{false};
    bool projector_consistency_evidence_present{false};
    Real initial_constraint_residual{};
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
    bool jacobian_nonsingularity_evidence_present{false};
    bool lipschitz_or_smoothness_evidence_present{false};
    bool residual_decrease_evidence_present{false};
    bool line_search_or_trust_region_evidence_present{false};
    bool monotonicity_or_convexity_evidence_present{false};
};

struct SpectralStructureSummary {
    OperatorBlockId block;
    std::string spectral_convergence_theorem_id;
    bool eigenproblem_declared{false};
    bool self_adjoint_evidence{false};
    bool compactness_evidence{false};
    bool operator_convergence_evidence{false};
    bool discrete_compactness_evidence{false};
    bool compatible_complex_evidence{false};
    bool compatible_complex_spectral_theorem_evidence{false};
    bool gap_convergence_evidence{false};
    bool refinement_scope_metadata_present{false};
    std::uint64_t refinement_sample_count{0};
    std::uint64_t spurious_mode_count{0};
    Real spectral_tolerance{};
    Real rayleigh_quotient_lower_bound{};
    NullspaceHandlingClass nullspace_handling{NullspaceHandlingClass::Unknown};
};

struct ErrorEstimatorSummary {
    std::string estimator_id;
    std::string estimator_norm_id;
    std::string estimator_theorem_id;
    OperatorBlockId block;
    Real reliability_constant{};
    Real efficiency_constant{};
    bool residual_metadata_present{false};
    bool jump_metadata_present{false};
    bool flux_reconstruction_present{false};
    bool adjoint_weighting_available{false};
    bool norm_metadata_present{false};
    bool estimator_norm_scope_metadata_present{false};
    bool pde_operator_class_metadata_present{false};
    bool boundary_residual_metadata_present{false};
    bool data_oscillation_metadata_present{false};
    bool coefficient_source_regularity_metadata_present{false};
    bool shape_regular_mesh_evidence_present{false};
    bool reliability_constant_metadata_present{false};
    bool efficiency_constant_metadata_present{false};
    bool effectivity_bounds_present{false};
    bool refinement_evidence_present{false};
    bool goal_functional_metadata_present{false};
    bool adjoint_residual_metadata_present{false};
    std::uint64_t missing_required_metadata_count{0};
    std::uint64_t effectivity_sample_count{0};
    Real effectivity_lower_bound{};
    Real effectivity_upper_bound{};
};

struct MinimumResidualStabilitySummary {
    std::string method_id;
    std::string residual_norm_id;
    std::string test_norm_id;
    std::string minimum_residual_theorem_id;
    OperatorBlockId block;
    std::vector<VariableKey> variables;
    MinimumResidualMethodClass method_class{
        MinimumResidualMethodClass::Unknown};
    bool trial_space_metadata_present{false};
    bool test_space_metadata_present{false};
    bool distinct_test_trial_spaces{false};
    bool residual_norm_metadata_present{false};
    bool test_norm_metadata_present{false};
    bool method_scope_metadata_present{false};
    bool riesz_map_metadata_present{false};
    bool optimal_test_metadata_present{false};
    bool fortin_operator_evidence_present{false};
    bool enrichment_sufficiency_evidence_present{false};
    bool residual_control_constant_present{false};
    Real residual_control_constant{};
    bool local_trial_to_test_conditioning_present{false};
    Real local_trial_to_test_condition_estimate{};
    bool normal_equation_conditioning_present{false};
    Real normal_equation_condition_estimate{};
    std::uint64_t missing_required_metadata_count{0};
    std::uint64_t violation_count{0};
};

struct QuadratureAdequacySummary {
    OperatorBlockId block;
    int integrand_polynomial_degree{-1};
    int quadrature_exact_degree{-1};
    bool affine_mapping_evidence_present{false};
    bool polynomial_integrand_metadata_complete{false};
    bool coefficient_degree_metadata_present{false};
    bool curved_or_nonlinear_mapping{false};
    bool overintegration_metadata_present{false};
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
    bool exchange_residual_present{false};
    Real coupling_tolerance{};
    bool coupling_tolerance_present{false};
    Real partition_iteration_spectral_radius{};
    bool partition_iteration_spectral_radius_present{false};
    Real constraint_drift_norm{};
    bool constraint_drift_present{false};
    std::uint64_t unstable_exchange_count{0};
    bool linear_stationary_iteration_evidence_present{false};
    bool contraction_norm_evidence_present{false};
    bool interface_energy_balance_evidence_present{false};
    bool relaxation_metadata_present{false};
    bool added_mass_risk_assessed{false};
    bool nonnormal_coupling_bound_present{false};
    bool coupled_norm_coercivity_evidence_present{false};
    bool coupled_operator_stability_evidence_present{false};
};

struct AnalysisSummarySet {
    std::vector<CoefficientPropertySummary> coefficient_properties;
    std::vector<DiscreteMatrixSummary> discrete_matrices;
    std::vector<ReducedMatrixSummary> reduced_matrices;
    std::vector<SchurComplementSummary> schur_complements;
    std::vector<LocalStencilSummary> local_stencils;
    std::vector<MeshGeometryQualitySummary> mesh_geometry_quality;
    std::vector<FluxBalanceSummary> flux_balances;
    std::vector<TemporalStabilitySummary> temporal_stability;
    std::vector<BoundarySymbolSummary> boundary_symbols;
    std::vector<InfSupEstimateSummary> inf_sup_estimates;
    std::vector<InfSupPairCertificationSummary> inf_sup_pair_certifications;
    std::vector<EnergyEntropySummary> energy_entropy;
    std::vector<InvariantDomainSummary> invariant_domains;
    std::vector<EquilibriumPreservationSummary> equilibrium_preservation;
    std::vector<MovingDomainSummary> moving_domain;
    std::vector<TransferOperatorSummary> transfer_operators;
    std::vector<AdjointConsistencySummary> adjoint_consistency;
    std::vector<ParameterScaleSummary> parameter_scales;
    std::vector<StabilizationAdequacySummary> stabilization_adequacy;
    std::vector<InitialCompatibilitySummary> initial_compatibility;
    std::vector<DAEStructureEvidenceSummary> dae_structure_evidence;
    std::vector<CompatibleComplexSummary> compatible_complexes;
    std::vector<NonlinearTangentSummary> nonlinear_tangents;
    std::vector<SpectralStructureSummary> spectral_structures;
    std::vector<ErrorEstimatorSummary> error_estimators;
    std::vector<QuadratureAdequacySummary> quadrature_adequacy;
    std::vector<CoupledSystemStabilitySummary> coupled_system_stability;
    std::vector<MinimumResidualStabilitySummary> minimum_residual_stability;

    [[nodiscard]] bool empty() const noexcept {
        return totalSummaryCount() == 0u;
    }

    [[nodiscard]] std::size_t totalSummaryCount() const noexcept {
        return coefficient_properties.size()
             + discrete_matrices.size()
             + reduced_matrices.size()
             + schur_complements.size()
             + local_stencils.size()
             + mesh_geometry_quality.size()
             + flux_balances.size()
             + temporal_stability.size()
             + boundary_symbols.size()
             + inf_sup_estimates.size()
             + inf_sup_pair_certifications.size()
             + energy_entropy.size()
             + invariant_domains.size()
             + equilibrium_preservation.size()
             + moving_domain.size()
             + transfer_operators.size()
             + adjoint_consistency.size()
             + parameter_scales.size()
             + stabilization_adequacy.size()
             + initial_compatibility.size()
             + dae_structure_evidence.size()
             + compatible_complexes.size()
             + nonlinear_tangents.size()
             + spectral_structures.size()
             + error_estimators.size()
             + quadrature_adequacy.size()
             + coupled_system_stability.size()
             + minimum_residual_stability.size();
    }

    [[nodiscard]] bool has(AnalysisSummaryKind kind) const noexcept {
        switch (kind) {
            case AnalysisSummaryKind::CoefficientProperties: return !coefficient_properties.empty();
            case AnalysisSummaryKind::DiscreteMatrix: return !discrete_matrices.empty();
            case AnalysisSummaryKind::ReducedMatrix: return !reduced_matrices.empty();
            case AnalysisSummaryKind::SchurComplement: return !schur_complements.empty();
            case AnalysisSummaryKind::LocalStencil: return !local_stencils.empty();
            case AnalysisSummaryKind::MeshGeometryQuality: return !mesh_geometry_quality.empty();
            case AnalysisSummaryKind::FluxBalance: return !flux_balances.empty();
            case AnalysisSummaryKind::TemporalStability: return !temporal_stability.empty();
            case AnalysisSummaryKind::BoundarySymbol: return !boundary_symbols.empty();
            case AnalysisSummaryKind::InfSupEstimate: return !inf_sup_estimates.empty();
            case AnalysisSummaryKind::InfSupPairCertification: return !inf_sup_pair_certifications.empty();
            case AnalysisSummaryKind::EnergyEntropyBalance: return !energy_entropy.empty();
            case AnalysisSummaryKind::InvariantDomain: return !invariant_domains.empty();
            case AnalysisSummaryKind::EquilibriumPreservation: return !equilibrium_preservation.empty();
            case AnalysisSummaryKind::MovingDomain: return !moving_domain.empty();
            case AnalysisSummaryKind::TransferOperator: return !transfer_operators.empty();
            case AnalysisSummaryKind::AdjointConsistency: return !adjoint_consistency.empty();
            case AnalysisSummaryKind::ParameterScale: return !parameter_scales.empty();
            case AnalysisSummaryKind::StabilizationAdequacy: return !stabilization_adequacy.empty();
            case AnalysisSummaryKind::InitialCompatibility: return !initial_compatibility.empty();
            case AnalysisSummaryKind::DAEStructureEvidence: return !dae_structure_evidence.empty();
            case AnalysisSummaryKind::CompatibleComplex: return !compatible_complexes.empty();
            case AnalysisSummaryKind::NonlinearTangent: return !nonlinear_tangents.empty();
            case AnalysisSummaryKind::SpectralStructure: return !spectral_structures.empty();
            case AnalysisSummaryKind::ErrorEstimator: return !error_estimators.empty();
            case AnalysisSummaryKind::QuadratureAdequacy: return !quadrature_adequacy.empty();
            case AnalysisSummaryKind::CoupledSystemStability: return !coupled_system_stability.empty();
            case AnalysisSummaryKind::MinimumResidualStability:
                return !minimum_residual_stability.empty();
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
