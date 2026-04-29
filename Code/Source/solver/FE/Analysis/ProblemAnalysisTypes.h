/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_PROBLEM_ANALYSIS_TYPES_H
#define SVMP_FE_ANALYSIS_PROBLEM_ANALYSIS_TYPES_H

/**
 * @file ProblemAnalysisTypes.h
 * @brief Core type definitions for the FE problem analysis subsystem
 *
 * Provides the vocabulary types used throughout the Analysis module:
 *
 *  - PropertyKind: what mathematical property is being described
 *  - PropertyStatus: the state of that property (exact, likely, violated, ...)
 *  - AnalysisConfidence: how certain the analyzer is
 *  - PropertyClaim: a single assertion about the problem
 *  - PropertyEvidence: supporting data for a claim
 *  - AnalysisIssue: a warning or error surfaced by analysis
 *  - ProblemAnalysisReport: the aggregate output of all analyzer passes
 *
 * @see ProblemAnalyzer for the orchestrator that produces reports
 * @see ProblemAnalysisContext for the input metadata consumed by analyzers
 */

#include "Core/Types.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

// Forward declarations from ContributionDescriptor.h (used in PropertyClaim structured fields)
enum class NullspaceFamily : std::uint8_t;
enum class ContributionRole : std::uint8_t;
enum class OperatorTraitFlags : std::uint32_t;
enum class AnalysisSummaryKind : std::uint8_t;
struct AnalysisSummarySet;

// ============================================================================
// Enumerations
// ============================================================================

/**
 * @brief Kind of mathematical property being described
 */
enum class PropertyKind : std::uint8_t {
    Nullspace,               ///< Operator has a nontrivial nullspace (constant, rigid-body, ...)
    OverConstraint,          ///< More constraints than needed / conflicting constraints
    UnderConstraint,         ///< Nullspace present with insufficient BC anchoring
    MixedSaddlePoint,        ///< System has saddle-point (indefinite) block structure
    CompatibilityCondition,  ///< Solvability condition on RHS (e.g. ∫f=0 for pure Neumann)
    OperatorSymmetry,        ///< Bilinear form is (skew-)symmetric
    OperatorDefiniteness,    ///< Bilinear form is positive (semi-)definite / indefinite
    Stabilization,           ///< Stabilization mechanism detected (SUPG, PSPG, penalty, ...)
    TopologyScopedKernel,    ///< Nullspace anchored globally but not on a specific mesh region
    ConstraintRedundancy,    ///< Duplicate / redundant constraints that are compatible
    CoupledSystemStructure,  ///< FE↔auxiliary-state↔boundary-functional coupling topology
    InterfaceCondition,      ///< Variables coupled only through interface/global kernels

    // Phase 21 advanced claim kinds
    InfSupCondition,              ///< Inf-sup (LBB) stability condition
    ConservationStructure,        ///< Local/global conservation / balance closure
    DifferentialAlgebraicStructure, ///< DAE index / algebraic vs dynamic classification
    SpaceCompatibility,           ///< FE space pair compatibility (e.g., Taylor-Hood)
    OperatorTransportCharacter,   ///< First-order / convection-like / transport-dominated character

    // Discrete stability and roadmap-wide claim kinds
    DiscreteMaximumPrinciple,      ///< Discrete maximum principle applicability/certification
    ZMatrixStructure,              ///< Off-diagonal sign structure of a discrete operator
    MMatrixStructure,              ///< M-matrix monotonicity structure
    MatrixMonotonicityRisk,        ///< Matrix sign/monotonicity risk evidence
    CompatibleComplexStructure,    ///< de Rham/exact-sequence compatibility
    EnergyStability,               ///< Discrete energy law/stability structure
    EntropyStability,              ///< Discrete entropy law/stability structure
    TemporalStability,             ///< Time-integration stability structure
    WeakBoundaryCoercivity,        ///< DG/Nitsche/penalty boundary coercivity
    MeshGeometryValidity,          ///< Mesh mapping and geometry validity
    CoefficientPositivity,         ///< Scalar/tensor coefficient positivity
    NonlinearTangentStructure,     ///< Nonlinear residual/tangent consistency
    LockingRisk,                   ///< Constraint/parameter overstiffness or locking risk
    SpectralCorrectness,           ///< Eigenproblem compactness/spurious-mode structure
    ErrorEstimatorEligibility,     ///< A posteriori estimator applicability
    SolverCompatibility,           ///< Solver/preconditioner compatibility with claims
    QuadratureAdequacy,            ///< Quadrature exactness/underintegration structure
    BoundaryComplementingCondition, ///< Boundary-symbol complementing-condition evidence
    IndefiniteOperatorResolution,  ///< Helmholtz/shifted/indefinite operator resolution
    MinimumResidualStability,      ///< Petrov-Galerkin/least-squares stability structure
    InvariantDomainPreservation,   ///< Bound/positivity/invariant-domain preservation
    EquilibriumPreservation,       ///< Well-balanced/equilibrium preservation
    GeometricConservation,         ///< Moving-domain geometric conservation
    TransferOperatorCompatibility, ///< Projection/mortar/transfer compatibility
    AdjointConsistency,            ///< Adjoint-consistency of operators/interfaces
    ParameterRobustness,           ///< Singular-perturbation/parameter robustness
    InitialDataCompatibility,      ///< Initial data and constraint compatibility
};

/**
 * @brief Status of a property assertion
 */
enum class PropertyStatus : std::uint8_t {
    Exact,      ///< Proven by symbolic / structural analysis
    Likely,     ///< High probability but not proven (e.g. stabilization weakly breaks nullspace)
    Violated,   ///< Property is expected but violated (e.g. over-constrained)
    Preserved,  ///< Property is compatible / maintained (e.g. BC doesn't break symmetry)
    Unknown,    ///< Insufficient information
};

/**
 * @brief Confidence level for an analysis claim
 */
enum class AnalysisConfidence : std::uint8_t {
    High,    ///< Symbolic analysis is conclusive
    Medium,  ///< Likely correct but depends on heuristics
    Low,     ///< Heuristic guess; may be wrong
};

/**
 * @brief Severity of an analysis issue
 */
enum class IssueSeverity : std::uint8_t {
    Error,    ///< Problem will likely cause solver failure
    Warning,  ///< Problem may cause poor convergence or incorrect results
    Info,     ///< Informational note (e.g. "stabilization detected")
};

// ============================================================================
// Phase 21 — Advanced classification enums
// ============================================================================

/**
 * @brief Inf-sup (LBB) stability classification
 */
enum class InfSupClass : std::uint8_t {
    Required,                ///< Inf-sup condition must hold for well-posedness
    StructurallySupported,   ///< Space pair is known to satisfy inf-sup (e.g., Taylor-Hood)
    NumericallySupported,    ///< Scoped numerical estimate supports the pair
    StabilizedSurrogate,     ///< Inf-sup replaced by stabilization (e.g., PSPG)
    LikelyViolated,          ///< Space pair unlikely to satisfy inf-sup
    Unknown,
};

/**
 * @brief Conservation / balance closure classification
 */
enum class ConservationClass : std::uint8_t {
    LocalClosureExpected,    ///< Local conservation expected (e.g., HDiv flux)
    GlobalClosureExpected,   ///< Global balance expected (e.g., integral constraint)
    ExchangeBalanced,        ///< Exchange between subdomains is balanced
    ClosureBroken,           ///< Conservation violated (e.g., non-conservative stabilization)
    Unknown,
};

/**
 * @brief Differential-algebraic system structure
 */
enum class DAEClass : std::uint8_t {
    PureODELike,             ///< All variables are dynamic (have time derivatives)
    AlgebraicSystem,         ///< No time derivatives (steady-state or algebraic)
    Index1DAELike,           ///< Mixed dynamic + algebraic constraints (index-1)
    HigherIndexRisk,         ///< Potential higher-index DAE structure
    Unknown,
};

/**
 * @brief FE space pair compatibility
 */
enum class SpaceCompatibilityClass : std::uint8_t {
    Compatible,              ///< Space pair is known compatible (e.g., Taylor-Hood P2/P1)
    WeaklyCompatible,        ///< Compatible with stabilization or special treatment
    Incompatible,            ///< Space pair is known incompatible (e.g., P1/P1 unstabilized Stokes)
    Unknown,
};

/**
 * @brief Transport / convection character
 */
enum class TransportCharacterClass : std::uint8_t {
    None,                    ///< No first-order transport character
    DiffusionLike,           ///< Dominated by second-order (diffusion)
    DirectionalFirstOrderLike, ///< Has directional first-order terms (convection)
    NonNormalRisk,           ///< Non-normal operator risk (convection-dominated)
    TransportDominatedRisk,  ///< Transport-dominated regime (high Peclet)
    Unknown,
};

/**
 * @brief Applicability of an analyzer/property to the current mathematical structure
 */
enum class ApplicabilityClass : std::uint8_t {
    Applicable,
    NotApplicable,
    Unknown,
};

/**
 * @brief Certification state for a property when symbolic or numeric evidence exists
 */
enum class CertificationClass : std::uint8_t {
    Certified,
    Violated,
    NotCertified,
    Unknown,
};

/**
 * @brief Matrix sign/monotonicity structure classification
 */
enum class MatrixSignStructureClass : std::uint8_t {
    ZMatrix,
    NotZMatrix,
    MMatrixCertified,
    MMatrixNotCertified,
    Unknown,
};

/**
 * @brief Operator symmetry classification independent of raw trait flags
 */
enum class OperatorSymmetryClass : std::uint8_t {
    Symmetric,
    Skew,
    Nonsymmetric,
    Unknown,
};

/**
 * @brief Time integration stability classification
 */
enum class TemporalStabilityClass : std::uint8_t {
    AStable,
    LStable,
    BStable,
    SSP,
    ConditionallyStable,
    Unknown,
};

/**
 * @brief Coercivity classification for elliptic/energy-like operators
 */
enum class CoercivityClass : std::uint8_t {
    Coercive,
    Semicoercive,
    Indefinite,
    NotCoercive,
    Unknown,
};

/**
 * @brief How nullspace modes were treated by a diagnostic or estimate
 */
enum class NullspaceHandlingClass : std::uint8_t {
    NotApplicable,
    AnchoredByConstraints,
    ProjectedOut,
    Retained,
    Uncontrolled,
    Unknown,
};

/**
 * @brief Temporal state kind for variables
 */
enum class TemporalStateKind : std::uint8_t {
    Algebraic,               ///< Variable has no time derivative (constraint/static)
    Dynamic,                 ///< Variable has time derivative (evolving)
    Mixed,                   ///< Variable has both algebraic and dynamic contributions
    Unknown,
};

/**
 * @brief Function space family
 */
enum class SpaceFamily : std::uint8_t {
    H1,                      ///< Standard H1-conforming (nodal Lagrange, etc.)
    HDiv,                    ///< H(div)-conforming (Raviart-Thomas, BDM)
    HCurl,                   ///< H(curl)-conforming (Nedelec edge elements)
    L2,                      ///< L2 (discontinuous)
    Custom,                  ///< User-defined / non-standard
    Unknown,
};

/**
 * @brief Trace capability flags for a function space (bitmask)
 */
enum class TraceCapabilityFlags : std::uint32_t {
    None               = 0,
    Value              = 1u << 0,   ///< Pointwise value trace
    NormalComponent    = 1u << 1,   ///< Normal component trace (HDiv)
    TangentialComponent = 1u << 2,  ///< Tangential component trace (HCurl)
    NormalFlux         = 1u << 3,   ///< Normal flux (gradient · n)
    Jump               = 1u << 4,   ///< Jump across faces (DG)
    Average            = 1u << 5,   ///< Average across faces (DG)
};

inline constexpr TraceCapabilityFlags operator|(TraceCapabilityFlags a, TraceCapabilityFlags b) noexcept {
    return static_cast<TraceCapabilityFlags>(
        static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
inline constexpr TraceCapabilityFlags operator&(TraceCapabilityFlags a, TraceCapabilityFlags b) noexcept {
    return static_cast<TraceCapabilityFlags>(
        static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}
inline constexpr bool hasTraceFlag(TraceCapabilityFlags flags, TraceCapabilityFlags test) noexcept {
    return (static_cast<std::uint32_t>(flags) & static_cast<std::uint32_t>(test)) != 0;
}

// ============================================================================
// Variable / Field types
// ============================================================================

/**
 * @brief Kind of variable (unknown) in the coupled system
 */
enum class VariableKind : std::uint8_t {
    FieldComponent,      ///< FE field DOFs (identified by field_id + component)
    AuxiliaryState,      ///< Auxiliary state variable (identified by name)
    AuxiliaryInput,      ///< Generalized auxiliary input (identified by name)
    AuxiliaryOutput,     ///< Auxiliary model output (identified by name)
    BoundaryFunctional,  ///< Boundary integral quantity (identified by name, legacy)
    GlobalScalar,        ///< Global scalar unknown (identified by name)
};

/**
 * @brief Domain on which an operator or condition acts
 */
enum class DomainKind : std::uint8_t {
    Cell,              ///< Volumetric/cell integral
    Boundary,          ///< External boundary face
    InteriorFace,      ///< Interior face (DG)
    InterfaceFace,     ///< Interface between subdomains
    Global,            ///< Global operator (no mesh locality)
    CoupledBoundary,   ///< Boundary with coupled PDE-ODE model (legacy)
    AuxiliaryCoupling, ///< Generalized auxiliary coupling (field↔aux, aux↔aux)
};

// ============================================================================
// Variable Identification
// ============================================================================

/**
 * @brief Stable identity for a variable in the coupled system
 *
 * For FieldComponent: uses field_id + component.
 * For AuxiliaryState / BoundaryFunctional / GlobalScalar: uses a stable name.
 * Comparable and hashable for use in maps/sets.
 */
struct VariableKey {
    VariableKind kind{VariableKind::FieldComponent};
    FieldId field_id{INVALID_FIELD_ID};   ///< Valid for FieldComponent
    int component{-1};                    ///< -1 = all components (FieldComponent only)
    std::string name;                     ///< Stable name for non-field variables

    /// Construct a field-component key
    static VariableKey field(FieldId fid, int comp = -1) {
        VariableKey k;
        k.kind = VariableKind::FieldComponent;
        k.field_id = fid;
        k.component = comp;
        return k;
    }

    /// Construct a named non-field key
    static VariableKey named(VariableKind vk, std::string n) {
        VariableKey k;
        k.kind = vk;
        k.name = std::move(n);
        return k;
    }

    bool operator==(const VariableKey& o) const noexcept {
        if (kind != o.kind) return false;
        if (kind == VariableKind::FieldComponent)
            return field_id == o.field_id && component == o.component;
        return name == o.name;
    }

    bool operator!=(const VariableKey& o) const noexcept { return !(*this == o); }

    bool operator<(const VariableKey& o) const noexcept {
        if (kind != o.kind) return kind < o.kind;
        if (kind == VariableKind::FieldComponent) {
            if (field_id != o.field_id) return field_id < o.field_id;
            return component < o.component;
        }
        return name < o.name;
    }
};

/**
 * @brief Hash functor for VariableKey (enables use in unordered_map/set)
 */
struct VariableKeyHash {
    std::size_t operator()(const VariableKey& k) const noexcept {
        std::size_t h = std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(k.kind));
        if (k.kind == VariableKind::FieldComponent) {
            h ^= std::hash<FieldId>{}(k.field_id) * 2654435761u;
            h ^= std::hash<int>{}(k.component) * 40503u;
        } else {
            h ^= std::hash<std::string>{}(k.name) * 2654435761u;
        }
        return h;
    }
};

/**
 * @brief Extended descriptor for a variable (wraps VariableKey with metadata)
 */
struct VariableDescriptor {
    VariableKey key;
    std::string label;                           ///< Human-readable label
    FieldType field_type{FieldType::Scalar};      ///< Scalar/Vector/Tensor (meaningful for FieldComponent)
    int value_dimension{1};                       ///< 1=scalar, 2/3=vector, etc.
    int region{-1};                               ///< -1 = global

    // Phase 21 extensions
    TemporalStateKind temporal_state_kind{TemporalStateKind::Unknown};
    int max_time_derivative_order{0};
    bool participates_in_constraint_blocks{false};
    bool participates_in_mass_blocks{false};

    // Auxiliary-state metadata (meaningful when kind is AuxiliaryState/Input/Output)
    std::string auxiliary_scope{};       ///< "Global", "Boundary", "Node", "Cell", "QuadraturePoint", "Facet"
    std::string auxiliary_solve_mode{};  ///< "Partitioned" or "Monolithic"
    std::string auxiliary_region{};      ///< Deployment region identity (empty = whole domain)
};

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief A piece of evidence supporting a PropertyClaim
 */
struct PropertyEvidence {
    std::string source;                              ///< Origin of this evidence (e.g. "FormStructureAnalyzer")
    std::string description;                         ///< Human-readable explanation
    AnalysisConfidence confidence{AnalysisConfidence::High};
    int boundary_marker{-1};                         ///< -1 = not boundary-specific
};

/**
 * @brief A single assertion about a mathematical property of the problem
 */
struct PropertyClaim {
    PropertyKind kind{PropertyKind::Nullspace};
    PropertyStatus status{PropertyStatus::Unknown};
    AnalysisConfidence confidence{AnalysisConfidence::High};

    /// Primary field (INVALID_FIELD_ID = system-wide). Retained for convenience
    /// when the claim applies to a single FE field — equivalent to
    /// variables containing a single FieldComponent VariableKey.
    FieldId field{INVALID_FIELD_ID};
    int component{-1};                               ///< -1 = all components
    int region{-1};                                  ///< -1 = global, >=0 = specific connected component
    DomainKind domain{DomainKind::Cell};              ///< Domain the claim applies to

    /// Variables involved (generic — includes FE fields, aux states, boundary functionals, etc.)
    std::vector<VariableKey> variables;

    std::string description;                         ///< Human-readable summary
    std::vector<PropertyEvidence> evidence;           ///< Supporting data

    // ---- Structured fields (Phase 17) ----
    // These replace description-text parsing for downstream consumers.
    // Populated by analyzer passes that have the structured information available.

    /// Nullspace mode family (for Nullspace claims). Replaces parsing "rigid-body" etc. from description.
    std::optional<NullspaceFamily> nullspace_family;

    /// Constraint cause (for UnderConstraint/OverConstraint claims).
    std::optional<ContributionRole> constraint_cause;

    /// Symmetry class (for OperatorSymmetry claims).
    std::optional<OperatorTraitFlags> symmetry_class;

    /// Definiteness class (for OperatorDefiniteness claims).
    std::optional<OperatorTraitFlags> definiteness_class;

    /// Which analyzer pass produced this claim.
    std::string claim_origin;

    // ---- Phase 21 advanced classification fields ----

    std::optional<InfSupClass> inf_sup_class;
    std::optional<ConservationClass> conservation_class;
    std::optional<DAEClass> dae_class;
    std::optional<SpaceCompatibilityClass> space_compatibility_class;
    std::optional<TransportCharacterClass> transport_character_class;

    // ---- Phase 1 roadmap vocabulary plumbing ----

    std::optional<ApplicabilityClass> applicability_class;
    std::optional<CertificationClass> certification_class;
    std::optional<MatrixSignStructureClass> matrix_sign_structure_class;
    std::optional<OperatorSymmetryClass> operator_symmetry_class;
    std::optional<TemporalStabilityClass> temporal_stability_class;
    std::optional<CoercivityClass> coercivity_class;
    std::optional<CertificationClass> reduced_definiteness_class;
    std::optional<NullspaceHandlingClass> nullspace_handling_class;

    /// Optional numeric evidence slots. These are scalar summaries only; full
    /// summary objects belong to later metadata phases.
    std::optional<double> inf_sup_estimate;
    std::optional<double> peclet_number;
    std::optional<double> cfl_number;
    std::optional<double> nonnormality_indicator;
    std::optional<double> local_balance_residual;
    std::optional<double> global_balance_residual;
    std::optional<double> interface_balance_residual;
    std::optional<double> constraint_drift_norm;
    std::optional<double> penalty_scale;
    std::optional<double> weak_coercivity_lower_bound;
    std::optional<double> flux_balance_residual;

    /// Optional symbolic/metadata evidence slots for analyzer families that do
    /// not require numeric summary objects.
    std::optional<bool> exact_sequence_compatible;
    std::optional<bool> commuting_projection_available;
    std::optional<bool> boundary_complementing_condition_satisfied;
    std::optional<bool> initial_data_compatible;
    std::optional<bool> invariant_domain_metadata_present;
    std::optional<bool> well_balanced_metadata_present;

    /// Stable identifiers for richer evidence objects added in later phases.
    std::optional<std::string> tested_block_id;
    std::optional<std::string> estimate_scope;
    std::optional<std::string> coefficient_id;
    std::optional<std::string> equilibrium_id;
    std::optional<std::string> invariant_set_id;

    // NOTE: claim → contribution tracing is done indirectly through the
    // existing evidence and variables fields. Analyzer passes that need to
    // reference specific contributions should do so through VariableKey
    // matching against ContributionDescriptor::test_variables/trial_variables,
    // not through index-based coupling (which would be fragile across
    // context mutations).

    /// Convenience: add a piece of evidence
    void addEvidence(std::string src, std::string desc,
                     AnalysisConfidence conf = AnalysisConfidence::High,
                     int marker = -1) {
        evidence.push_back({std::move(src), std::move(desc), conf, marker});
    }
};

/**
 * @brief An issue (warning/error/info) surfaced during analysis
 */
struct AnalysisIssue {
    IssueSeverity severity{IssueSeverity::Warning};
    std::string message;
    std::vector<std::size_t> related_claim_indices;  ///< Indices into ProblemAnalysisReport::claims
};

/**
 * @brief Request for one compact numeric/discrete summary needed by symbolic claims
 */
struct AnalysisSummaryRequest {
    AnalysisSummaryKind summary_kind{};
    DomainKind domain{DomainKind::Cell};
    std::vector<VariableKey> variables;
    std::vector<std::size_t> source_claim_indices;
    std::vector<PropertyKind> source_claim_kinds;
    std::vector<std::string> source_analyzers;
    std::vector<std::string> reasons;
    std::string block_id;
    std::string contribution_id;
    std::string scope_id;
    std::string request_id;
    AnalysisConfidence confidence{AnalysisConfidence::Medium};
    bool already_available{false};
};

/**
 * @brief Plan of numeric/discrete summaries requested after symbolic analysis
 */
struct AnalysisRequestPlan {
    std::vector<AnalysisSummaryRequest> summary_requests;

    [[nodiscard]] bool empty() const noexcept { return summary_requests.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return summary_requests.size(); }
    [[nodiscard]] bool has(AnalysisSummaryKind kind) const noexcept;
    [[nodiscard]] std::vector<const AnalysisSummaryRequest*>
    requestsOfKind(AnalysisSummaryKind kind) const;
    [[nodiscard]] bool hasSourceAnalyzer(const std::string& analyzer) const noexcept;
};

// ============================================================================
// Report
// ============================================================================

/**
 * @brief Aggregate output of all analyzer passes
 *
 * Contains all PropertyClaims emitted by the registered passes and any
 * issues detected.  Provides print() and summary() for diagnostics.
 */
struct ProblemAnalysisReport {
    std::vector<PropertyClaim> claims;
    std::vector<AnalysisIssue> issues;
    AnalysisRequestPlan request_plan;

    // ---- Queries ----

    /// Count claims matching a given kind
    [[nodiscard]] std::size_t countByKind(PropertyKind kind) const noexcept;

    /// Count claims matching a given status
    [[nodiscard]] std::size_t countByStatus(PropertyStatus status) const noexcept;

    /// Count issues by severity
    [[nodiscard]] std::size_t countBySeverity(IssueSeverity severity) const noexcept;

    /// Find all claims for a specific field
    [[nodiscard]] std::vector<const PropertyClaim*> claimsForField(FieldId field) const;

    /// Find all claims involving a specific variable
    [[nodiscard]] std::vector<const PropertyClaim*> claimsForVariable(const VariableKey& var) const;

    /// Find all claims of a specific kind
    [[nodiscard]] std::vector<const PropertyClaim*> claimsOfKind(PropertyKind kind) const;

    /// True if any issue has Error severity
    [[nodiscard]] bool hasErrors() const noexcept;

    /// True if any issue has Warning severity
    [[nodiscard]] bool hasWarnings() const noexcept;

    // ---- Output ----

    /**
     * @brief Print a human-readable report grouped by PropertyKind
     *
     * Style follows SparsityAnalyzer::AnalysisReport::print().
     */
    void print(std::ostream& out) const;

    /**
     * @brief Print compact application-facing analysis decisions.
     *
     * The output is intentionally physics-agnostic and uses only analyzer
     * names, property kinds, generic variables/fields, statuses, request ids,
     * and short evidence text.
     */
    void printApplicationLog(std::ostream& out) const;

    /**
     * @brief Print trace-level analysis evidence.
     *
     * When optional numeric summaries are provided, this includes bounded
     * samples of worst matrix/stencil entries, row diagnostics, mesh elements,
     * and constraint-reduction evidence.
     */
    void printTraceLog(std::ostream& out,
                       const AnalysisSummarySet* summaries = nullptr) const;

    /**
     * @brief One-line summary of claim and issue counts
     *
     * Example: "7 claims (3 exact, 2 likely, 2 unknown), 1 issue (1 warning)"
     */
    [[nodiscard]] std::string summary() const;
};

// ============================================================================
// String Conversion
// ============================================================================

[[nodiscard]] const char* toString(PropertyKind k) noexcept;
[[nodiscard]] const char* toString(PropertyStatus s) noexcept;
[[nodiscard]] const char* toString(AnalysisConfidence c) noexcept;
[[nodiscard]] const char* toString(IssueSeverity s) noexcept;
[[nodiscard]] const char* toString(VariableKind k) noexcept;
[[nodiscard]] const char* toString(DomainKind k) noexcept;

// Phase 21 toString helpers
[[nodiscard]] const char* toString(InfSupClass c) noexcept;
[[nodiscard]] const char* toString(ConservationClass c) noexcept;
[[nodiscard]] const char* toString(DAEClass c) noexcept;
[[nodiscard]] const char* toString(SpaceCompatibilityClass c) noexcept;
[[nodiscard]] const char* toString(TransportCharacterClass c) noexcept;
[[nodiscard]] const char* toString(ApplicabilityClass c) noexcept;
[[nodiscard]] const char* toString(CertificationClass c) noexcept;
[[nodiscard]] const char* toString(MatrixSignStructureClass c) noexcept;
[[nodiscard]] const char* toString(OperatorSymmetryClass c) noexcept;
[[nodiscard]] const char* toString(TemporalStabilityClass c) noexcept;
[[nodiscard]] const char* toString(CoercivityClass c) noexcept;
[[nodiscard]] const char* toString(NullspaceHandlingClass c) noexcept;
[[nodiscard]] const char* toString(TemporalStateKind k) noexcept;
[[nodiscard]] const char* toString(SpaceFamily f) noexcept;

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_PROBLEM_ANALYSIS_TYPES_H
