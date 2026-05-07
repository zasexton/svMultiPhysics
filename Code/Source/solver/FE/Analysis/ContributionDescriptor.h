/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_CONTRIBUTION_DESCRIPTOR_H
#define SVMP_FE_ANALYSIS_CONTRIBUTION_DESCRIPTOR_H

/**
 * @file ContributionDescriptor.h
 * @brief Normalized operator IR for the analysis subsystem
 *
 * ContributionDescriptor is the unified representation that both FormExpr
 * formulations and handwritten kernels lower into.  All analyzer passes
 * consume contributions as their primary input.
 *
 * @see FormContributionLowerer (Phase 11) for FormExpr → ContributionDescriptor
 * @see AssemblyKernel::analysisContributions() (Phase 12) for kernel → ContributionDescriptor
 */

#include "Core/Types.h"
#include "Analysis/FormRuntimeMetadata.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {

namespace forms {
class FormExprNode;
} // namespace forms

namespace analysis {

// ============================================================================
// Enumerations
// ============================================================================

/**
 * @brief Role of an operator contribution in the block system
 */
enum class ContributionRole : std::uint8_t {
    DiagonalBlock,       ///< Self-coupling diagonal block (e.g., gradient inner product on a primal block)
    OffDiagonalBlock,    ///< Cross-coupling off-diagonal block (e.g., vector-divergence/scalar-multiplier coupling)
    ConstraintBlock,     ///< Lagrange multiplier / constraint coupling (e.g., p*div(v))
    StabilizationBlock,  ///< Stabilization term (e.g., PSPG, GLS, penalty)
    BoundaryConstraint,  ///< Boundary condition contribution (Dirichlet, Robin, Nitsche)
    InterfaceCoupling,   ///< Interface-local coupling not known to be a constraint
    SourceVector,        ///< Residual vector/source term with no trial dependency
    ExternalForcing,     ///< Externally supplied forcing contribution
    InitialCondition,    ///< Initial-condition contribution, not an operator block
    DiagnosticOnly,      ///< Analysis/diagnostic metadata, not assembled as an operator
    GlobalCoupling,      ///< Non-local coupling (coupled boundary, global kernel)
    FieldToAuxiliary,    ///< FE field → auxiliary block coupling
    AuxiliaryToField,    ///< Auxiliary block → FE field coupling
    AuxiliaryToAuxiliary,///< Auxiliary block → auxiliary block coupling
    AuxiliarySelf,       ///< Auxiliary block self-coupling (diagonal)
};

/**
 * @brief Bitmask flags for operator mathematical traits
 */
enum class OperatorTraitFlags : std::uint32_t {
    None                    = 0,
    SymmetricLike           = 1u << 0,   ///< Bilinear form is approximately symmetric
    SkewLike                = 1u << 1,   ///< Bilinear form has skew-symmetric character
    PositiveSemiDefiniteLike = 1u << 2,  ///< a(u,u) >= 0 for all u
    PositiveDefiniteLike    = 1u << 3,   ///< a(u,u) > 0 for all u != 0
    HasMass                 = 1u << 4,   ///< Contains mass-like (zeroth order) terms
    HasFirstOrder           = 1u << 5,   ///< Contains first-order (convection-like) terms
    HasSecondOrder          = 1u << 6,   ///< Contains second-order (diffusion-like) terms
    NullspacePreserving     = 1u << 7,   ///< Preserves nullspace modes (e.g., periodic BC)
    NullspaceLifting        = 1u << 8,   ///< Lifts/anchors nullspace modes (e.g., Dirichlet BC)
    SourceLike              = 1u << 9,   ///< Residual-vector forcing/source-like term
    BoundaryFluxLike        = 1u << 10,  ///< Boundary flux-like vector contribution
    ConstraintLike          = 1u << 11,  ///< Explicit constraint-like metadata
    StabilizationLike       = 1u << 12,  ///< Explicit stabilization-method metadata
    MeshScaleDependentHint  = 1u << 13,  ///< Mesh-scale dependence; not proof of stabilization
};

inline constexpr OperatorTraitFlags operator|(OperatorTraitFlags a, OperatorTraitFlags b) noexcept {
    return static_cast<OperatorTraitFlags>(
        static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
inline constexpr OperatorTraitFlags operator&(OperatorTraitFlags a, OperatorTraitFlags b) noexcept {
    return static_cast<OperatorTraitFlags>(
        static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}
inline constexpr bool hasFlag(OperatorTraitFlags flags, OperatorTraitFlags test) noexcept {
    return (static_cast<std::uint32_t>(flags) & static_cast<std::uint32_t>(test)) != 0;
}

/**
 * @brief Nullspace mode family for structured claims
 */
enum class NullspaceFamily : std::uint8_t {
    ScalarConstant,          ///< p → p + c
    ComponentwiseConstant,   ///< u_i → u_i + c_i
    VectorConstant,          ///< all vector components shifted by a constant vector
    RigidTranslation,        ///< translational part of a rigid-motion kernel
    RigidRotation,           ///< rotational part of a rigid-motion kernel
    RigidBody,               ///< translations plus rotations
    GaugeConstant,           ///< gauge shift in a scalar/potential-like field
    HarmonicField,           ///< harmonic representatives in a topology-dependent kernel
    GradientKernel,          ///< kernel generated by gradients/exact fields
    CurlKernel,              ///< kernel generated by curls
    DivergenceFreeKernel,    ///< solenoidal/divergence-free kernel family
    KernelOfSymGrad,         ///< ker(sym(grad)): translations + rotations
    UserDefined,             ///< Custom nullspace mode from kernel author
};

/**
 * @brief Scope of an interface-face contribution
 */
enum class InterfaceScope : std::uint8_t {
    SpecificMarker,          ///< Targets a specific InterfaceMesh(marker)
    AllRegisteredInterfaces,  ///< Wildcard: applies to all registered interfaces
};

// ============================================================================
// Phase 22 — Extended metadata enums
// ============================================================================

/**
 * @brief How a contribution affects nullspace modes
 */
enum class NullspaceEffect : std::uint8_t {
    Preserves,               ///< Preserves nullspace (e.g., periodic BC)
    WeaklyLifts,             ///< Weakly breaks/lifts nullspace (e.g., Robin, Nitsche)
    ExactlyRemoves,          ///< Exactly removes nullspace (e.g., strong Dirichlet)
    Unknown,
};

/**
 * @brief Consistency of a discrete contribution relative to the continuum operator
 */
enum class ConsistencyKind : std::uint8_t {
    ExactContinuum,          ///< Exact representation of the continuum operator
    ConsistentPerturbation,  ///< Consistent stabilization (e.g., SUPG, Nitsche)
    InconsistentPerturbation, ///< Inconsistent stabilization (e.g., artificial viscosity)
    Unknown,
};

/**
 * @brief Adjoint consistency of a bilinear form contribution
 */
enum class AdjointConsistencyKind : std::uint8_t {
    Yes,                     ///< Adjoint consistent (symmetric Nitsche, SUPG with adjoint)
    No,                      ///< Not adjoint consistent (non-symmetric Nitsche)
    Unknown,
};

/**
 * @brief Temporal character of a contribution
 */
enum class TemporalContributionKind : std::uint8_t {
    Unknown,
    TimeIndependentResidual, ///< Stiffness/reaction/source term in a residual
    MassLike,                ///< Mass matrix (d/dt u · v)
    DampedMassLike,          ///< Damped mass (α d/dt u · v with α time-dependent)
    PureAlgebraicConstraint, ///< g(u,z,t)=0 algebraic constraint block
    LagrangeMultiplierConstraint, ///< Multiplier-enforced algebraic constraint
    PreviousTimeState,       ///< History/previous-time-state contribution
    TimeIntegratorResidual,  ///< Fully discrete time-integrator residual
    None,                    ///< Deprecated: no time derivative; not algebraic evidence
    PureConstraint,          ///< Deprecated: algebraic constraint
};

enum class TimeIntegrationRole : std::uint8_t {
    Unknown,
    CurrentState,
    DifferentiatedState,
    HistoryState,
    TimeStepParameter,
    FullyDiscreteResidual,
};

/**
 * @brief Role of a contribution in a conservation balance
 */
enum class BalanceRole : std::uint8_t {
    Accumulation,            ///< Time-derivative / storage term
    FluxLike,                ///< Divergence-form flux
    SourceLike,              ///< Volumetric source
    SinkLike,                ///< Volumetric sink
    ExchangeLike,            ///< Inter-domain exchange (interface, coupled boundary)
    Unknown,
};

/**
 * @brief Kind of variable pairing in a block system
 */
enum class PairingKind : std::uint8_t {
    FormalAdjointPair,       ///< Primal-dual pair with adjoint structure (grad/div)
    ConstraintPair,          ///< Lagrange multiplier constraint
    StabilizedConstraintPair, ///< Constraint with stabilization surrogate
    Unknown,
};

/**
 * @brief Transport / convection character of a contribution
 */
enum class TransportCharacter : std::uint8_t {
    None,                    ///< No transport character
    DirectionalFirstOrder,   ///< Directional first-order (convection)
    DiffusionLike,           ///< Diffusion-dominated
    NonNormalLike,           ///< Non-normal operator character
    TransportDominatedRisk,  ///< Transport-dominated regime risk
};

// ============================================================================
// Phase 22 — Extended metadata structs
// ============================================================================

/**
 * @brief Scaling structure of a contribution (h-power, dt-power)
 */
struct ScalingDescriptor {
    int h_power{0};              ///< Power of mesh size h (e.g., -1 for penalty/h)
    int dt_power{0};             ///< Power of time step dt
    bool parameter_scaled{false}; ///< Scaled by a user parameter
    bool coefficient_scaled{false}; ///< Scaled by a material coefficient
};

struct ContributionDomainScope {
    DomainKind domain{DomainKind::Cell};
    int marker{-1};
    std::string subexpression_id;
};

/**
 * @brief Temporal structure of a contribution
 */
struct TemporalDescriptor {
    int derivative_order{0};     ///< 0=steady, 1=first-order in time, 2=second-order
    TemporalContributionKind kind{TemporalContributionKind::Unknown};
    std::vector<VariableKey> differentiated_variables;
    std::vector<VariableKey> history_variables;
    TimeIntegrationRole time_role{TimeIntegrationRole::Unknown};
    std::string timestep_scope_id;

    TemporalDescriptor() = default;
    TemporalDescriptor(int order, TemporalContributionKind temporal_kind)
        : derivative_order(order), kind(temporal_kind)
    {}
};

/**
 * @brief Conservation / balance role of a contribution
 */
struct BalanceDescriptor {
    std::string balance_group;   ///< Balance group name (not physics-labeled)
    BalanceRole role{BalanceRole::Unknown};
    int sign{1};                 ///< +1 = positive contribution, -1 = negative
    bool local_closure_expected{false};
};

/**
 * @brief Variable pairing descriptor for block systems
 */
struct PairingDescriptor {
    VariableKey row_var;
    VariableKey col_var;
    PairingKind kind{PairingKind::Unknown};
    std::string pairing_group;   ///< Group name (not physics-labeled)
    bool has_stabilizing_surrogate{false};

    /// True when the trial (col_var) field appears undifferentiated in the
    /// coupling block.  For mixed blocks where the trial appears both with
    /// and without differential operators (e.g., NS-VMS VP block has both
    /// `p div(v)` and `τ_m grad(v) · grad(p)`), this flag is true even
    /// though the pairing kind is FormalAdjointPair.  Used by the IBP
    /// coupling analysis to detect cross-field nullspace anchoring.
    bool trial_has_undifferentiated{false};
};

// ============================================================================
// Core Data Structures
// ============================================================================

/**
 * @brief Structured nullspace hint from a contribution
 */
struct NullspaceHint {
    NullspaceFamily family{NullspaceFamily::ScalarConstant};
    FieldId field{INVALID_FIELD_ID};
    int component{-1};
    AnalysisConfidence confidence{AnalysisConfidence::High};
    NullspaceEvidenceKind evidence_kind{NullspaceEvidenceKind::DescriptorHint};
    std::string reason;
};

/**
 * @brief Normalized operator contribution descriptor
 *
 * This is the unified representation for all operator contributions in the
 * analysis subsystem.  Both FormExpr formulations and handwritten kernels
 * lower into this type.  Analyzer passes consume contributions via
 * ProblemAnalysisContext::contributions().
 */
struct ContributionDescriptor {
    std::string contribution_id;    ///< Stable id distinct from display/operator tag
    std::string operator_tag;       ///< e.g., "equations", "penalty", "contact"
    std::string origin;             ///< e.g., "FormsInstaller", "AssemblyKernel:MyKernel"
    DomainKind domain{DomainKind::Cell};
    int boundary_marker{-1};
    InterfaceScope interface_scope{InterfaceScope::SpecificMarker};
    int interface_marker{-1};       ///< Only meaningful when scope == SpecificMarker
    std::vector<ContributionDomainScope> domain_scopes;

    std::vector<VariableKey> test_variables;
    std::vector<VariableKey> trial_variables;
    std::vector<VariableKey> related_variables;

    ContributionRole role{ContributionRole::DiagonalBlock};
    OperatorTraitFlags traits{OperatorTraitFlags::None};
    AnalysisConfidence confidence{AnalysisConfidence::High};
    DefinitenessInterpretation definiteness_interpretation{
        DefinitenessInterpretation::Unknown};

    std::vector<NullspaceHint> nullspace_hints;

    // ---- Phase 22 extended metadata ----

    std::optional<NullspaceEffect> nullspace_effect;
    std::optional<ConsistencyKind> consistency_kind;
    std::optional<AdjointConsistencyKind> adjoint_consistency;
    std::optional<ScalingDescriptor> scaling;
    std::optional<TemporalDescriptor> temporal;
    std::optional<BalanceDescriptor> balance;
    std::vector<PairingDescriptor> pairings;
    std::optional<TransportCharacter> transport_character;
    std::vector<FormParameterUsage> parameter_usages;
    std::vector<FormCoefficientUsage> coefficient_usages;
    std::vector<FormScaleUsage> scale_usages;

    // ---- Mixed-form provenance (Phase 4) ----

    /// Block key from the source FormulationRecord::block_residual_exprs.
    /// Allows tracing this contribution back to the specific (test, trial) block
    /// that generated it.
    std::optional<std::pair<FieldId, FieldId>> source_block_key;

    /// Source block expression handle.  Retained for diagnostics so analysis
    /// issues can point back to the specific block sub-expression.
    std::shared_ptr<const forms::FormExprNode> source_expression;

    /// Human-readable block context string for diagnostic messages.
    /// E.g., "test=primal_vector, trial=scalar_multiplier" for a mixed off-diagonal block.
    std::string block_context;

    // ---- Builder helpers ----

    /// Build a diagonal symmetric block (e.g., Laplacian, elasticity)
    [[nodiscard]] static ContributionDescriptor diagonalSymmetric(
        VariableKey field, std::string op_tag, std::string orig);

    /// Build a constraint/multiplier block.
    [[nodiscard]] static ContributionDescriptor constraintBlock(
        VariableKey test, VariableKey trial, std::string op_tag, std::string orig);

    /// Build a stabilization block (e.g., PSPG, GLS)
    [[nodiscard]] static ContributionDescriptor stabilization(
        VariableKey field, std::string op_tag, std::string orig);

    /// Build a global coupling contribution (e.g., coupled boundary)
    [[nodiscard]] static ContributionDescriptor globalCoupling(
        std::vector<VariableKey> test, std::vector<VariableKey> trial,
        std::string op_tag, std::string orig);

    // ---- Phase 22 builder helpers ----

    /// Build a mass-like contribution (temporal accumulation)
    [[nodiscard]] static ContributionDescriptor massLike(
        VariableKey field, std::string op_tag, std::string orig);

    /// Build an exchange coupling (inter-domain / interface)
    [[nodiscard]] static ContributionDescriptor exchangeCoupling(
        VariableKey test, VariableKey trial,
        std::string balance_group, std::string op_tag, std::string orig);

    /// Build a constraint pair (Lagrange multiplier / adjoint pair)
    [[nodiscard]] static ContributionDescriptor constraintPairDesc(
        VariableKey primal, VariableKey dual,
        std::string pairing_group, std::string op_tag, std::string orig);

    /// Build a transport-like contribution (first-order / convection)
    [[nodiscard]] static ContributionDescriptor transportLike(
        VariableKey field, std::string op_tag, std::string orig);

    /// Deterministic identity derived from stable provenance and block scope.
    [[nodiscard]] static std::string stableContributionId(
        const ContributionDescriptor& desc);

    /// Populate contribution_id when the producer did not provide one.
    void ensureStableContributionId();
};

// ============================================================================
// String conversion
// ============================================================================

[[nodiscard]] const char* toString(ContributionRole r) noexcept;
[[nodiscard]] const char* toString(NullspaceFamily f) noexcept;
[[nodiscard]] const char* toString(InterfaceScope s) noexcept;

// Phase 22 toString helpers
[[nodiscard]] const char* toString(NullspaceEffect e) noexcept;
[[nodiscard]] const char* toString(ConsistencyKind k) noexcept;
[[nodiscard]] const char* toString(AdjointConsistencyKind k) noexcept;
[[nodiscard]] const char* toString(TemporalContributionKind k) noexcept;
[[nodiscard]] const char* toString(BalanceRole r) noexcept;
[[nodiscard]] const char* toString(PairingKind k) noexcept;
[[nodiscard]] const char* toString(TransportCharacter c) noexcept;

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_CONTRIBUTION_DESCRIPTOR_H
