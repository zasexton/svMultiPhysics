/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_BOUNDARY_CONDITION_DESCRIPTOR_H
#define SVMP_FE_ANALYSIS_BOUNDARY_CONDITION_DESCRIPTOR_H

/**
 * @file BoundaryConditionDescriptor.h
 * @brief Rich mathematical descriptor for a boundary condition
 *
 * Provides a structured description of a boundary condition
 * consumable by multiple analyzers (nullspace, constraint rank, compatibility).
 *
 * @see BoundaryCondition::analysisMetadata() for the producer interface
 * @see ConstraintRankAnalyzer for the primary consumer
 */

#include "Core/Types.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Constraints/GaugeRegistry.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {

// Forward declaration for compatibility adapter
namespace gauge {
enum class NullspaceModeFamily : std::uint8_t;
enum class AnchoringVerdict : std::uint8_t;
} // namespace gauge

namespace analysis {

/**
 * @brief What is being prescribed on the boundary
 */
enum class TraceKind : std::uint8_t {
    Value,                ///< u = g (Dirichlet)
    NormalComponent,      ///< u·n = g
    TangentialComponent,  ///< u×n = g
    Flux,                 ///< σ·n = h (Neumann)
    NormalFlux,           ///< (k∇u)·n = h
    Mixed,                ///< Robin: αu + β(k∇u)·n = g
    AlgebraicRelation,    ///< Algebraic DOF relation (periodic, MPC)
};

/**
 * @brief How the boundary condition is enforced
 */
enum class EnforcementKind : std::uint8_t {
    Strong,           ///< Strong imposition (row elimination)
    WeakConsistent,   ///< Consistent weak form (Neumann-type)
    WeakPenalty,      ///< Penalty method (Robin-type)
    WeakNitsche,      ///< Nitsche's method (symmetric or nonsymmetric)
    WeakInequality,   ///< One-sided weak trace law with state-dependent activation
    AffineRelation,   ///< Algebraic constraint (periodic, MPC)
};

/**
 * @brief One-sided sense for inequality or complementarity trace laws
 */
enum class InequalitySense : std::uint8_t {
    None,
    LessEqual,
    GreaterEqual,
    Complementarity,
};

/**
 * @brief Nitsche formulation variant declared by the boundary-condition producer
 */
enum class NitscheVariant : std::uint8_t {
    Unknown,
    Symmetric,
    Nonsymmetric,
    Skew,
    Incomplete,
};

/**
 * @brief Explicit evidence for Nitsche consistency and penalty stability
 *
 * Homogeneity of the prescribed data does not imply primal consistency,
 * adjoint consistency, or penalty stability. Producers should populate this
 * metadata when a weak Nitsche condition is lowered to analysis
 * contributions.
 */
struct NitscheMetadata {
    NitscheVariant variant{NitscheVariant::Unknown};
    bool primal_consistency_terms_present{false};
    bool adjoint_consistency_terms_present{false};
    bool rhs_consistency_terms_present{false};
    bool penalty_positive{false};
    bool penalty_scaling_verified{false};
    bool penalty_trace_bound_verified{false};
    std::string trace_inequality_certificate_id;
};

/**
 * @brief Rich mathematical descriptor for a boundary condition
 */
struct BoundaryConditionDescriptor {
    /// Primary variable this BC targets
    VariableKey primary_variable;

    int component{-1};                         ///< -1 = all components
    DomainKind domain{DomainKind::Boundary};   ///< Boundary / InterfaceFace / CoupledBoundary / Global
    int boundary_marker{-1};                   ///< Mesh boundary tag (-1 if not applicable)
    int interface_marker{-1};                  ///< Interface tag (-1 if not applicable)

    TraceKind trace_kind{TraceKind::Value};
    EnforcementKind enforcement_kind{EnforcementKind::Strong};

    bool is_homogeneous{false};                ///< g=0 (Neumann) or u=0 (Dirichlet)
    bool anchors_constant_mode{false};         ///< Removes constant-shift invariance
    bool anchors_rigid_body_translation{false}; ///< Removes translation invariance
    bool anchors_rigid_body_rotation{false};   ///< Removes rotation invariance

    /// Other variables coupled through this BC (e.g., RCR couples velocity + aux state)
    std::vector<VariableKey> related_variables;

    bool introduces_global_coupling{false};    ///< Non-local coupling (e.g., integral constraints)
    std::string trace_side;                    ///< "master"/"slave"/"both" when relevant
    std::string source;                        ///< Human-readable origin ("EssentialBC on marker 3")

    // Phase 22 extended metadata
    std::optional<NullspaceEffect> nullspace_effect;
    std::optional<ConsistencyKind> consistency_kind;
    std::optional<AdjointConsistencyKind> adjoint_consistency;
    std::optional<ScalingDescriptor> scaling;
    std::optional<BalanceDescriptor> balance;
    std::optional<PairingKind> relation_kind;
    std::optional<InequalitySense> inequality_sense;
    std::optional<NitscheMetadata> nitsche;
    std::string pairing_group;
    bool state_dependent_activation{false};
};

// ============================================================================
// String conversion
// ============================================================================

[[nodiscard]] const char* toString(TraceKind k) noexcept;
[[nodiscard]] const char* toString(EnforcementKind k) noexcept;
[[nodiscard]] const char* toString(InequalitySense k) noexcept;
[[nodiscard]] const char* toString(NitscheVariant k) noexcept;

// ============================================================================
// Lower BC descriptor → ContributionDescriptor(s)
// ============================================================================

/**
 * @brief Lower a BoundaryConditionDescriptor into normalized ContributionDescriptors
 *
 * Maps trace/enforcement semantics to structured contribution roles and traits:
 *   - Strong Dirichlet → BoundaryConstraint + NullspaceLifting
 *   - Periodic/MPC → ConstraintBlock + NullspacePreserving
 *   - Robin → BoundaryConstraint + HasMass + NullspaceLifting
 *   - Nitsche → BoundaryConstraint + HasSecondOrder + NullspaceLifting
 *   - Natural (Neumann) → BoundaryConstraint (no nullspace traits)
 *   - CoupledBoundary → GlobalCoupling
 */
[[nodiscard]] std::vector<ContributionDescriptor>
lowerBCDescriptor(const BoundaryConditionDescriptor& desc);

// ============================================================================
// Compatibility adapter: descriptor → gauge AnchoringVerdict
// ============================================================================

/**
 * @brief Map a BoundaryConditionDescriptor to a gauge AnchoringVerdict
 *
 * Maps the structured BoundaryConditionDescriptor produced by
 * analysisMetadata() into a gauge AnchoringVerdict for a given mode family.
 *
 * @param desc    The new-style descriptor
 * @param family  The nullspace mode family being queried
 * @return        The equivalent AnchoringVerdict
 */
[[nodiscard]] gauge::AnchoringVerdict
descriptorToVerdict(const BoundaryConditionDescriptor& desc,
                    gauge::NullspaceModeFamily family);

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_BOUNDARY_CONDITION_DESCRIPTOR_H
