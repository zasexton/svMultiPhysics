/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_GAUGE_REGISTRY_H
#define SVMP_FE_CONSTRAINTS_GAUGE_REGISTRY_H

/**
 * @file GaugeRegistry.h
 * @brief Automatic nullspace detection and gauge-condition enforcement
 *
 * GaugeRegistry is a systems-side registry that is separate from AffineConstraints.
 * It stores candidate nullspace modes, anchoring evidence, and enforcement policy
 * decisions.  Two paths produce candidates:
 *
 *   Path A — automatic inference from Forms residual expressions
 *            (NullspaceAnalyzer walks the FormExpr DAG).
 *   Path B — explicit declarations from hand-written AssemblyKernel /
 *            GlobalKernel / custom-BC gaugeMetadata() hooks.
 *
 * After all candidates and anchoring evidence have been collected, the resolver
 * merges, deduplicates, classifies each mode, and chooses an enforcement policy.
 * For exact nullspace modes it auto-creates GlobalConstraint objects; for
 * uncertain/near-nullspace modes it logs a warning and optionally falls back to
 * a conservative strategy (pinning).
 *
 * @see GlobalConstraint for the algebraic enforcement machinery
 * @see NullspaceAnalyzer for FormExpr-based inference (Path A)
 */

#include "Core/Types.h"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {

namespace constraints {
class AffineConstraints;
class GlobalConstraint;
} // namespace constraints

namespace gauge {

// ============================================================================
// Enumerations
// ============================================================================

/**
 * @brief Canonical nullspace mode families
 *
 * Phase 1: ScalarConstant
 * Phase 2: ComponentwiseConstant, KernelOfSymGrad
 */
enum class NullspaceModeFamily : std::uint8_t {
    ScalarConstant,           ///< p → p + c  (scalar field constant shift)
    ComponentwiseConstant,    ///< u_i → u_i + c_i  (per-component vector shift)
    KernelOfSymGrad,          ///< ker(sym(grad)): translations enforced; rotations deferred
};

/**
 * @brief Confidence level attached to an inferred nullspace mode
 */
enum class Confidence : std::uint8_t {
    High,       ///< Symbolic analysis is conclusive
    Medium,     ///< Likely nullspace but stabilization/penalty may break it
    Low,        ///< Heuristic guess
};

/**
 * @brief How a BC or operator term affects a candidate nullspace mode
 */
enum class AnchoringVerdict : std::uint8_t {
    Anchored,            ///< Fully fixes the mode (e.g., Dirichlet on the field)
    PartiallyAnchored,   ///< Weakly broken (penalty, stabilization)
    Preserved,           ///< Compatible with the mode (e.g., pure Neumann)
    Unknown,             ///< Cannot determine
};

/**
 * @brief Classification of a resolved mode
 */
enum class GaugeStatus : std::uint8_t {
    Anchored,         ///< Fully constrained by BCs/terms — no action needed
    ExactNullspace,   ///< Confirmed nullspace — needs enforcement
    NearNullspace,    ///< Stabilization weakly breaks it — warn only
    Unknown,          ///< Insufficient information
};

/**
 * @brief Strategy for enforcing a detected nullspace mode
 */
enum class EnforcementPolicy : std::uint8_t {
    None,                   ///< Mode is anchored, nothing to do
    PinDof,                 ///< Deterministic fallback: pin one DOF to zero
    MeanZeroElimination,    ///< Constrain mean to zero via GlobalConstraint
    LagrangeMultiplier,     ///< Augment system with Lagrange multiplier
    SolverNullspace,        ///< Pass nullspace basis to solver (currently not assigned by resolver)
};

// ============================================================================
// Data structures
// ============================================================================

/**
 * @brief Source of a gauge candidate
 */
enum class CandidateSource : std::uint8_t {
    FormsInference,       ///< Automatic inference from FormExpr DAG
    ExplicitDeclaration,  ///< Explicit declaration from kernel/BC metadata
};

/**
 * @brief A candidate nullspace mode for a field
 */
struct GaugeCandidate {
    FieldId field{INVALID_FIELD_ID};   ///< Field that may have a nullspace
    int component{-1};                 ///< -1 = all components (scalar or full vector)
    int region{-1};                    ///< -1 = global (all regions), >=0 = specific connected component
    NullspaceModeFamily family{NullspaceModeFamily::ScalarConstant};
    Confidence confidence{Confidence::High};
    std::string reason;                ///< Human-readable provenance
    CandidateSource source{CandidateSource::FormsInference};
};

/**
 * @brief Evidence that a BC or operator term anchors (or preserves) a mode
 *
 * The family field distinguishes e.g. "Robin anchors ScalarConstant" from
 * "Robin only partially anchors KernelOfSymGrad."  When family is not set
 * (nullopt), the evidence applies to all families for the given field/component.
 */
struct AnchoringEvidence {
    FieldId field{INVALID_FIELD_ID};
    int component{-1};
    int region{-1};                                ///< -1 = global, >=0 = specific region
    std::optional<NullspaceModeFamily> family{};   ///< nullopt = applies to all families
    AnchoringVerdict verdict{AnchoringVerdict::Unknown};
    std::string source;  ///< e.g., "DirichletBC on boundary 3"
    int boundary_marker{-1};                       ///< -1 = unknown, >=0 = specific mesh boundary marker
};

/**
 * @brief A fully resolved nullspace mode with status and enforcement decision
 */
struct ResolvedMode {
    GaugeCandidate candidate;
    GaugeStatus status{GaugeStatus::Unknown};
    EnforcementPolicy policy{EnforcementPolicy::None};
    std::vector<AnchoringEvidence> anchoring;
};

// ============================================================================
// GaugeRegistry
// ============================================================================

/**
 * @brief Registry for candidate nullspace modes and anchoring evidence
 *
 * Lifecycle:
 *   1. During formulation installation (FormsInstaller), NullspaceAnalyzer
 *      adds candidates via addCandidate().
 *   2. During system setup (SystemSetup), BC anchoring evidence is added
 *      via addAnchoring().  Non-Forms kernels may also contribute candidates.
 *   3. resolve() merges candidates, applies anchoring rules, classifies
 *      each mode, and chooses enforcement policies.
 *   4. applyEnforcement() auto-creates GlobalConstraint objects for
 *      exact nullspace modes and registers them with AffineConstraints.
 */
class GaugeRegistry {
public:
    GaugeRegistry() = default;

    // ---- Candidate registration ----

    void addCandidate(GaugeCandidate candidate);
    void addAnchoring(AnchoringEvidence evidence);

    /**
     * @brief Post-process anchoring evidence to assign regions from boundary markers.
     *
     * For each Anchored/PartiallyAnchored evidence with region=-1 and a valid
     * boundary_marker (>= 0), calls @p get_marker_regions to determine which
     * connected components the boundary marker touches.  Replaces the global
     * evidence with per-region copies.
     *
     * Evidence without a boundary_marker (e.g., kernel metadata or formulation-
     * level anchors for unlabeled boundaries) is left unchanged.  On disconnected
     * meshes, such unresolved global anchoring evidence is blocked by the resolver
     * from matching per-region candidates — the affected regions get gauge
     * enforcement regardless.  Labeled boundary markers are required for correct
     * per-region scoping.
     *
     * Must be called AFTER the region_provider is built and BEFORE resolve().
     */
    using MarkerRegionResolver = std::function<std::vector<int>(int boundary_marker)>;
    void retagEvidenceRegions(const MarkerRegionResolver& get_marker_regions);

    // ---- Accessors ----

    [[nodiscard]] const std::vector<GaugeCandidate>& candidates() const noexcept {
        return candidates_;
    }
    [[nodiscard]] const std::vector<AnchoringEvidence>& anchoring() const noexcept {
        return anchoring_;
    }
    [[nodiscard]] const std::vector<ResolvedMode>& resolvedModes() const noexcept {
        return resolved_;
    }
    [[nodiscard]] bool isResolved() const noexcept { return resolved_flag_; }

    // ---- Resolution ----

    /**
     * @brief Resolve all candidates against anchoring evidence
     *
     * @param get_field_dofs  Callback: given (field_id, component), returns the
     *                        global DOF indices for that field/component.
     *                        Component -1 means "all DOFs of the field".
     *
     * After resolution, resolvedModes() returns the classified modes and
     * applyEnforcement() can auto-create constraints.
     */
    using DofProvider = std::function<std::vector<GlobalIndex>(FieldId, int component)>;

    /**
     * @brief Callback: given a DOF index, returns its connected-component ID.
     *
     * When null, all DOFs are treated as region -1 (global).
     * When set and a candidate has region=-1, expand it into one candidate
     * per connected component.
     */
    using RegionProvider = std::function<int(GlobalIndex dof)>;

    /**
     * @brief Callback: given a field and DOF, returns its physical coordinates.
     *
     * Used for constructing rotation mode nullspace vectors for KernelOfSymGrad.
     * When null, only translation modes are produced.
     */
    using CoordinateProvider = std::function<std::array<double,3>(FieldId, GlobalIndex dof)>;

    /**
     * @brief Callback: returns lumped mass weights w_i = integral(phi_i) for a field/component.
     *
     * Used for FE-correct weighted mean-zero enforcement.
     * When null, uniform weights are used (current behavior).
     */
    using MassWeightProvider = std::function<std::vector<double>(FieldId, int component)>;

    void resolve(const DofProvider& get_field_dofs,
                 const RegionProvider& get_region = nullptr,
                 const CoordinateProvider& get_coords = nullptr);

    /**
     * @brief Apply enforcement for resolved exact-nullspace modes
     *
     * Auto-creates GlobalConstraint objects and applies them to the given
     * AffineConstraints.  For uncertain/near-nullspace modes, logs a warning
     * and optionally applies a conservative fallback (PinDof).
     *
     * @param constraints  AffineConstraints to register new constraints with
     *                     (must still be open, not yet closed)
     * @param get_field_dofs  Same callback as resolve()
     * @return Number of constraints created
     */
    int applyEnforcement(constraints::AffineConstraints& constraints,
                         const DofProvider& get_field_dofs,
                         const MassWeightProvider& get_mass_weights = nullptr);

    // ---- Nullspace basis construction ----

    /**
     * @brief Build dense nullspace basis vectors for SolverNullspace modes
     *
     * Returns one vector per nullspace mode resolved with
     * EnforcementPolicy::SolverNullspace.  Each vector has length @p n_total_dofs
     * and is normalized to unit L2 norm.
     *
     * For ScalarConstant: one vector with 1/sqrt(n) at every field DOF.
     * For ComponentwiseConstant: one vector per component.
     * For KernelOfSymGrad: currently falls back to componentwise constant
     * (translation-only; full RB mode vectors require mesh coordinates,
     * deferred to a future phase).
     *
     * @param n_total_dofs  Total DOFs in the monolithic system
     * @param get_field_dofs  Same callback as resolve()
     * @return  Orthonormalized basis vectors (may be empty if no SolverNullspace modes)
     */
    [[nodiscard]] std::vector<std::vector<double>>
    buildNullspaceBasis(GlobalIndex n_total_dofs,
                        const DofProvider& get_field_dofs,
                        const CoordinateProvider& get_coords = nullptr) const;

    // ---- Diagnostics ----

    /**
     * @brief Human-readable diagnostic report of all candidates and resolved modes
     */
    [[nodiscard]] std::string diagnosticReport() const;

    // ---- Reset ----

    void clear() noexcept;

private:
    std::vector<GaugeCandidate> candidates_;
    std::vector<AnchoringEvidence> anchoring_;
    std::vector<ResolvedMode> resolved_;
    bool resolved_flag_{false};

    // Retained from resolve() for use in applyEnforcement/buildNullspaceBasis
    RegionProvider region_provider_{};

    // Helper: filter DOFs by region when the resolved mode has a specific region.
    [[nodiscard]] std::vector<GlobalIndex>
    getRegionFilteredDofs(const DofProvider& get_field_dofs,
                          FieldId field, int component, int region) const;
};

// ============================================================================
// String conversion utilities
// ============================================================================

[[nodiscard]] const char* toString(NullspaceModeFamily f) noexcept;
[[nodiscard]] const char* toString(Confidence c) noexcept;
[[nodiscard]] const char* toString(AnchoringVerdict v) noexcept;
[[nodiscard]] const char* toString(GaugeStatus s) noexcept;
[[nodiscard]] const char* toString(EnforcementPolicy p) noexcept;

} // namespace gauge
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_GAUGE_REGISTRY_H
