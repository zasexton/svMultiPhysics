/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_CONSTRAINT_ANALYSIS_SUMMARY_H
#define SVMP_FE_ANALYSIS_CONSTRAINT_ANALYSIS_SUMMARY_H

/**
 * @file ConstraintAnalysisSummary.h
 * @brief Summary of constraint state for problem analysis
 *
 * Built after constraints are assembled, provides enough information for
 * under/over-constraint checks without re-walking AffineConstraints internals.
 *
 * Intentionally FE-DOF-focused.  Non-FE couplings (aux states, boundary
 * functionals) are represented via KernelContributionRecord and
 * BoundaryConditionDescriptor, not here.
 *
 * @see ConstraintRankAnalyzer (Phase 7) for the primary consumer
 * @see ProblemAnalysisContext for storage
 */

#include "Core/Types.h"

#include <functional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {

namespace constraints {
class AffineConstraints;
} // namespace constraints

namespace analysis {

class TopologyAnalysisContext;

/**
 * @brief Summary of constrained DOFs for one field/component/region slice
 */
struct ConstrainedDofSet {
    FieldId field{INVALID_FIELD_ID};
    int component{-1};           ///< -1 = all components aggregated
    int region{-1};              ///< -1 = global
    int num_constrained_dofs{0};
    int num_total_dofs{0};
    double constrained_fraction{0.0};
    std::string constraint_source;  ///< "StrongDirichlet", "AffineRelation", "Mixed"
};

/**
 * @brief A master-slave DOF relation
 */
struct ConstraintRelation {
    GlobalIndex slave_dof{INVALID_GLOBAL_INDEX};
    GlobalIndex master_dof{INVALID_GLOBAL_INDEX};
    double coefficient{1.0};
    std::string type;  ///< "dirichlet", "periodic", "mpc", "affine"
};

/**
 * @brief A detected constraint conflict (same DOF constrained incompatibly)
 */
struct ConstraintConflict {
    GlobalIndex dof{INVALID_GLOBAL_INDEX};
    std::vector<std::string> conflicting_sources;
    std::string description;
};

/**
 * @brief Aggregate constraint analysis for the problem
 */
class ConstraintAnalysisSummary {
public:
    ConstraintAnalysisSummary() = default;

    /// Per-field/component/region constrained DOF sets
    std::vector<ConstrainedDofSet> constrained_sets;

    /// Detected constraint conflicts
    std::vector<ConstraintConflict> conflicts;

    // ---- Queries ----

    [[nodiscard]] bool hasConflicts() const noexcept {
        return !conflicts.empty();
    }

    /// Total number of constrained DOFs across all fields
    [[nodiscard]] int totalConstrainedDofs() const noexcept;

    /// Total number of DOFs across all fields
    [[nodiscard]] int totalDofs() const noexcept;

    /// Constrained fraction for a specific field/component/region
    /// Returns -1.0 if no matching set found
    [[nodiscard]] double constrainedFraction(FieldId field, int component = -1, int region = -1) const noexcept;

    /// Fields with zero constrained DOFs
    [[nodiscard]] std::vector<FieldId> unconstrainedFields() const;

    /// Fields with 100% constrained DOFs
    [[nodiscard]] std::vector<FieldId> fullyConstrainedFields() const;

    // ---- Factory ----

    /**
     * @brief Description of a field's DOF range for constraint scanning
     */
    struct FieldDofRange {
        FieldId field_id{INVALID_FIELD_ID};
        GlobalIndex dof_offset{0};      ///< First DOF index for this field
        GlobalIndex num_dofs{0};         ///< Total DOFs for this field
        int num_components{1};           ///< Number of components
    };

    /// Callback: given a DOF index, returns its connected-component region ID.
    /// Returns -1 if region is unknown or DOF is not mesh-associated.
    using DofRegionProvider = std::function<int(GlobalIndex dof)>;

    /// Callback: given (field_id, component), returns the DOF indices for
    /// that component. Returns empty if component extraction is not supported
    /// (e.g., VectorBasis fields like HDiv/HCurl).
    using ComponentDofProvider = std::function<std::vector<GlobalIndex>(FieldId field_id, int component)>;

    /**
     * @brief Build constraint summary from AffineConstraints and field DOF ranges
     *
     * @param ac               Finalized (closed) AffineConstraints
     * @param fields           DOF ranges for each field
     * @param topology         Optional topology context (unused; kept for API compat)
     * @param dof_region       Optional DOF→region callback for per-region grouping
     * @param comp_dofs        Optional component DOF provider for layout-correct per-component sets
     */
    [[nodiscard]] static ConstraintAnalysisSummary
    build(const constraints::AffineConstraints& ac,
          std::span<const FieldDofRange> fields,
          const TopologyAnalysisContext* topology = nullptr,
          const DofRegionProvider& dof_region = nullptr,
          const ComponentDofProvider& comp_dofs = nullptr);
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_CONSTRAINT_ANALYSIS_SUMMARY_H
