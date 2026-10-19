#ifndef SVMP_FE_CONSTRAINTS_SYSTEMCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_SYSTEMCONSTRAINT_H

/**
 * @file SystemConstraint.h
 * @brief Constraints that require mesh + DOF access for lowering
 *
 * Some constraints (e.g., strong Dirichlet specified by boundary marker) need
 * access to mesh topology and the finalized DOF maps to determine which DOFs
 * are constrained.
 *
 * This interface allows those constraints to be declared during the definition
 * phase and lowered/applied during `FESystem::setup()`.
 */

#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintDependency.h"
#include "Systems/SetupStoragePlan.h"

#include <string>

namespace svmp {
namespace FE {

namespace systems { class FESystem; }

namespace constraints {

class ISystemConstraint {
public:
    virtual ~ISystemConstraint() = default;

    /**
     * @brief Apply this constraint into the system's AffineConstraints
     *
     * Called during `FESystem::setup()` after field DOFs are distributed/finalized
     * and before constraints are closed/synchronized.
     */
    virtual void apply(const systems::FESystem& system, AffineConstraints& constraints) = 0;

    /**
     * @brief Update inhomogeneities for a new time (for time-dependent constraints)
     *
     * Called by `FESystem::updateConstraints(...)` after `setup()` has completed.
     */
    virtual bool updateValues(const systems::FESystem& system,
                              AffineConstraints& constraints,
                              double time,
                              double dt) = 0;

    [[nodiscard]] virtual bool isTimeDependent() const noexcept = 0;

    /**
     * @brief Revision domains that can invalidate this setup-time constraint.
     *
     * Mesh-aware system constraints default to depending structurally on
     * boundary topology/labels/ownership/numbering and FE DOF layout because
     * setup-time lowering maps mesh entities to algebraic DOFs. Subclasses
     * with geometry-dependent values should add value.geometry dependencies.
     */
    [[nodiscard]] virtual ConstraintDependencyDeclaration dependencyDeclaration() const
    {
        ConstraintDependencyDeclaration out;
        out.structural = ConstraintDependencyMask::meshBoundaryTopology();
        merge_into(out.structural, ConstraintDependencyMask::feDofLayout());
        if (isTimeDependent()) {
            out.value.time = true;
        }
        return out;
    }

    /**
     * @brief Storage required to lower this setup-time constraint.
     *
     * Implementations must state their requirements explicitly so FE setup can
     * avoid conservative topology/global-lookup allocation while still failing
     * deterministically when a constraint needs unavailable metadata.
     */
    [[nodiscard]] virtual systems::SetupStorageRequirements storageRequirements() const noexcept = 0;

    /**
     * @brief Parameters of this constraint for the ordered constraint manifest
     *
     * One line of space-separated key=value pairs naming what the constraint
     * acts on (field, marker, side, values), used by
     * FESystem::constraintManifestLines() to compare the registered
     * constraint set before and after a refactoring.  It must depend only on
     * definition-time data (no mesh, DOF or rank state).  The default is
     * empty: the manifest then shows the type alone.
     */
    [[nodiscard]] virtual std::string manifestDescription() const { return {}; }
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_SYSTEMCONSTRAINT_H
