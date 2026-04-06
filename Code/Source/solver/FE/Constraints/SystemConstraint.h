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
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_SYSTEMCONSTRAINT_H

