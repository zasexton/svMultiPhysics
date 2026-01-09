#ifndef SVMP_FE_SYSTEMS_SYSTEMCONSTRAINT_H
#define SVMP_FE_SYSTEMS_SYSTEMCONSTRAINT_H

/**
 * @file SystemConstraint.h
 * @brief Systems-side constraints that require mesh + DOF access
 *
 * FE/Constraints defines algebraic constraints in terms of DOF indices.
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
namespace systems {

class FESystem;

class ISystemConstraint {
public:
    virtual ~ISystemConstraint() = default;

    /**
     * @brief Apply this constraint into the system's AffineConstraints
     *
     * Called during `FESystem::setup()` after field DOFs are distributed/finalized
     * and before constraints are closed/synchronized.
     */
    virtual void apply(const FESystem& system, constraints::AffineConstraints& constraints) = 0;

    /**
     * @brief Update inhomogeneities for a new time (for time-dependent constraints)
     *
     * Called by `FESystem::updateConstraints(...)` after `setup()` has completed.
     */
    virtual bool updateValues(const FESystem& system,
                              constraints::AffineConstraints& constraints,
                              double time,
                              double dt) = 0;

    [[nodiscard]] virtual bool isTimeDependent() const noexcept = 0;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SYSTEMCONSTRAINT_H

