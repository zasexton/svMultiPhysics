/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_COUPLEDFIELDGAUGECONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_COUPLEDFIELDGAUGECONSTRAINT_H

/**
 * @file CoupledFieldGaugeConstraint.h
 * @brief One shared algebraic gauge for two coupled scalar fields.
 */

#include "Constraints/SystemConstraint.h"
#include "Core/Types.h"

#include <optional>

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief Remove one common constant mode from a pair of coupled scalar fields.
 *
 * The constraint pins the lowest globally numbered, locally owned DOF that is
 * still free after earlier system constraints have been lowered.  It searches
 * the first field before the second field through the monolithic numbering.
 * Registration order is therefore part of the contract: activity and
 * aggregation constraints for both fields must be registered first.
 *
 * This constraint removes exactly one common mode.  It must only be used when
 * the coupled operator already controls the relative constant between fields.
 */
class CoupledFieldGaugeConstraint final : public ISystemConstraint {
public:
    CoupledFieldGaugeConstraint(FieldId first_field,
                                FieldId second_field,
                                Real pinned_value = Real{0.0});

    void apply(const systems::FESystem& system,
               AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    [[nodiscard]] bool isTimeDependent() const noexcept override
    {
        return false;
    }

    [[nodiscard]] ConstraintDependencyDeclaration
    dependencyDeclaration() const override;

    [[nodiscard]] systems::SetupStorageRequirements
    storageRequirements() const noexcept override;

    [[nodiscard]] std::optional<GlobalIndex> pinnedDof() const noexcept
    {
        return pinned_dof_;
    }

private:
    FieldId first_field_{INVALID_FIELD_ID};
    FieldId second_field_{INVALID_FIELD_ID};
    Real pinned_value_{Real{0.0}};
    std::optional<GlobalIndex> pinned_dof_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_COUPLEDFIELDGAUGECONSTRAINT_H
