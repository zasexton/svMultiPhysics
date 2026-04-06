/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_HDIVNORMALCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_HDIVNORMALCONSTRAINT_H

/**
 * @file HDivNormalConstraint.h
 * @brief Essential (strong) boundary constraint for H(div) normal traces
 *
 * This constraint enforces a homogeneous normal boundary condition:
 *   n · B = 0  on boundary marker Γ
 * by constraining all H(div) DOFs associated with boundary facets.
 *
 * NOTE: This initial implementation supports homogeneous constraints only.
 */

#include "Constraints/SystemConstraint.h"

#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

class HDivNormalConstraint final : public ISystemConstraint {
public:
    HDivNormalConstraint(FieldId field, int boundary_marker);

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& /*system*/,
                      AffineConstraints& /*constraints*/,
                      double /*time*/,
                      double /*dt*/) override
    {
        return false;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }

private:
    FieldId field_{INVALID_FIELD_ID};
    int boundary_marker_{-1};

    std::vector<GlobalIndex> dofs_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_HDIVNORMALCONSTRAINT_H

