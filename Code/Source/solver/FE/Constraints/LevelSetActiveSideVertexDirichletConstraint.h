/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_LEVELSETACTIVESIDEVERTEXDIRICHLETCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_LEVELSETACTIVESIDEVERTEXDIRICHLETCONSTRAINT_H

/**
 * @file LevelSetActiveSideVertexDirichletConstraint.h
 * @brief Dirichlet constraints for scalar H1 DOFs outside a level-set side.
 */

#include "Constraints/SystemConstraint.h"
#include "Core/Types.h"

#include <string>

namespace svmp {
namespace FE {
namespace constraints {

enum class LevelSetConstraintSide {
    Negative,
    Positive,
};

class LevelSetActiveSideVertexDirichletConstraint final : public ISystemConstraint {
public:
    LevelSetActiveSideVertexDirichletConstraint(FieldId field,
                                                std::string level_set_field_name,
                                                LevelSetConstraintSide active_side,
                                                Real isovalue = Real{0.0},
                                                Real inactive_value = Real{0.0});

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }

    [[nodiscard]] ConstraintDependencyDeclaration dependencyDeclaration() const override;

    [[nodiscard]] systems::SetupStorageRequirements storageRequirements() const noexcept override;

private:
    FieldId field_{INVALID_FIELD_ID};
    std::string level_set_field_name_{};
    LevelSetConstraintSide active_side_{LevelSetConstraintSide::Negative};
    Real isovalue_{Real{0.0}};
    Real inactive_value_{Real{0.0}};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_LEVELSETACTIVESIDEVERTEXDIRICHLETCONSTRAINT_H
