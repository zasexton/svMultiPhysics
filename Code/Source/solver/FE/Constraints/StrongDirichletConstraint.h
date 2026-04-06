#ifndef SVMP_FE_CONSTRAINTS_STRONGDIRICHLETCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_STRONGDIRICHLETCONSTRAINT_H

/**
 * @file StrongDirichletConstraint.h
 * @brief Mesh-aware lowering for Forms strong Dirichlet declarations
 */

#include "Constraints/SystemConstraint.h"

#include "Forms/FormExpr.h"

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

class StrongDirichletConstraint final : public ISystemConstraint {
public:
    StrongDirichletConstraint(FieldId field, int boundary_marker, forms::FormExpr value, int component = -1);

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    [[nodiscard]] bool isTimeDependent() const noexcept override { return is_time_dependent_; }

private:
    FieldId field_{INVALID_FIELD_ID};
    int boundary_marker_{-1};
    int component_{-1};
    forms::FormExpr value_{};
    bool is_time_dependent_{false};

    std::vector<GlobalIndex> dofs_{};
    std::vector<std::array<Real, 3>> coords_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_STRONGDIRICHLETCONSTRAINT_H
