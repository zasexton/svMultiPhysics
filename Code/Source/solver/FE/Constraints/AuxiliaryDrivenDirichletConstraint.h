#ifndef SVMP_FE_CONSTRAINTS_AUXILIARY_DRIVEN_DIRICHLET_CONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_AUXILIARY_DRIVEN_DIRICHLET_CONSTRAINT_H

#include "Constraints/SystemConstraint.h"

#include "Auxiliary/AuxiliaryBindings.h"

#include <array>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

class AuxiliaryDrivenDirichletConstraint final : public ISystemConstraint {
public:
    AuxiliaryDrivenDirichletConstraint(std::string instance_name,
                                       systems::AuxiliaryConstraintBinding binding);

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    [[nodiscard]] bool isTimeDependent() const noexcept override { return true; }

private:
    std::string instance_name_{};
    systems::AuxiliaryConstraintBinding binding_{};
    std::vector<GlobalIndex> dofs_{};
    std::vector<std::array<Real, 3>> coords_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_AUXILIARY_DRIVEN_DIRICHLET_CONSTRAINT_H
