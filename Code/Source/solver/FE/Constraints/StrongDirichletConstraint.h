#ifndef SVMP_FE_CONSTRAINTS_STRONGDIRICHLETCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_STRONGDIRICHLETCONSTRAINT_H

/**
 * @file StrongDirichletConstraint.h
 * @brief Mesh-aware lowering for Forms strong Dirichlet declarations
 *
 * The value is evaluated at each constrained DOF's node (active coordinates)
 * and may read the position, the time and the time step, the time-function
 * parameters of the system (systems::timeFunction, evaluated at the update
 * time), prescribed fields that share the constrained field's scalar
 * element (their coefficients at the node; zeros at setup when they are not
 * set yet, and required at every update), and FormExpr::normal(): the
 * marker's averaged facet normal at the node (boundaryPointNormals, the same
 * at any rank count), refused at a node where the facets meet at an edge or
 * corner.  It must not read an unknown.
 */

#include "Constraints/SystemConstraint.h"

#include "Forms/FormExpr.h"

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
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

    [[nodiscard]] ConstraintDependencyDeclaration dependencyDeclaration() const override;

    [[nodiscard]] systems::SetupStorageRequirements storageRequirements() const noexcept override;

    [[nodiscard]] std::string manifestDescription() const override;

private:
    /// The value of each parameter of the value at @p time.
    [[nodiscard]] std::vector<Real> parameterValues(const systems::FESystem& system, double time) const;

    /// The boundary normal at each owned node of the marker when the value reads it.
    [[nodiscard]] std::unordered_map<GlobalIndex, std::optional<std::array<Real, 3>>>
    pointNormals(const systems::FESystem& system) const;

    FieldId field_{INVALID_FIELD_ID};
    int boundary_marker_{-1};
    int component_{-1};
    forms::FormExpr value_{};
    forms::FormExpr point_value_{};             ///< value_ with parameter i as slot i
    std::vector<std::string> parameter_names_{};
    std::vector<FieldId> sampled_fields_{};     ///< the prescribed fields the value reads
    bool reads_normal_{false};                  ///< whether the value reads FormExpr::normal()
    bool is_time_dependent_{false};

    std::vector<GlobalIndex> dofs_{};
    std::vector<std::array<Real, 3>> coords_{};
    std::vector<int> sample_components_{};      ///< per sampled field
    std::vector<Real> samples_{};               ///< the sampled field values, DOF by DOF
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_STRONGDIRICHLETCONSTRAINT_H
