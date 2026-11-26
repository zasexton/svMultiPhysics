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
 *
 * With a generated active subset of the marker (the wetted part of a cut
 * boundary, by its generated marker in the installed cut-integration
 * context), only the faces of positive wetted measure hold nodes: a degree of
 * freedom whose basis function has a nonzero trace only on dry faces is left
 * to the constraints of the active side (a dry node of a cut cell would
 * otherwise shape the liquid velocity inside that cell).  A new cut context
 * rebuilds the constraint state, so the set follows the cut; while the
 * installed context has no such subset, the whole marker counts.
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
    /// A strong declaration of higher precedence that holds the degrees of
    /// freedom it shares with this one: its marker and component.
    struct Yield {
        int boundary_marker{-1};
        int component{-1};
        int active_boundary_marker{-1};   ///< its wetted subset, when restricted
    };

    /// @p active_boundary_marker, when >= 0, is the generated active subset
    /// of @p boundary_marker the condition acts on.
    StrongDirichletConstraint(FieldId field,
                              int boundary_marker,
                              forms::FormExpr value,
                              int component = -1,
                              std::vector<Yield> yields = {},
                              int active_boundary_marker = -1);

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

    /// Removes from the extracted points those that a yielded declaration holds.
    void dropYieldedPoints(const systems::FESystem& system,
                           std::vector<GlobalIndex>& dofs,
                           std::vector<std::array<Real, 3>>& coords,
                           std::vector<Real>& samples) const;

    std::vector<Yield> yields_{};
    FieldId field_{INVALID_FIELD_ID};
    int boundary_marker_{-1};
    int component_{-1};
    int active_boundary_marker_{-1};
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
