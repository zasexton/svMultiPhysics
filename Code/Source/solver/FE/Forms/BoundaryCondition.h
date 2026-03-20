#ifndef SVMP_FE_FORMS_BOUNDARY_CONDITION_H
#define SVMP_FE_FORMS_BOUNDARY_CONDITION_H

/**
 * @file BoundaryCondition.h
 * @brief Polymorphic interface for physics-agnostic boundary conditions
 *
 * Boundary conditions are declared in FE/Forms (expression-level) and applied
 * through FE/Systems, which manages validation and lowering to enforcement
 * mechanisms (e.g., strong Dirichlet -> constraints).
 */

#include "Forms/BoundaryConditions.h"
#include "Analysis/BoundaryConditionDescriptor.h"

#include <vector>

namespace svmp {
namespace FE {

namespace constraints {
class AffineConstraints;
} // namespace constraints

namespace systems {
class FESystem;
} // namespace systems

namespace forms {
namespace bc {

class BoundaryCondition {
public:
    virtual ~BoundaryCondition() = default;

    [[nodiscard]] virtual int boundaryMarker() const = 0;

    /**
     * @brief Optional initialization hook (definition-time)
     *
     * Allows boundary conditions to register auxiliary state variables (ODEs),
     * coupled boundary functionals, or other system-side infrastructure before
     * assembly begins.
     */
    virtual void setup(systems::FESystem& /*system*/, FieldId /*field_id*/) {}

    virtual void contributeToResidual(FormExpr& residual,
                                      const FormExpr& u,
                                      const FormExpr& v) const = 0;

    [[nodiscard]] virtual std::vector<StrongDirichlet> getStrongConstraints(FieldId field_id) const = 0;

    /**
     * @brief Optional hook for general affine constraints (setup-time)
     *
     * Allows boundary conditions to add algebraic constraints that are more
     * general than simple strong Dirichlet declarations (e.g., periodicity,
     * multi-point constraints).
     *
     * This method is invoked during `FESystem::setup()` via a systems-side
     * wrapper installed by BoundaryConditionManager.
     */
    virtual void addAffineConstraints(constraints::AffineConstraints& /*constraints*/,
                                      FieldId /*field_id*/) const
    {
    }

    /**
     * @brief Produce rich mathematical descriptors for this BC
     *
     * Returns one or more BoundaryConditionDescriptor objects describing
     * what this BC prescribes, how it's enforced, and what it anchors.
     * Consumed by the Analysis subsystem for nullspace, constraint rank,
     * compatibility, and operator structure analysis.
     *
     * @param field_id  The field this BC is applied to
     * @param system    Optional FESystem for coupled BCs that need system context
     * @return          Descriptors (empty = no analysis metadata available)
     */
    [[nodiscard]] virtual std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId /*field_id*/,
                     const systems::FESystem* /*system*/ = nullptr) const
    {
        return {};
    }
};

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_BOUNDARY_CONDITION_H
