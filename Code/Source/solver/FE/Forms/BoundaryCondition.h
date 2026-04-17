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

#include <string>
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
    [[nodiscard]] virtual int interfaceMarker() const { return -1; }

    [[nodiscard]] virtual analysis::DomainKind targetDomain() const
    {
        return interfaceMarker() >= 0 ? analysis::DomainKind::InterfaceFace
                                      : analysis::DomainKind::Boundary;
    }

    [[nodiscard]] virtual int targetMarker() const
    {
        return targetDomain() == analysis::DomainKind::InterfaceFace
                   ? interfaceMarker()
                   : boundaryMarker();
    }

    [[nodiscard]] std::string targetDescription() const
    {
        const int marker = targetMarker();
        if (targetDomain() == analysis::DomainKind::InterfaceFace) {
            return marker >= 0 ? ("interface_marker " + std::to_string(marker))
                               : "global interface relation";
        }
        return marker >= 0 ? ("boundary_marker " + std::to_string(marker))
                           : "global boundary relation";
    }

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

    /**
     * @brief Optional hook for strong non-pointwise system constraints
     *
     * Allows a boundary condition to install setup-time constraints that are
     * not representable as component-wise pointwise StrongDirichlet
     * declarations, such as H(div) normal-trace essential conditions.
     *
     * This hook is invoked by BoundaryConditionManager after setup() and
     * before metadata collection / residual contribution.
     */
    virtual void installSystemConstraints(systems::FESystem& /*system*/,
                                          FieldId /*field_id*/) const
    {
    }

    /**
     * @brief Whether this BC contributes weak terms to the residual
     *
     * Returns true if contributeToResidual() modifies the residual expression
     * (Neumann, Robin, Nitsche, etc.). Returns false for purely strong/algebraic
     * BCs (Dirichlet, periodic, multi-point constraints) whose
     * contributeToResidual() is a no-op.
     *
     * Used by BoundaryConditionManager::applyAll(system, field_id) to reject
     * weak BCs before any system mutation occurs.
     *
     * The default returns true (conservative). Override to false in subclasses
     * whose contributeToResidual() is guaranteed to be a no-op.
     */
    [[nodiscard]] virtual bool hasWeakTerms() const { return true; }

    /**
     * @brief Whether another condition may target the same marker/domain
     *
     * Weak conditions are composable by default. Strong and affine conditions
     * remain exclusive unless a subclass explicitly relaxes that rule.
     */
    [[nodiscard]] virtual bool allowsMarkerSharing() const { return hasWeakTerms(); }

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
