/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_SUPPORTRESTRICTIONCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_SUPPORTRESTRICTIONCONSTRAINT_H

/**
 * @file SupportRestrictionConstraint.h
 * @brief A volume field restricted to a boundary or an interface (a trace multiplier)
 *
 * A trace multiplier can be written as an ordinary volume field whose degrees
 * of freedom off the trace are pinned (D54, T1): a test function that
 * vanishes on the domain gives no equation, so this constraint sets those
 * degrees of freedom to 0 and the multiplier lives on the trace.  The support
 * is the set of degrees of freedom whose basis function is nonzero on the
 * domain:
 *
 * - a physical boundary, `BoundaryDomain::fullPhysical(b)`: the degrees of
 *   freedom on the closure of the boundary faces labelled b;
 * - a fitted interface, an `InterfaceDomain` whose marker has a registered
 *   InterfaceMesh: the degrees of freedom on the closure of its faces, from the
 *   cells on both sides;
 * - a generated interface, the interface of a level-set geometry: every
 *   degree of freedom of the cells its rules cut, read from the installed cut
 *   integration context.  A new cut epoch installs a new context, which
 *   rebuilds the constraint state, so the pinned set follows the cut without a
 *   new setup.
 *
 * A degree of freedom another constraint already fixes keeps that constraint
 * (per-DOF precedence, as for every constraint of the InactiveSupport phase).
 * The field must be an H1 (C0) scalar or Product H1 vector field; a cell type
 * whose face degrees of freedom FE cannot tabulate is refused by name.
 */

#include "Constraints/SystemConstraint.h"
#include "Core/Types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace svmp {
namespace FE {

namespace forms {
class ExteriorBoundaryMeasure;
class InterfaceDomain;
} // namespace forms

namespace constraints {

/// The kinds of domain a support restriction takes.
enum class SupportDomainKind : std::uint8_t {
    PhysicalBoundary,
    FittedInterface,
    GeneratedInterface,
};

[[nodiscard]] const char* supportDomainKindName(SupportDomainKind kind) noexcept;

class SupportRestrictionConstraint final : public ISystemConstraint {
public:
    /// The field restricted to the closure of a physical boundary
    /// (`BoundaryDomain::fullPhysical(b)`; a generated active subset is refused).
    SupportRestrictionConstraint(FieldId field, const forms::ExteriorBoundaryMeasure& domain);

    /// The field restricted to the closure of a fitted interface (the
    /// InterfaceMesh registered at the domain's marker).
    SupportRestrictionConstraint(FieldId field, const forms::InterfaceDomain& domain);

    /// The field restricted to the cells a generated interface cuts: the
    /// interface @p interface_marker of the level set @p level_set_field_name.
    SupportRestrictionConstraint(FieldId field, int interface_marker, std::string level_set_field_name);

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override;

    /// The pinned values are 0 at any time; a change of the support is a
    /// structural rebuild (dependencyDeclaration), never a value update.
    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }

    [[nodiscard]] ConstraintDependencyDeclaration dependencyDeclaration() const override;

    [[nodiscard]] systems::SetupStorageRequirements storageRequirements() const noexcept override;

    [[nodiscard]] std::string manifestDescription() const override;

    [[nodiscard]] FieldId field() const noexcept { return field_; }
    [[nodiscard]] SupportDomainKind domainKind() const noexcept { return kind_; }
    /// The physical boundary marker, or the interface marker.
    [[nodiscard]] int marker() const noexcept { return marker_; }
    [[nodiscard]] const std::string& levelSetFieldName() const noexcept { return level_set_field_name_; }

    /// Owned degrees of freedom of the field on the support at the last
    /// apply(), and those it pinned (on this rank).
    [[nodiscard]] std::size_t ownedSupportDofs() const noexcept { return owned_support_dofs_; }
    [[nodiscard]] std::size_t pinnedDofs() const noexcept { return pinned_dofs_; }

private:
    [[nodiscard]] std::string context(const systems::FESystem& system) const;
    /// Visits every cell of the support with the local face of the domain in
    /// it (or -1: the whole cell, for a generated interface).
    void forEachSupportEntity(const systems::FESystem& system,
                              const std::function<void(GlobalIndex cell, int local_face)>& visit) const;

    FieldId field_{INVALID_FIELD_ID};
    SupportDomainKind kind_{SupportDomainKind::PhysicalBoundary};
    int marker_{-1};
    std::string level_set_field_name_{};
    std::size_t owned_support_dofs_{0u};
    std::size_t pinned_dofs_{0u};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_SUPPORTRESTRICTIONCONSTRAINT_H
