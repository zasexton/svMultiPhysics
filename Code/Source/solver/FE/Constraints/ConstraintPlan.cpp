/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/ConstraintPlan.h"

#include "Assembly/Assembler.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/BoundaryDofOwnerCompletion.h"
#include "Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "Constraints/SupportRestrictionConstraint.h"
#include "Forms/FormExpr.h"
#include "Forms/IntegrationDomain.h"
#include "LevelSet/LevelSetGeometry.h"
#include "Systems/BoundaryConditionManager.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace constraints {

namespace {

[[nodiscard]] std::size_t phaseIndex(ConstraintPhase phase) noexcept
{
    return static_cast<std::size_t>(phase);
}

[[nodiscard]] LevelSetConstraintSide levelSetConstraintSide(forms::CutVolumeSide side) noexcept
{
    return side == forms::CutVolumeSide::Positive ? LevelSetConstraintSide::Positive
                                                  : LevelSetConstraintSide::Negative;
}

[[nodiscard]] geometry::CutIntegrationSide cutIntegrationSide(forms::CutVolumeSide side) noexcept
{
    return side == forms::CutVolumeSide::Positive ? geometry::CutIntegrationSide::Positive
                                                  : geometry::CutIntegrationSide::Negative;
}

/// Pins the degrees of freedom of a field that no cell of the subdomain
/// (domain id) touches: the field lives on the subdomain.
class CellDomainSupportConstraint final : public ISystemConstraint {
public:
    CellDomainSupportConstraint(FieldId field, int domain_id)
        : field_(field)
        , domain_id_(domain_id)
    {
    }

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override
    {
        const auto& dof_map = system.fieldDofHandler(field_).getDofMap();
        const GlobalIndex offset = system.fieldDofOffset(field_);
        const GlobalIndex n_field_dofs = system.fieldDofHandler(field_).getNumDofs();
        const auto& mesh = system.meshAccess();
        std::vector<unsigned char> on_support(static_cast<std::size_t>(n_field_dofs), 0u);
        for (GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
            if (mesh.getCellDomainId(cell) != domain_id_) {
                continue;
            }
            for (const auto local_dof : dof_map.getCellDofs(cell)) {
                if (local_dof >= 0 && local_dof < n_field_dofs) {
                    on_support[static_cast<std::size_t>(local_dof)] = 1u;
                }
            }
        }
        // Owned degrees of freedom on subdomain cells that only other ranks
        // hold come from those ranks.
        std::vector<GlobalIndex> support;
        for (GlobalIndex d = 0; d < n_field_dofs; ++d) {
            if (on_support[static_cast<std::size_t>(d)] != 0u) {
                support.push_back(offset + d);
            }
        }
        std::vector<Real> no_payload;
        (void)completeOwnedBoundaryDofs(system, support, no_payload, 0u, {});
        for (const auto dof : support) {
            const auto local = dof - offset;
            if (local >= 0 && local < n_field_dofs) {
                on_support[static_cast<std::size_t>(local)] = 1u;
            }
        }
        const auto& owned = system.dofHandler().getPartition().locallyOwned();
        for (GlobalIndex d = 0; d < n_field_dofs; ++d) {
            const GlobalIndex dof = offset + d;
            if (on_support[static_cast<std::size_t>(d)] != 0u || !owned.contains(dof) ||
                constraints.isConstrained(dof)) {
                continue;
            }
            constraints.addDirichlet(dof, Real{0.0});
        }
    }

    bool updateValues(const systems::FESystem&, AffineConstraints&, double, double) override { return false; }
    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }
    [[nodiscard]] systems::SetupStorageRequirements storageRequirements() const noexcept override
    {
        return {};
    }
    [[nodiscard]] std::string manifestDescription() const override
    {
        return "CellDomainSupportConstraint field=" + std::to_string(field_) +
               " domain=" + std::to_string(domain_id_);
    }

private:
    FieldId field_{INVALID_FIELD_ID};
    int domain_id_{-1};
};

} // namespace

const char* constraintPhaseName(ConstraintPhase phase) noexcept
{
    switch (phase) {
    case ConstraintPhase::VertexValues:
        return "VertexValues";
    case ConstraintPhase::Essential:
        return "Essential";
    case ConstraintPhase::InactiveSupport:
        return "InactiveSupport";
    case ConstraintPhase::Aggregation:
        return "Aggregation";
    case ConstraintPhase::PointLaw:
        return "PointLaw";
    }
    return "Unknown";
}

ConstraintPlan::ConstraintPlan(systems::FESystem& system)
    : system_(&system)
{
}

ConstraintPlan::~ConstraintPlan() = default;

void ConstraintPlan::requireNotPassed(ConstraintPhase phase, const char* operation) const
{
    if (phaseIndex(phase) < phaseIndex(phase_)) {
        throw std::logic_error(
            std::string("ConstraintPlan::") + operation + ": phase " + constraintPhaseName(phase) +
            " comes before the phase the plan has reached (" + constraintPhaseName(phase_) +
            "); constraints are registered in the order VertexValues < Essential < "
            "InactiveSupport < Aggregation < PointLaw");
    }
}

ConstraintPlan& ConstraintPlan::declare(ConstraintPhase phase, std::unique_ptr<ISystemConstraint> constraint)
{
    if (!constraint) {
        throw std::invalid_argument("ConstraintPlan::declare: null constraint");
    }
    requireNotPassed(phase, "declare");
    held_[phaseIndex(phase)].push_back(std::move(constraint));
    return *this;
}

ConstraintPlan& ConstraintPlan::vertexValues(FieldId field,
                                             std::vector<VertexDirichletValue> values,
                                             VertexIdMode mode)
{
    return declare(ConstraintPhase::VertexValues,
                   std::make_unique<VertexDirichletConstraint>(field, std::move(values), mode));
}

ConstraintPlan& ConstraintPlan::essentials(const std::function<void(systems::FESystem&)>& install)
{
    if (!install) {
        throw std::invalid_argument("ConstraintPlan::essentials: empty installer");
    }
    requireNotPassed(ConstraintPhase::Essential, "essentials");
    commitThrough(ConstraintPhase::Essential);
    install(*system_);
    return *this;
}

ConstraintPlan& ConstraintPlan::essentials(systems::BoundaryConditionManager& manager,
                                           forms::FormExpr& residual,
                                           const forms::FormExpr& state,
                                           const forms::FormExpr& test,
                                           FieldId field,
                                           std::string_view operator_tag)
{
    return essentials([&](systems::FESystem& system) {
        manager.applyAll(system, residual, state, test, field, operator_tag);
    });
}

ConstraintPlan& ConstraintPlan::essentials(systems::BoundaryConditionManager& manager, FieldId field)
{
    return essentials([&](systems::FESystem& system) { manager.applyAll(system, field); });
}

ConstraintPlan& ConstraintPlan::essentials(std::span<const forms::bc::StrongDirichlet> declarations)
{
    return essentials([&](systems::FESystem& system) { systems::installStrongDirichlet(system, declarations); });
}

ConstraintPlan& ConstraintPlan::inactiveSide(std::initializer_list<FieldId> fields,
                                             const level_set::CutSide& active_side,
                                             Real value)
{
    requireNotPassed(ConstraintPhase::InactiveSupport, "inactiveSide");
    const auto& geometry = active_side.geometry();
    const auto& level_set_name = system_->fieldRecord(geometry.levelSetField()).name;
    for (const auto field : fields) {
        declare(ConstraintPhase::InactiveSupport,
                std::make_unique<LevelSetActiveSideVertexDirichletConstraint>(
                    field,
                    level_set_name,
                    levelSetConstraintSide(active_side.side()),
                    geometry.isovalue(),
                    value,
                    geometry.interfaceMarker()));
    }
    return *this;
}

ConstraintPlan& ConstraintPlan::restrictSupport(std::initializer_list<FieldId> fields,
                                                const forms::ExteriorBoundaryMeasure& domain)
{
    requireNotPassed(ConstraintPhase::InactiveSupport, "restrictSupport");
    for (const auto field : fields) {
        declare(ConstraintPhase::InactiveSupport, std::make_unique<SupportRestrictionConstraint>(field, domain));
    }
    return *this;
}

ConstraintPlan& ConstraintPlan::restrictSupport(std::initializer_list<FieldId> fields,
                                                const forms::InterfaceDomain& domain)
{
    requireNotPassed(ConstraintPhase::InactiveSupport, "restrictSupport");
    for (const auto field : fields) {
        declare(ConstraintPhase::InactiveSupport, std::make_unique<SupportRestrictionConstraint>(field, domain));
    }
    return *this;
}

ConstraintPlan& ConstraintPlan::restrictSupport(std::initializer_list<FieldId> fields,
                                                const level_set::LevelSetGeometry& geometry)
{
    requireNotPassed(ConstraintPhase::InactiveSupport, "restrictSupport");
    const auto& level_set_name = system_->fieldRecord(geometry.levelSetField()).name;
    for (const auto field : fields) {
        declare(ConstraintPhase::InactiveSupport,
                std::make_unique<SupportRestrictionConstraint>(field, geometry.interfaceMarker(), level_set_name));
    }
    return *this;
}

ConstraintPlan& ConstraintPlan::aggregate(std::initializer_list<FieldId> fields,
                                          const level_set::CutSide& active_side,
                                          SmallCutAggregationGuardOptions guards)
{
    requireNotPassed(ConstraintPhase::Aggregation, "aggregate");
    for (const auto field : fields) {
        declare(ConstraintPhase::Aggregation,
                std::make_unique<SmallCutAggregationConstraint>(
                    field,
                    cutIntegrationSide(active_side.side()),
                    active_side.geometry().interfaceMarker(),
                    std::vector<int>{},
                    std::vector<GlobalIndex>{},
                    guards));
    }
    return *this;
}

ConstraintPlan& ConstraintPlan::activeOn(std::initializer_list<FieldId> fields,
                                         const level_set::CutSide& active_side,
                                         ActiveSideOptions options)
{
    requireNotPassed(ConstraintPhase::InactiveSupport, "activeOn");
    inactiveSide(fields, active_side, options.inactive_value);
    if (options.aggregate) {
        aggregate(fields, active_side, std::move(options.guards));
        (void)active_side.geometry().requireBothSidesRetained();
    }
    return *this;
}

ConstraintPlan& ConstraintPlan::activeOn(std::initializer_list<FieldId> fields,
                                         const forms::VolumeIntegrationDomain& subdomain)
{
    requireNotPassed(ConstraintPhase::InactiveSupport, "activeOn");
    const auto domain_id = subdomain.domainId();
    if (!domain_id) {
        throw std::invalid_argument(
            "ConstraintPlan::activeOn: a fitted subdomain is a VolumeDomain::cells() domain");
    }
    for (const auto field : fields) {
        declare(ConstraintPhase::InactiveSupport, std::make_unique<CellDomainSupportConstraint>(field, *domain_id));
    }
    return *this;
}

ConstraintPlan& ConstraintPlan::contactPointVelocity(FieldId velocity, ContactPointVelocityLaw law)
{
    return declare(ConstraintPhase::PointLaw,
                   std::make_unique<ContactPointVelocityConstraint>(velocity, std::move(law)));
}

ConstraintPlan& ConstraintPlan::contactPointVelocity(FieldId velocity,
                                                     const level_set::LevelSetGeometry& geometry,
                                                     int wall_marker,
                                                     const std::array<Real, 3>& wall_normal,
                                                     std::function<Real(Real)> contact_speed,
                                                     std::string name)
{
    ContactPointVelocityLaw law;
    law.wall_boundary_markers = {wall_marker};
    law.wall_normal = wall_normal;
    law.level_set_field_name = system_->fieldRecord(geometry.levelSetField()).name;
    law.isovalue = geometry.isovalue();
    law.contact_speed = std::move(contact_speed);
    law.name = std::move(name);
    return contactPointVelocity(velocity, std::move(law));
}

ConstraintPlan& ConstraintPlan::anchorGauge(const std::string& operator_tag,
                                            const InterfaceMeasureGaugeAnchor& anchor)
{
    anchorGaugeOnInterfaceMeasure(*system_, operator_tag, anchor);
    return *this;
}

ConstraintPlan& ConstraintPlan::commitThrough(ConstraintPhase through)
{
    requireNotPassed(through, "commitThrough");
    for (std::size_t phase = 0; phase <= phaseIndex(through); ++phase) {
        auto& held = held_[phase];
        for (auto& constraint : held) {
            system_->addSystemConstraint(std::move(constraint));
            ++registered_;
        }
        held.clear();
    }
    phase_ = through;
    return *this;
}

void ConstraintPlan::commit()
{
    commitThrough(ConstraintPhase::PointLaw);
}

std::size_t ConstraintPlan::pendingCount() const noexcept
{
    std::size_t count = 0u;
    for (const auto& held : held_) {
        count += held.size();
    }
    return count;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
